#!/bin/bash
set -euo pipefail

# Defaults aligned with distance_based_rl/agent/train.py
CONTAINER_NAME="franka_ros2_rl"
COMPOSE_FILE="docker-compose.yaml"
WORKDIR="/ros2_rl_ws"
WRAPPER="/entrypoint.sh"
CONTROLLER="forward_position_controller"
RVIZ="false"
WAIT_TIMEOUT_SEC=90
KEEP_CONTAINER="true"

# Training defaults
NUM_EPISODES=10
MAX_STEPS=500
BATCH_SIZE=256
LEARNING_RATE="3e-4"
BUFFER_SIZE=10000
HIDDEN_DIM=256
OUTPUT_DIR="output/"
CHECKPOINT_INTERVAL=50
NO_TENSORBOARD="false"
LOAD_MODEL=""
CONFIG_PATH=""

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_help() {
    cat <<EOF
Usage: ./execute_training.sh [options]

Docker/ROS options:
  --container-name NAME         Container name (default: ${CONTAINER_NAME})
  --compose-file FILE           docker compose file (default: ${COMPOSE_FILE})
  --controller NAME             Controller for launch (default: ${CONTROLLER})
  --rviz true|false             Enable RViz in launch (default: ${RVIZ})
  --wait-timeout SEC            Timeout waiting ROS topics (default: ${WAIT_TIMEOUT_SEC})
  --keep-container              Do not stop container on exit

Training options (forwarded to train_agent):
  --num-episodes N              (default: ${NUM_EPISODES})
  --max-steps N                 (default: ${MAX_STEPS})
  --batch-size N                (default: ${BATCH_SIZE})
  --learning-rate FLOAT         (default: ${LEARNING_RATE})
  --buffer-size N               (default: ${BUFFER_SIZE})
  --hidden-dim N                (default: ${HIDDEN_DIM})
  --output-dir PATH             (default: ${OUTPUT_DIR})
  --checkpoint-interval N       (default: ${CHECKPOINT_INTERVAL})
  --no-tensorboard              Disable TensorBoard
  --load-model PATH             Load checkpoint before training
  --config PATH                 Load config JSON

Other:
  -h, --help                    Show this help

Examples:
  ./execute_training.sh
  ./execute_training.sh --num-episodes 200 --max-steps 300
  ./execute_training.sh --no-tensorboard --output-dir output/run_01
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container-name) CONTAINER_NAME="$2"; shift 2 ;;
        --compose-file) COMPOSE_FILE="$2"; shift 2 ;;
        --controller) CONTROLLER="$2"; shift 2 ;;
        --rviz) RVIZ="$2"; shift 2 ;;
        --wait-timeout) WAIT_TIMEOUT_SEC="$2"; shift 2 ;;
        --keep-container) KEEP_CONTAINER="true"; shift 1 ;;

        --num-episodes) NUM_EPISODES="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
        --buffer-size) BUFFER_SIZE="$2"; shift 2 ;;
        --hidden-dim) HIDDEN_DIM="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --checkpoint-interval) CHECKPOINT_INTERVAL="$2"; shift 2 ;;
        --no-tensorboard) NO_TENSORBOARD="true"; shift 1 ;;
        --load-model) LOAD_MODEL="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;

        -h|--help) print_help; exit 0 ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

echo -e "${CYAN}=== Franka ROS2 RL Training Setup ===${NC}"

topic_exists() {
    local topic="$1"
    docker exec -i "$CONTAINER_NAME" "$WRAPPER" ros2 topic list 2>/dev/null | grep -Fx "$topic" >/dev/null
}

wait_for_topic() {
    local topic="$1"
    local timeout="$2"
    local elapsed=0

    while (( elapsed < timeout )); do
        if topic_exists "$topic"; then
            echo -e "${GREEN}Topic available:${NC} $topic"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    echo -e "${RED}Timeout waiting for topic:${NC} $topic"
    return 1
}

cleanup() {
    echo -e "\n${YELLOW}Stopping training stack...${NC}"

    # Stop nodes/processes inside container
    docker exec "$CONTAINER_NAME" pkill -SIGINT -f "ros_gz_bridge" 2>/dev/null || true
    docker exec "$CONTAINER_NAME" pkill -SIGINT -f "ros2 launch franka_gazebo_bringup" 2>/dev/null || true
    docker exec "$CONTAINER_NAME" pkill -SIGINT -f "ros2 run distance_based_rl train_agent" 2>/dev/null || true

    # Stop host-side docker exec wrappers
    [[ -n "${BRIDGE_PID:-}" ]] && kill "$BRIDGE_PID" 2>/dev/null || true
    [[ -n "${LAUNCH_PID:-}" ]] && kill "$LAUNCH_PID" 2>/dev/null || true

    sleep 2

    if [[ "$KEEP_CONTAINER" == "false" ]]; then
        echo -e "${CYAN}Stopping Docker container...${NC}"
        docker compose -f "$COMPOSE_FILE" stop >/dev/null
    else
        echo -e "${CYAN}Leaving Docker container running (--keep-container).${NC}"
    fi

    echo -e "${GREEN}Cleanup complete.${NC}"
}

trap cleanup SIGINT SIGTERM

# 1) Start container
echo -e "${CYAN}1. Launching Docker container...${NC}"
docker compose -f "$COMPOSE_FILE" up -d

if [[ -z "$(docker ps -q -f name="$CONTAINER_NAME")" ]]; then
    echo -e "${RED}Error: Failed to start container ${CONTAINER_NAME}.${NC}"
    exit 1
fi
echo -e "${GREEN}Container is up.${NC}"

# 2) Start bridge
echo -e "${CYAN}2. Starting ROS-GZ Bridge...${NC}"
docker exec -i "$CONTAINER_NAME" "$WRAPPER" ros2 run ros_gz_bridge parameter_bridge \
    --ros-args -p config_file:="$WORKDIR/config/bridge.yaml" &
BRIDGE_PID=$!

sleep 2

# 3) Start simulation
echo -e "${CYAN}3. Starting simulation (Gazebo + optional RViz)...${NC}"
docker exec -i "$CONTAINER_NAME" "$WRAPPER" ros2 launch franka_gazebo_bringup gazebo_franka_arm_example_controller.launch.py \
    load_gripper:=true \
    controller:="$CONTROLLER" \
    rviz:="$RVIZ" &
LAUNCH_PID=$!

echo -e "${CYAN}Waiting for ROS topics required by training...${NC}"
wait_for_topic "/franka_robot_state_broadcaster/robot_state" "$WAIT_TIMEOUT_SEC"
wait_for_topic "/forward_position_controller/commands" "$WAIT_TIMEOUT_SEC"

# 4) Build training command with defaults + optional args
TRAIN_CMD=(
    ros2 run distance_based_rl train_agent
    --num-episodes "$NUM_EPISODES"
    --max-steps "$MAX_STEPS"
    --batch-size "$BATCH_SIZE"
    --learning-rate "$LEARNING_RATE"
    --buffer-size "$BUFFER_SIZE"
    --hidden-dim "$HIDDEN_DIM"
    --output-dir "$OUTPUT_DIR"
    --checkpoint-interval "$CHECKPOINT_INTERVAL"
)

if [[ "$NO_TENSORBOARD" == "true" ]]; then
    TRAIN_CMD+=(--no-tensorboard)
fi
if [[ -n "$LOAD_MODEL" ]]; then
    TRAIN_CMD+=(--load-model "$LOAD_MODEL")
fi
if [[ -n "$CONFIG_PATH" ]]; then
    TRAIN_CMD+=(--config "$CONFIG_PATH")
fi

echo -e "${CYAN}4. Starting training...${NC}"
echo -e "${GREEN}Press Ctrl+C to stop training and cleanup.${NC}"

docker exec -it "$CONTAINER_NAME" "$WRAPPER" "${TRAIN_CMD[@]}"
STATUS=$?

cleanup
exit "$STATUS"
