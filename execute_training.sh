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
NUM_EPISODES=1000
MAX_STEPS=500
BATCH_SIZE=256
LEARNING_RATE="3e-4"
BUFFER_SIZE=50000
HIDDEN_DIM=256
OUTPUT_DIR="output/"
CHECKPOINT_INTERVAL=50
GRADIENT_STEPS=6
NO_TENSORBOARD="false"
LOAD_MODEL=""
CONFIG_PATH=""
STATE_WAIT_TIMEOUT_SEC="5.0"
STATE_WAIT_POLL_SEC="0.02"
CUDA_MODE="auto"

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
  --gradient-steps N            Gradient updates per env step (default: ${GRADIENT_STEPS})
  --no-tensorboard              Disable TensorBoard
  --load-model PATH             Load checkpoint before training
  --config PATH                 Load config JSON
  --state-wait-timeout SEC      Wait timeout inside env reset (default: ${STATE_WAIT_TIMEOUT_SEC})
  --state-wait-poll SEC         Poll interval inside env reset (default: ${STATE_WAIT_POLL_SEC})
  --cuda auto|off               CUDA mode for PyTorch (default: ${CUDA_MODE})

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
        --gradient-steps) GRADIENT_STEPS="$2"; shift 2 ;;
        --no-tensorboard) NO_TENSORBOARD="true"; shift 1 ;;
        --load-model) LOAD_MODEL="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --state-wait-timeout) STATE_WAIT_TIMEOUT_SEC="$2"; shift 2 ;;
        --state-wait-poll) STATE_WAIT_POLL_SEC="$2"; shift 2 ;;
        --cuda) CUDA_MODE="$2"; shift 2 ;;

        -h|--help) print_help; exit 0 ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

if [[ "$CUDA_MODE" != "auto" && "$CUDA_MODE" != "off" ]]; then
    echo -e "${RED}Invalid --cuda value:${NC} $CUDA_MODE"
    echo -e "${YELLOW}Allowed values:${NC} auto, off"
    exit 1
fi

echo -e "${CYAN}=== Franka ROS2 RL Training Setup ===${NC}"

topic_exists() {
    local topic="$1"
    docker exec -i "$CONTAINER_NAME" "$WRAPPER" ros2 topic list 2>/dev/null | grep -Fx "$topic" >/dev/null
}

find_robot_state_topic() {
    docker exec -i "$CONTAINER_NAME" "$WRAPPER" ros2 topic list 2>/dev/null | \
        grep -E '^(/[^ ]+)?/franka_robot_state_broadcaster/robot_state$' | \
        head -n 1
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

        if (( elapsed % 5 == 0 )); then
            echo -e "${CYAN}Waiting for topic:${NC} $topic (${elapsed}s/${timeout}s)"
        fi

        sleep 1
        elapsed=$((elapsed + 1))
    done

    echo -e "${RED}Timeout waiting for topic:${NC} $topic"
    return 1
}

wait_for_robot_state_topic() {
    local timeout="$1"
    local elapsed=0

    while (( elapsed < timeout )); do
        local discovered_topic
        discovered_topic="$(find_robot_state_topic || true)"

        if [[ -n "$discovered_topic" ]]; then
            ROBOT_STATE_TOPIC="$discovered_topic"
            echo -e "${GREEN}Robot state topic available:${NC} $ROBOT_STATE_TOPIC"
            return 0
        fi

        if (( elapsed % 5 == 0 )); then
            echo -e "${CYAN}Waiting for robot state topic matching:${NC} */franka_robot_state_broadcaster/robot_state (${elapsed}s/${timeout}s)"
        fi

        sleep 1
        elapsed=$((elapsed + 1))
    done

    return 1
}

require_ros_executable() {
    local pkg="$1"
    local exe="$2"

    if ! docker exec -i "$CONTAINER_NAME" "$WRAPPER" ros2 pkg executables "$pkg" 2>/dev/null | grep -Fx "$pkg $exe" >/dev/null; then
        echo -e "${RED}Missing ROS2 executable:${NC} $pkg $exe"
        echo -e "${YELLOW}Hint:${NC} build and source the workspace, e.g. colcon build --packages-select $pkg"
        return 1
    fi
}

cleanup() {
    if [[ "${CLEANUP_DONE:-false}" == "true" ]]; then
        return
    fi
    CLEANUP_DONE="true"

    echo -e "\n${YELLOW}Stopping training stack...${NC}"

    docker exec "$CONTAINER_NAME" pkill -SIGINT -f "ros_gz_bridge" 2>/dev/null || true
    docker exec "$CONTAINER_NAME" pkill -SIGINT -f "ros2 launch franka_gazebo_bringup" 2>/dev/null || true
    docker exec "$CONTAINER_NAME" pkill -SIGINT -f "ros2 run distance_based_rl train_agent" 2>/dev/null || true

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

trap cleanup EXIT SIGINT SIGTERM

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
    rviz:="$RVIZ" \
    gz_args:="-s -r empty.sdf" &
LAUNCH_PID=$!

echo -e "${CYAN}Waiting for ROS topics required by training...${NC}"
if ! wait_for_robot_state_topic "$WAIT_TIMEOUT_SEC"; then
    echo -e "${YELLOW}Warning:${NC} no matching robot state topic found for */franka_robot_state_broadcaster/robot_state."
    echo -e "${YELLOW}Training will still start; environment may use default state until messages arrive.${NC}"
fi

if ! wait_for_topic "/${CONTROLLER}/commands" "$WAIT_TIMEOUT_SEC"; then
    echo -e "${YELLOW}Warning:${NC} /${CONTROLLER}/commands not available."
    echo -e "${YELLOW}Training will start anyway; check controller_manager status if actions have no effect.${NC}"
fi

# 4) Build training command with defaults + optional args
ROBOT_STATE_TOPIC="${ROBOT_STATE_TOPIC:-/franka_robot_state_broadcaster/robot_state}"
echo -e "${CYAN}Using robot state topic:${NC} $ROBOT_STATE_TOPIC"

DOCKER_ENV_ARGS=(
    -e PYTHONUNBUFFERED=1
    -e FRANKA_ROBOT_STATE_TOPIC="$ROBOT_STATE_TOPIC"
    -e FRANKA_STATE_WAIT_TIMEOUT_SEC="$STATE_WAIT_TIMEOUT_SEC"
    -e FRANKA_STATE_WAIT_POLL_SEC="$STATE_WAIT_POLL_SEC"
    -e MAX_STEPS_PER_EPISODE="$MAX_STEPS"
)

if [[ "$CUDA_MODE" == "off" ]]; then
    DOCKER_ENV_ARGS+=(-e CUDA_VISIBLE_DEVICES="")
fi

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
    --gradient-steps "$GRADIENT_STEPS"
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

require_ros_executable "distance_based_rl" "train_agent"

echo -e "${CYAN}4. Starting training...${NC}"
echo -e "${GREEN}Press Ctrl+C to stop training and cleanup.${NC}"
echo -e "${CYAN}Training command:${NC} ${TRAIN_CMD[*]}"

set +e
docker exec -it "${DOCKER_ENV_ARGS[@]}" "$CONTAINER_NAME" "$WRAPPER" "${TRAIN_CMD[@]}"
STATUS=$?
set -e

if [[ "$STATUS" -ne 0 ]]; then
    echo -e "${RED}Training process exited with status:${NC} $STATUS"
else
    echo -e "${GREEN}Training process completed successfully.${NC}"
fi

exit "$STATUS"
