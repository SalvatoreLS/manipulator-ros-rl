#!/bin/bash
set -euo pipefail

# Defaults aligned with distance_based_rl/agent/train.py
WORKDIR="/ros2_rl_ws"
WRAPPER="/entrypoint.sh"
CONTROLLER="forward_position_controller"
RVIZ="false"
WAIT_TIMEOUT_SEC=40

# Training defaults
NUM_EPISODES=1000
MAX_STEPS=500
BATCH_SIZE=256
LEARNING_RATE="3e-4"
BUFFER_SIZE=50000
HIDDEN_DIM=256
OUTPUT_DIR="output/"
CHECKPOINT_INTERVAL=50
GRADIENT_STEPS=1
WARMUP_STEPS=1000
NO_TENSORBOARD="false"
LOAD_MODEL=""
CONFIG_PATH=""
STATE_WAIT_TIMEOUT_SEC="5.0"
STATE_WAIT_POLL_SEC="0.02"
CUDA_MODE="auto"
SEED=""
AGENT="sac"            # sac = from-scratch implementation, sb3 = Stable-Baselines3 baseline
DOMAIN_ID=""           # ROS_DOMAIN_ID; isolates concurrent runs from each other
BASELINE_STEPS=""      # total env steps for the sb3 baseline

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_help() {
    cat <<EOF
Usage: ./execute_training_docker.sh [options]

Docker/ROS options:
  --controller NAME             Controller for launch (default: ${CONTROLLER})
  --rviz true|false             Enable RViz in launch (default: ${RVIZ})
  --wait-timeout SEC            Timeout waiting ROS topics (default: ${WAIT_TIMEOUT_SEC})

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
  --warmup-steps N              Random-action steps before the policy takes over (default: ${WARMUP_STEPS})
  --no-tensorboard              Disable TensorBoard
  --load-model PATH             Load checkpoint before training
  --config PATH                 Load config JSON
  --state-wait-timeout SEC      Wait timeout inside env reset (default: ${STATE_WAIT_TIMEOUT_SEC})
  --state-wait-poll SEC         Poll interval inside env reset (default: ${STATE_WAIT_POLL_SEC})
  --cuda auto|off               CUDA mode for PyTorch (default: ${CUDA_MODE})
  --seed N                      Master seed, makes the run reproducible (default: unseeded)
  --agent sac|sb3               sac = from-scratch agent, sb3 = Stable-Baselines3 baseline (default: ${AGENT})
  --baseline-steps N            Total env steps for --agent sb3 (default: num-episodes * max-steps)
  --domain-id N                 ROS_DOMAIN_ID (0-101). Required to run several trainings
                                concurrently, so their Gazebo/DDS graphs stay separate.

Other:
  -h, --help                    Show this help

Examples:
    ./execute_training_docker.sh
    ./execute_training_docker.sh --num-episodes 200 --max-steps 300
    ./execute_training_docker.sh --no-tensorboard --output-dir output/run_01
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --controller) CONTROLLER="$2"; shift 2 ;;
        --rviz) RVIZ="$2"; shift 2 ;;
        --wait-timeout) WAIT_TIMEOUT_SEC="$2"; shift 2 ;;

        --num-episodes) NUM_EPISODES="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
        --buffer-size) BUFFER_SIZE="$2"; shift 2 ;;
        --hidden-dim) HIDDEN_DIM="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --checkpoint-interval) CHECKPOINT_INTERVAL="$2"; shift 2 ;;
        --gradient-steps) GRADIENT_STEPS="$2"; shift 2 ;;
        --warmup-steps) WARMUP_STEPS="$2"; shift 2 ;;
        --no-tensorboard) NO_TENSORBOARD="true"; shift 1 ;;
        --load-model) LOAD_MODEL="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --state-wait-timeout) STATE_WAIT_TIMEOUT_SEC="$2"; shift 2 ;;
        --state-wait-poll) STATE_WAIT_POLL_SEC="$2"; shift 2 ;;
        --cuda) CUDA_MODE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --agent) AGENT="$2"; shift 2 ;;
        --baseline-steps) BASELINE_STEPS="$2"; shift 2 ;;
        --domain-id) DOMAIN_ID="$2"; shift 2 ;;

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

if [[ "$AGENT" != "sac" && "$AGENT" != "sb3" ]]; then
    echo -e "${RED}Invalid --agent value:${NC} $AGENT"
    echo -e "${YELLOW}Allowed values:${NC} sac, sb3"
    exit 1
fi

: "${BASELINE_STEPS:=$((NUM_EPISODES * MAX_STEPS))}"

if [[ -n "$DOMAIN_ID" ]]; then
    if ! [[ "$DOMAIN_ID" =~ ^[0-9]+$ ]] || (( DOMAIN_ID > 101 )); then
        echo -e "${RED}Invalid --domain-id:${NC} $DOMAIN_ID (expected 0-101)"
        exit 1
    fi
    # Exported before anything ROS starts, so the bridge, the simulation and the
    # trainer all join the same isolated DDS domain.
    export ROS_DOMAIN_ID="$DOMAIN_ID"
    # Gazebo transport has its own discovery, independent of DDS: without a distinct
    # partition two concurrent simulations would share gz topics (both publish
    # /world/default/model/fr3/joint_state) and the bridges would cross-feed.
    export GZ_PARTITION="rl${DOMAIN_ID}"
    export IGN_PARTITION="rl${DOMAIN_ID}"   # Fortress still reads the IGN_ name
    echo -e "${CYAN}ROS_DOMAIN_ID:${NC} $ROS_DOMAIN_ID  ${CYAN}GZ_PARTITION:${NC} $GZ_PARTITION"
fi

echo -e "${CYAN}=== Franka ROS2 RL Training Setup ===${NC}"

topic_exists() {
    local topic="$1"
    ros2 topic list 2>/dev/null | grep -Fx "$topic" >/dev/null
}

find_robot_state_topic() {
    ros2 topic list 2>/dev/null | \
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

    if ! ros2 pkg executables "$pkg" 2>/dev/null | grep -Fx "$pkg $exe" >/dev/null; then
        echo -e "${RED}Missing ROS2 executable:${NC} $pkg $exe"
        echo -e "${YELLOW}Hint:${NC} build and source the workspace, e.g. colcon build --packages-select $pkg"
        return 1
    fi
}

# This script's own process group, so cleanup() can refuse to signal it.
SELF_PGID="$(ps -o pgid= -p $$ | tr -d ' ')"

# Start a command in its own process group and report that group's id, so the whole
# process tree it spawns can be signalled as a unit later.  setsid execs the command
# directly when the shell is non-interactive (no job control), so the pid it reports
# is the group leader; resolving the pgid from /proc rather than assuming pgid == pid
# keeps this correct either way.
start_in_own_group() {
    setsid "$@" &
    local pid=$!
    local pgid=""
    local i
    # Wait for the child to actually land in its OWN group.  Reading the pgid too early
    # returns this shell's group — setsid() has not run yet — and cleanup would then
    # skip it via the self-guard, silently leaking the whole tree.  Retry until the pgid
    # differs from ours; fall back to the pid, which is what setsid makes it anyway.
    for i in 1 2 3 4 5 6 7 8 9 10; do
        pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
        [[ -n "$pgid" && "$pgid" != "$SELF_PGID" ]] && break
        sleep 0.1
    done
    if [[ -z "$pgid" || "$pgid" == "$SELF_PGID" ]]; then
        echo -e "${YELLOW}Warning:${NC} could not resolve a private process group for pid ${pid}; using pid as pgid." >&2
        pgid="$pid"
    fi
    STARTED_PID="$pid"
    STARTED_PGID="${pgid:-$pid}"
}

cleanup() {
    if [[ "${CLEANUP_DONE:-false}" == "true" ]]; then
        return
    fi
    CLEANUP_DONE="true"

    echo -e "\n${YELLOW}Stopping training stack...${NC}"

    # Kill only the processes THIS invocation started, by process GROUP — never a
    # global pkill on the process name.  Several of these scripts may run concurrently
    # on different ROS_DOMAIN_IDs (see scripts/run_experiments.sh), and a name-matching
    # pkill would tear down the other runs' simulations too.
    #
    # Signalling the group (rather than the PID, or the PID's direct children) is what
    # makes this complete: `ros2 launch` starts Gazebo as a `sh -c ruby /usr/bin/ign
    # gazebo` wrapper, so the simulator is a *grandchild*.  Killing LAUNCH_PID alone
    # left those reparented to init, and every run leaked an idle Gazebo that kept
    # burning cores for the rest of the sweep.  Both processes are started under
    # `setsid`, so each owns a process group containing its whole tree.
    # SIGTERM, not SIGINT.  A non-interactive shell starts background jobs with
    # SIGINT and SIGQUIT set to SIG_IGN, and that disposition survives exec — so every
    # process in these trees is literally incapable of dying from the SIGINT that the
    # old cleanup (and the launch file's shutdown handler) sent it.  Verified:
    # /proc/<pid>/status reports SigIgn: 0000000000000006 for the simulator.
    # SIGTERM is not masked, and `ros2 launch` handles it as a graceful shutdown.
    local pgid remaining
    for pgid in "${LAUNCH_PGID:-}" "${BRIDGE_PGID:-}"; do
        [[ -z "$pgid" ]] && continue
        # Never signal our own group: that would kill this script (and, under
        # run_experiments.sh, its siblings) instead of the run's stack.  Say so out
        # loud — a silent skip here is exactly how a leaked stack goes unnoticed.
        if [[ "$pgid" == "$SELF_PGID" ]]; then
            echo -e "${RED}Warning:${NC} refusing to signal own process group (${pgid}); a run stack may be leaking." >&2
            continue
        fi
        echo -e "  terminating process group ${pgid}"
        kill -SIGTERM -- "-$pgid" 2>/dev/null || true
    done

    sleep 3

    # Escalate: anything still standing gets SIGKILL, which nothing can mask.
    for pgid in "${LAUNCH_PGID:-}" "${BRIDGE_PGID:-}"; do
        [[ -z "$pgid" ]] && continue
        [[ "$pgid" == "$SELF_PGID" ]] && continue
        kill -SIGKILL -- "-$pgid" 2>/dev/null || true
        sleep 0.5
        # Report anything that somehow outlived SIGKILL, so a wedged run is visible
        # in the console log instead of silently eating cores.
        remaining="$(pgrep -g "$pgid" 2>/dev/null | tr '\n' ' ' || true)"
        [[ -n "${remaining// /}" ]] && \
            echo -e "${RED}Warning:${NC} PIDs still alive in group ${pgid}: ${remaining}"
    done

    echo -e "${GREEN}Cleanup complete.${NC}"
}

trap cleanup EXIT SIGINT SIGTERM

source_ros_environment() {
    echo -e "${CYAN}1. Sourcing ROS2 environment...${NC}"

    # ROS setup scripts may reference unset vars and are not nounset-safe.
    set +u

    if [[ -f "/opt/ros/humble/setup.bash" ]]; then
        source /opt/ros/humble/setup.bash
    elif [[ -f "/opt/ros/humble/setup.sh" ]]; then
        source /opt/ros/humble/setup.sh
    else
        echo -e "${RED}ROS base setup script not found under /opt/ros/humble${NC}"
        exit 1
    fi

    if [[ -f "$WORKDIR/install/setup.bash" ]]; then
        source "$WORKDIR/install/setup.bash"
    elif [[ -f "$WORKDIR/install/setup.sh" ]]; then
        source "$WORKDIR/install/setup.sh"
    else
        echo -e "${RED}Workspace setup script not found in $WORKDIR/install${NC}"
        exit 1
    fi

    set -u
}

# 1) Source ROS2 workspace
source_ros_environment

# 2) Start bridge
echo -e "${CYAN}2. Starting ROS-GZ Bridge...${NC}"
start_in_own_group ros2 run ros_gz_bridge parameter_bridge \
    --ros-args -p config_file:="$WORKDIR/config/bridge.yaml"
BRIDGE_PID="$STARTED_PID"
BRIDGE_PGID="$STARTED_PGID"

sleep 2

# 3) Start simulation
echo -e "${CYAN}3. Starting simulation (Gazebo + optional RViz)...${NC}"

# Training runs an unthrottled world (real_time_factor 0) instead of Gazebo's stock
# empty.sdf, which is pinned to real time.  The environment is bound by waiting for
# simulated motion, not by compute, so the sim clock is the main lever on run duration.
# Set FRANKA_GZ_WORLD=empty.sdf to fall back to the stock real-time world.
#
# The world ships in this repository (config/worlds/), not in the vcs-fetched
# franka_gazebo_bringup: anything dropped into src/franka_ros2/ is gitignored as
# third-party and would not survive a fresh clone.  The bringup share directory is
# still checked second, so an existing install that has it there keeps working.
GZ_WORLD="${FRANKA_GZ_WORLD:-}"
if [[ -z "$GZ_WORLD" ]]; then
    if [[ -f "$WORKDIR/config/worlds/rl_empty.sdf" ]]; then
        GZ_WORLD="$WORKDIR/config/worlds/rl_empty.sdf"
    else
        _bringup_share="$(ros2 pkg prefix franka_gazebo_bringup 2>/dev/null)/share/franka_gazebo_bringup"
        if [[ -f "${_bringup_share}/worlds/rl_empty.sdf" ]]; then
            GZ_WORLD="${_bringup_share}/worlds/rl_empty.sdf"
        else
            echo -e "${YELLOW}Warning:${NC} rl_empty.sdf not found; falling back to stock empty.sdf (real-time)."
            GZ_WORLD="empty.sdf"
        fi
    fi
fi
echo -e "${CYAN}Gazebo world:${NC} $GZ_WORLD"

start_in_own_group ros2 launch franka_gazebo_bringup gazebo_franka_arm_example_controller.launch.py \
    load_gripper:=true \
    controller:="$CONTROLLER" \
    rviz:="$RVIZ" \
    gz_args:="-s -r $GZ_WORLD"
LAUNCH_PID="$STARTED_PID"
LAUNCH_PGID="$STARTED_PGID"

echo -e "${CYAN}Waiting for ROS topics required by training...${NC}"
if ! wait_for_robot_state_topic "$WAIT_TIMEOUT_SEC"; then
    echo -e "${YELLOW}Warning:${NC} no matching robot state topic found for */franka_robot_state_broadcaster/robot_state."
    echo -e "${YELLOW}Training will still start; environment may use default state until messages arrive.${NC}"
fi

# The command topic may be missing if controller spawner reports configuration errors.
# Do not block training forever; warn and continue so train_agent can still start.
if ! wait_for_topic "/${CONTROLLER}/commands" "$WAIT_TIMEOUT_SEC"; then
    echo -e "${YELLOW}Warning:${NC} /${CONTROLLER}/commands not available."
    echo -e "${YELLOW}Training will start anyway; check controller_manager status if actions have no effect.${NC}"
fi

# 4) Build training command with defaults + optional args
ROBOT_STATE_TOPIC="${ROBOT_STATE_TOPIC:-/franka_robot_state_broadcaster/robot_state}"
echo -e "${CYAN}Using robot state topic:${NC} $ROBOT_STATE_TOPIC"

TRAIN_ENV=(
    env
    PYTHONUNBUFFERED=1
    FRANKA_ROBOT_STATE_TOPIC="$ROBOT_STATE_TOPIC"
    FRANKA_STATE_WAIT_TIMEOUT_SEC="$STATE_WAIT_TIMEOUT_SEC"
    FRANKA_STATE_WAIT_POLL_SEC="$STATE_WAIT_POLL_SEC"
    MAX_STEPS_PER_EPISODE="$MAX_STEPS"
    # Keep the numeric libraries single-threaded.  torch.set_num_threads() in train.py
    # covers torch's own pools, but OpenMP/MKL read these before torch is imported and
    # would otherwise spawn one thread per core inside every concurrent run.  The
    # simulators are the bottleneck; they need those cores more than a 256-wide MLP does.
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
)

if [[ "$CUDA_MODE" == "off" ]]; then
    TRAIN_ENV+=(CUDA_VISIBLE_DEVICES="")
fi

if [[ "$AGENT" == "sb3" ]]; then
    # Stable-Baselines3 cross-check: same env, same hyperparameters, same step budget.
    TRAIN_CMD=(
        "${TRAIN_ENV[@]}" python3 "$WORKDIR/scripts/sb3_baseline.py"
        --total-steps "$BASELINE_STEPS"
        --max-steps "$MAX_STEPS"
        --batch-size "$BATCH_SIZE"
        --learning-rate "$LEARNING_RATE"
        --buffer-size "$BUFFER_SIZE"
        --hidden-dim "$HIDDEN_DIM"
        --gradient-steps "$GRADIENT_STEPS"
        --output-dir "$OUTPUT_DIR"
    )
    if [[ -n "$SEED" ]]; then
        TRAIN_CMD+=(--seed "$SEED")
    fi
else
    TRAIN_CMD=(
        "${TRAIN_ENV[@]}" ros2 run distance_based_rl train_agent
        --num-episodes "$NUM_EPISODES"
        --max-steps "$MAX_STEPS"
        --batch-size "$BATCH_SIZE"
        --learning-rate "$LEARNING_RATE"
        --buffer-size "$BUFFER_SIZE"
        --hidden-dim "$HIDDEN_DIM"
        --output-dir "$OUTPUT_DIR"
        --checkpoint-interval "$CHECKPOINT_INTERVAL"
        --gradient-steps "$GRADIENT_STEPS"
        --warmup-steps "$WARMUP_STEPS"
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
    if [[ -n "$SEED" ]]; then
        TRAIN_CMD+=(--seed "$SEED")
    fi

    require_ros_executable "distance_based_rl" "train_agent"
fi

echo -e "${CYAN}4. Starting training...${NC}"
echo -e "${GREEN}Press Ctrl+C to stop training and cleanup.${NC}"
echo -e "${CYAN}Training command:${NC} ${TRAIN_CMD[*]}"

set +e
"${TRAIN_CMD[@]}"
STATUS=$?
set -e

if [[ "$STATUS" -ne 0 ]]; then
    echo -e "${RED}Training process exited with status:${NC} $STATUS"
else
    echo -e "${GREEN}Training process completed successfully.${NC}"
fi

exit "$STATUS"
