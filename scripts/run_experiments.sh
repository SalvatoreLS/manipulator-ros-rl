#!/bin/bash
# Run the multi-seed evidence sweep behind docs/RESULTS.md.
#
# Each run is a full training against its own headless Gazebo, started by
# execute_training_docker.sh.  Runs land in:
#
#     output/runs/sac/seed<N>/     from-scratch SAC
#     output/runs/sb3/seed<N>/     Stable-Baselines3 cross-check
#
# Concurrent runs are isolated by ROS_DOMAIN_ID + GZ_PARTITION, so their DDS graphs
# and Gazebo transport never mix.  Per-run console output goes to <run>/console.log.
#
# Must be run INSIDE the container:   docker exec -it franka_ros2_rl bash
#
# Sizing note: the environment runs at ~2.9 steps/s (the controller round-trip
# dominates, not compute).  The defaults below are ~22.5k steps ≈ 2.1 h per run;
# with --parallel 3 the whole 6-run sweep is roughly 4-5 h.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SEEDS=(0 1 2)
NUM_EPISODES=150
MAX_STEPS=150
RUNS_DIR="output/runs"
WITH_SB3="true"
SB3_TOTAL_STEPS=""
PARALLEL=3
BASE_DOMAIN_ID=40
MIN_FREE_GB=8

print_help() {
    cat <<EOF
Usage: ./scripts/run_experiments.sh [options]

  --seeds "0 1 2"        Seeds to run (default: ${SEEDS[*]})
  --num-episodes N       Episodes per run (default: ${NUM_EPISODES})
  --max-steps N          Max steps per episode (default: ${MAX_STEPS})
  --runs-dir PATH        Output root (default: ${RUNS_DIR})
  --parallel N           Concurrent runs, each on its own ROS_DOMAIN_ID (default: ${PARALLEL})
  --base-domain-id N     First ROS_DOMAIN_ID to use (default: ${BASE_DOMAIN_ID})
  --no-sb3               Skip the Stable-Baselines3 baseline
  --sb3-steps N          Total env steps for SB3 (default: num-episodes * max-steps)
  -h, --help             Show this help

After it finishes:
  python3 scripts/plot_results.py --runs-dir ${RUNS_DIR} --out-dir docs/figures
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds) read -r -a SEEDS <<< "$2"; shift 2 ;;
        --num-episodes) NUM_EPISODES="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --runs-dir) RUNS_DIR="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --base-domain-id) BASE_DOMAIN_ID="$2"; shift 2 ;;
        --no-sb3) WITH_SB3="false"; shift 1 ;;
        --sb3-steps) SB3_TOTAL_STEPS="$2"; shift 2 ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "Unknown option: $1"; print_help; exit 1 ;;
    esac
done

: "${SB3_TOTAL_STEPS:=$((NUM_EPISODES * MAX_STEPS))}"

cd "$WORKDIR"

check_disk() {
    local free_gb
    free_gb="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
    if (( free_gb < MIN_FREE_GB )); then
        echo "Refusing to start: only ${free_gb}G free on / (need ${MIN_FREE_GB}G)." >&2
        exit 1
    fi
}

# Launch one run in the background, logging to its own file.
# $1 agent (sac|sb3)   $2 seed   $3 domain id
start_run() {
    local agent="$1" seed="$2" domain="$3"
    local out="${RUNS_DIR}/${agent}/seed${seed}"
    mkdir -p "$out"

    local args=(
        --num-episodes "$NUM_EPISODES"
        --max-steps "$MAX_STEPS"
        --output-dir "$out"
        --seed "$seed"
        --domain-id "$domain"
        --agent "$agent"
    )
    [[ "$agent" == "sb3" ]] && args+=(--baseline-steps "$SB3_TOTAL_STEPS")

    echo "  → ${agent} seed ${seed}  domain ${domain}  log ${out}/console.log"
    ./execute_training_docker.sh "${args[@]}" > "${out}/console.log" 2>&1 &
    RUNNING_PIDS+=($!)
    RUNNING_NAMES+=("${agent}/seed${seed}")
}

# Wait for every launched run, reporting any that failed.
wait_for_batch() {
    local failed=0 i
    for i in "${!RUNNING_PIDS[@]}"; do
        if wait "${RUNNING_PIDS[$i]}"; then
            echo "  ✓ ${RUNNING_NAMES[$i]}"
        else
            echo "  ✗ ${RUNNING_NAMES[$i]} (see ${RUNS_DIR}/${RUNNING_NAMES[$i]}/console.log)"
            failed=$((failed + 1))
        fi
    done
    RUNNING_PIDS=()
    RUNNING_NAMES=()
    return $failed
}

# Build the full job list: every seed for sac, then every seed for sb3.
JOBS=()
for seed in "${SEEDS[@]}"; do JOBS+=("sac:${seed}"); done
if [[ "$WITH_SB3" == "true" ]]; then
    for seed in "${SEEDS[@]}"; do JOBS+=("sb3:${seed}"); done
fi

echo "=== Evidence sweep ==="
echo "  seeds        : ${SEEDS[*]}"
echo "  episodes     : ${NUM_EPISODES} x ${MAX_STEPS} steps"
echo "  jobs         : ${#JOBS[@]} (${PARALLEL} at a time)"
echo "  output       : ${RUNS_DIR}"
echo "  sb3 baseline : ${WITH_SB3} (${SB3_TOTAL_STEPS} steps)"
echo

RUNNING_PIDS=()
RUNNING_NAMES=()
TOTAL_FAILED=0
batch=0

for (( i = 0; i < ${#JOBS[@]}; i += PARALLEL )); do
    batch=$((batch + 1))
    check_disk
    echo "--- batch ${batch}"
    for (( j = i; j < i + PARALLEL && j < ${#JOBS[@]}; j++ )); do
        job="${JOBS[$j]}"
        start_run "${job%%:*}" "${job##*:}" "$(( BASE_DOMAIN_ID + j - i ))"
        # Stagger startup: three Gazebo instances coming up at the same instant
        # contend badly and the topic waits can time out.
        sleep 10
    done
    wait_for_batch || TOTAL_FAILED=$((TOTAL_FAILED + $?))
    echo
done

echo "Sweep complete (${TOTAL_FAILED} failed run(s))."
echo "Render the figures with:"
echo "  python3 scripts/plot_results.py --runs-dir ${RUNS_DIR} --out-dir docs/figures"
exit $(( TOTAL_FAILED > 0 ? 1 : 0 ))
