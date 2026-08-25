# Distance-based RL for a Franka FR3 in ROS 2 + Gazebo

A reinforcement-learning reaching task for a 7-DOF Franka FR3, built as a **Gymnasium
environment over a live ROS 2 control stack** and trained with a **Soft Actor-Critic agent
written from scratch in PyTorch**.

The agent observes the arm's joint state and the end-effector and target positions, and
commands incremental joint moves through `ros2_control`. The reward is the negative
Euclidean distance to a randomly sampled target inside the reachable workspace.

The interesting part is not the algorithm — SAC is standard — but the interface: making a
message-passing, asynchronous, real-time robot stack behave like a synchronous
`env.step()` without lying to the learner about what happened. Those decisions are written
up in **[docs/DESIGN.md](docs/DESIGN.md)**.

<!-- Populated by scripts/plot_results.py once the evidence sweep has run:
     ![demo](docs/figures/demo.gif)
     ![learning curve](docs/figures/success_rate.png) -->

---

## What is in this repository

| Path | Contents |
|---|---|
| `src/distance_based_rl/` | **The contribution**: the Gymnasium environment, the ROS 2 interface node, the SAC agent, training and deployment entry points, and the test suite. |
| `src/keyboard_movement/` | Small teleoperation node used to sanity-check the control stack by hand. |
| `scripts/` | Evidence sweep, Stable-Baselines3 baseline, figure generation. |
| `docs/` | Design notes and results. |
| `src/deps.repos` | Third-party ROS 2 packages (`franka_ros2`, `libfranka`, `franka_description`, `olvx_descriptions_module`). Fetched with `vcs`, **not** written here. |

### Stack

| Component | Version |
|---|---|
| OS | Ubuntu 22.04 (Jammy) |
| ROS 2 | Humble Hawksbill |
| Simulator | Gazebo (Ignition Fortress) via `ros_gz` |
| Control | `ros2_control`, `forward_position_controller` |
| RL | PyTorch — SAC implemented from scratch; Stable-Baselines3 used only as a baseline |
| Interface | Gymnasium |

---

## Quickstart

Everything runs inside the container — ROS 2, `colcon` and the Python stack are not
expected on the host.

```bash
# 1. Fetch the third-party ROS 2 sources
vcs import src < src/deps.repos
git -C src/libfranka submodule update --init --recursive

# 2. Build and enter the container (X11 for the Gazebo GUI)
xhost +local:docker
docker compose build
docker compose up -d
docker exec -it franka_ros2_rl bash

# 3. Build the workspace (inside the container)
colcon build --symlink-install
source install/setup.bash

# 4. Train — starts the bridge, Gazebo and the training run
./execute_training_docker.sh --num-episodes 300 --max-steps 300 --seed 0

# 5. Watch it learn
tensorboard --logdir output/logs
```

### Deploying a trained policy

```bash
ros2 run distance_based_rl eval_agent --load-model output/best_model.pt
# in another shell, publish a target the arm should reach and hold:
ros2 topic pub --once /manipulator_target geometry_msgs/msg/Point "{x: 0.4, y: 0.0, z: 0.5}"
```

The deployment path does not home the arm and does not randomise the target: it moves from
wherever the arm currently is toward the published target, then holds, and tracks a new
target whenever one is published.

### Reproducing the results

```bash
# 3 seeds x {from-scratch SAC, SB3 baseline}, 3 runs at a time on separate
# ROS_DOMAIN_IDs.  ~4-5 h total at the measured ~2.9 environment steps/s.
./scripts/run_experiments.sh
python3 scripts/plot_results.py --runs-dir output/runs --out-dir docs/figures
```

See **[docs/RESULTS.md](docs/RESULTS.md)** for what the runs show, and for the limitations.

---

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │  train.py — SAC training loop        │
                    │  gradient updates on a worker thread,│
                    │  overlapping env.step()'s settle wait│
                    └───────────────┬──────────────────────┘
                                    │ Gymnasium API
                    ┌───────────────▼──────────────────────┐
                    │  ManipulatorEnv (arm_env.py)         │
                    │  reset / step / reward / termination │
                    └───────────────┬──────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────┐
                    │  DataHandler (data_handler.py)       │
                    │  rclpy Node, MultiThreadedExecutor   │
                    └───┬───────────────────────────┬──────┘
                        │ subscribes                │ publishes
   /franka_robot_state_broadcaster/robot_state      /forward_position_controller/commands
   /joint_states  (+ TF: world → fr3_link8)         /manipulator_target
                        │                           │
                    ┌───▼───────────────────────────▼──────┐
                    │  ros2_control  ·  Gazebo  /  FR3     │
                    └──────────────────────────────────────┘
```

**Observation** (28-D, normalised): EE position (3) · target position (3) · joint
positions (7) · joint velocities (7) · joint efforts (7) · EE-target distance (1).

**Action** (7-D, `[-1, 1]`): incremental joint targets, `q + a · 0.1 rad`, clipped to the
FR3 joint limits.

**Reward**: `-distance`, plus an additive `+50` on reaching the target (< 10 cm), which
terminates the episode. Success is additionally reported at 5 cm and 2 cm.

---

## Documentation

- **[docs/DESIGN.md](docs/DESIGN.md)** — why the environment is built the way it is
- **[docs/RESULTS.md](docs/RESULTS.md)** — learning curves, SB3 cross-check, limitations
- **[src/distance_based_rl/README.md](src/distance_based_rl/README.md)** — package reference and CLI
- **[DOCKER.md](DOCKER.md)** — container setup
- **[VISUALIZATION.md](VISUALIZATION.md)** — launching and driving the robot by hand
- **[TOPICS.md](TOPICS.md)** — ROS 2 topics, services and actions in the workspace

## Tests

```bash
colcon test --packages-select distance_based_rl && colcon test-result --verbose
# or directly:
pytest src/distance_based_rl/test -v
```

The suite runs without a simulator: the ROS 2 node is exercised with its publishers mocked
and its `__init__` bypassed, so callbacks, the action mapping, joint-limit clipping, target
sampling, reward and termination are all covered offline.

## License

Apache 2.0 — see [LICENSE](LICENSE). Third-party packages listed in `src/deps.repos` carry
their own licenses.
