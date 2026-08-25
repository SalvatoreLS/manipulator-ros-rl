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

## Status

**The training runs in this repository were not carried to convergence.** The budget the
sweep is configured for — 150 episodes x 150 steps, ~22.5k environment steps per run — is
modest for continuous control, and it is set by wall clock rather than by choice. The
bottleneck is not compute: a single non-vectorised environment sustains ~9.8 steps/s
(~102 ms/step measured), because every step publishes a command and waits for the arm to
settle through a real `ros2_control` round-trip. That is ~40 min per run in isolation, and
the sweep is six runs three at a time, each with its own Gazebo sharing the same CPU. The
hardware available for this work — a 12-core CPU with a 4 GB Quadro T2000 — does not make
a materially longer sweep practical, and the environment is not vectorised.

What that means for reading this repository:

- The code path is complete and exercised end to end: environment, ROS 2 interface, SAC
  agent, training loop, deployment entry point, and an offline test suite that runs without
  a simulator.
- The curves that the sweep produces show *learning*, not converged asymptotic
  performance. No claim is made about final success rate at any tolerance.
- Numbers are not quoted in [docs/RESULTS.md](docs/RESULTS.md) until the sweep is run;
  the tables are left empty rather than filled with estimates.

Treat this as an engineering artefact — the interface between an asynchronous robot stack
and a synchronous RL loop — rather than as a benchmark result.

---

## What is in this repository

| Path | Contents |
|---|---|
| `src/distance_based_rl/` | **The contribution**: the Gymnasium environment, the ROS 2 interface node, the SAC agent, training and deployment entry points, and the test suite. |
| `src/keyboard_movement/` | Small teleoperation node used to sanity-check the control stack by hand. |
| `scripts/` | Evidence sweep, Stable-Baselines3 baseline, figure generation. |
| `docs/` | Design notes and results. |
| `config/` | ROS-GZ bridge config, and `worlds/rl_empty.sdf` — the stock empty world with the real-time factor unlocked, used for training. |
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
./execute_training_docker.sh --num-episodes 150 --max-steps 150 --seed 0

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

### Reproducing the sweep

```bash
# 3 seeds x {from-scratch SAC, SB3 baseline}, 3 runs at a time on separate
# ROS_DOMAIN_IDs.  ~40 min per run in isolation at the measured ~9.8 environment
# steps/s; longer under --parallel 3, where three Gazebos share the CPU.
./scripts/run_experiments.sh
python3 scripts/plot_results.py --runs-dir output/runs --out-dir docs/figures
```

This is the sweep that has not been run to convergence here; see the
[Status](#status) note above. **[docs/RESULTS.md](docs/RESULTS.md)** records the setup,
what the curves are expected to show, and the limitations.

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

**Reward**: potential-based — `10·(d_prev - d) - 0.1·d - 0.01`, plus an additive `+5` on
reaching the target (< 10 cm), which terminates the episode. The dominant term is the
distance *closed* over the step rather than the absolute distance, which leaves the optimal
policy unchanged while giving a dense per-step signal. Success is additionally reported at
5 cm and 2 cm.

---

## Documentation

- **[docs/DESIGN.md](docs/DESIGN.md)** — why the environment is built the way it is
- **[docs/RESULTS.md](docs/RESULTS.md)** — learning curves, SB3 cross-check, limitations
- **[src/distance_based_rl/README.md](src/distance_based_rl/README.md)** — package reference and CLI
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

MIT — see [LICENSE](LICENSE). Third-party packages listed in `src/deps.repos` carry their
own licenses.
