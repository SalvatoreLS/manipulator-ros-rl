# `distance_based_rl`

ROS 2 package holding the Gymnasium environment, the from-scratch SAC agent, and the
training / deployment entry points for the distance-based reaching task.

Design rationale lives in [`docs/DESIGN.md`](../../docs/DESIGN.md); results and limitations
in [`docs/RESULTS.md`](../../docs/RESULTS.md).

---

## Layout

```
distance_based_rl/
├── distance_based_rl/
│   ├── agent/
│   │   ├── sac_agent.py       SAC: tanh-Gaussian policy, twin critics, entropy tuning,
│   │   │                      pre-allocated thread-safe replay buffer
│   │   ├── train.py           Training loop (argparse, TensorBoard, async gradients)
│   │   ├── eval.py            Deployment: track a published target and hold
│   │   └── config.py          TrainingConfig
│   └── environment/
│       ├── arm_env.py         ManipulatorEnv (Gymnasium) + State / StateActionReward
│       ├── data_handler.py    rclpy Node: subscriptions, publishers, target sampling
│       ├── env_config.py      EnvConfig — every environment knob, serialisable
│       └── __init__.py        registers "ManipulatorReach-v0"
├── test/                      offline test suite (no simulator required)
├── package.xml
└── setup.py                   entry points: train_agent, eval_agent
```

## Interfaces

**Subscribes**

| Topic | Type | Role |
|---|---|---|
| `/franka_robot_state_broadcaster/robot_state` | `franka_msgs/FrankaRobotState` | EE pose + joint state (hardware) |
| `/joint_states` | `sensor_msgs/JointState` | joint state; EE via TF fallback (Gazebo) |
| `/manipulator_target` | `geometry_msgs/Point` | current goal |

**Publishes**

| Topic | Type | Role |
|---|---|---|
| `/forward_position_controller/commands` | `std_msgs/Float64MultiArray` | joint position targets |
| `/manipulator_target` | `geometry_msgs/Point` | new random goal on reset (training only) |

## MDP

**Observation** — 28-D `float32`, each block normalised:

| Block | Dim | Scale |
|---|---|---|
| EE position | 3 | 0.75 m |
| Target position | 3 | 0.75 m |
| Joint positions | 7 | π rad |
| Joint velocities | 7 | 2.5 rad/s |
| Joint efforts | 7 | 87 N·m |
| EE–target distance | 1 | 0.75 m |

**Action** — 7-D in `[-1, 1]`, an incremental joint move `q + a · 0.1 rad`, clipped to the
FR3 joint limits.

**Reward** — potential-based: `10·(d_prev - d) - 0.1·d - 0.01`, plus `+5` on reaching the
target. The dominant term is the distance *closed* over the step, not the absolute
distance. Episode terminates below 10 cm and truncates at the step budget; success is also
reported at 5 cm and 2 cm in `info["success_at"]`. Weights are overridable through
`FRANKA_PROGRESS_WEIGHT`, `FRANKA_DISTANCE_WEIGHT`, `FRANKA_STEP_COST` and
`FRANKA_SUCCESS_BONUS`.

---

## Usage

```bash
colcon build --packages-select distance_based_rl
source install/setup.bash
```

### Train

```bash
ros2 run distance_based_rl train_agent \
  --num-episodes 150 --max-steps 150 --seed 0 --output-dir output/runs/sac/seed0
```

Or let `execute_training_docker.sh` bring up the bridge, Gazebo and the training run
together:

```bash
./execute_training_docker.sh --num-episodes 150 --max-steps 150 --seed 0
```

### Deploy

```bash
ros2 run distance_based_rl eval_agent --load-model output/best_model.pt
ros2 topic pub --once /manipulator_target geometry_msgs/msg/Point "{x: 0.4, y: 0.0, z: 0.5}"
```

### Resume / reuse a configuration

```bash
# every run writes config.json and env_config.json next to its checkpoints
ros2 run distance_based_rl train_agent --config output/runs/sac/seed0/config.json
ros2 run distance_based_rl train_agent --load-model output/checkpoint_episode_50.pt
```

---

## CLI reference — `train_agent`

| Flag | Default | Description |
|---|---|---|
| `--num-episodes` | 1000 | Training episodes |
| `--max-steps` | 500 | Max steps per episode |
| `--batch-size` | 256 | Optimisation batch size |
| `--learning-rate` | 3e-4 | Adam learning rate |
| `--buffer-size` | 10000 | Replay buffer capacity |
| `--hidden-dim` | 256 | Hidden width of policy and critics |
| `--gradient-steps` | 6 | Gradient updates per environment step |
| `--warmup-steps` | 1000 | Random-action steps before the policy takes over |
| `--seed` | none | Seeds torch, numpy, the action space and target sampling |
| `--output-dir` | `output/` | Where checkpoints, configs and logs go |
| `--checkpoint-interval` | 50 | Checkpoint every N episodes |
| `--no-tensorboard` | off | Disable TensorBoard logging |
| `--load-model` | none | Resume from a checkpoint |
| `--config` | none | Load a saved `config.json` |

## Environment variables

`EnvConfig` reads these, so the shell scripts can tune the environment without code
changes: `MAX_STEPS_PER_EPISODE`, `FRANKA_MIN_DISTANCE_THRESHOLD`,
`FRANKA_MAX_JOINT_DELTA`, `FRANKA_STATE_WAIT_TIMEOUT_SEC`, `FRANKA_STATE_WAIT_POLL_SEC`,
`FRANKA_SETTLE_MIN_DWELL_SEC`, `FRANKA_SETTLE_TIMEOUT_SEC`, `FRANKA_SETTLE_POLL_SEC`,
`FRANKA_SETTLE_VEL_THRESH`, `FRANKA_ROBOT_STATE_TOPIC`, `FRANKA_JOINT_STATES_TOPIC`,
`FRANKA_COMMAND_TOPIC`, `FRANKA_EE_FRAME`, `FRANKA_BASE_FRAME`.

Explicit constructor arguments always win over the environment.

---

## Tests

```bash
colcon test --packages-select distance_based_rl && colcon test-result --verbose
# or
pytest test -v
```

No simulator needed: `DataHandler` is constructed with its `__init__` bypassed and its
publishers mocked, and `ManipulatorEnv` is built with `rclpy` patched out. Covered:
observation assembly and normalisation, both EE sources, joint reordering by name, the
incremental action mapping and joint-limit clipping, target-sampling bounds and
reproducibility, reward, termination vs truncation, the missing-data guard, and
`EnvConfig` round-trips.

## Known limitations

Position-only reward (no orientation, collision or effort terms), a single non-vectorised
environment, no domain randomisation, and no sim-to-real transfer. See
[`docs/RESULTS.md`](../../docs/RESULTS.md).
