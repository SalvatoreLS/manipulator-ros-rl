"""
Environment configuration.

Every tunable knob of the ROS2/Gazebo environment lives here, so a run is fully
described by an ``EnvConfig`` plus a ``TrainingConfig`` (``agent/config.py``) — both
serialisable to JSON and both saved next to the checkpoints.

Physical constants of the robot (joint limits, home configuration) are *not* here: they
belong to the hardware, not to the experiment, and stay as class constants on
``DataHandler``.

Environment variables are kept as the fallback for every field so the existing shell
scripts (``execute_training_docker.sh``) keep working unchanged.
"""

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Tuple


def _env_float(name: str, default: float) -> float:
    """Read *name* from the environment as a float, falling back to *default*."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Read *name* from the environment as an int, falling back to *default*."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class EnvConfig:
    """Tunable parameters of :class:`~distance_based_rl.environment.arm_env.ManipulatorEnv`."""

    # ── Episode ────────────────────────────────────────────────────────────────
    max_episode_steps: int = field(default_factory=lambda: _env_int('MAX_STEPS_PER_EPISODE', 500))

    # Distance below which the episode terminates successfully (metres).
    min_distance_threshold: float = field(
        default_factory=lambda: _env_float('FRANKA_MIN_DISTANCE_THRESHOLD', 0.1)
    )

    # Additional thresholds reported alongside the terminating one, so success rate is
    # quoted honestly at several tolerances rather than only the most generous.
    report_thresholds: Tuple[float, ...] = (0.10, 0.05, 0.02)

    # Additive bonus on the terminating step. Additive (not replacing) so the
    # distance-shaped gradient survives at the terminal step.
    success_bonus: float = 50.0

    # ── Action ─────────────────────────────────────────────────────────────────
    # Maximum joint movement commanded per environment step (rad). Small increments keep
    # each move reachable within the controller-settle window, so the measured state
    # actually reflects the action.
    max_joint_delta: float = field(
        default_factory=lambda: _env_float('FRANKA_MAX_JOINT_DELTA', 0.1)
    )

    # ── Waiting for ROS2 state ─────────────────────────────────────────────────
    state_wait_timeout_sec: float = field(
        default_factory=lambda: _env_float('FRANKA_STATE_WAIT_TIMEOUT_SEC', 5.0)
    )
    state_wait_poll_sec: float = field(
        default_factory=lambda: _env_float('FRANKA_STATE_WAIT_POLL_SEC', 0.02)
    )

    # ── Settle detection (replaces a fixed sleep after each command) ───────────
    settle_min_dwell_sec: float = field(
        default_factory=lambda: _env_float('FRANKA_SETTLE_MIN_DWELL_SEC', 0.05)
    )
    settle_timeout_sec: float = field(
        default_factory=lambda: _env_float('FRANKA_SETTLE_TIMEOUT_SEC', 0.5)
    )
    settle_poll_sec: float = field(
        default_factory=lambda: _env_float('FRANKA_SETTLE_POLL_SEC', 0.02)
    )
    settle_vel_thresh: float = field(
        default_factory=lambda: _env_float('FRANKA_SETTLE_VEL_THRESH', 0.05)
    )  # rad/s

    # ── ROS2 topics and frames ─────────────────────────────────────────────────
    robot_state_topic: str = field(
        default_factory=lambda: os.getenv(
            'FRANKA_ROBOT_STATE_TOPIC', '/franka_robot_state_broadcaster/robot_state'
        )
    )
    joint_states_topic: str = field(
        default_factory=lambda: os.getenv('FRANKA_JOINT_STATES_TOPIC', '/joint_states')
    )
    target_topic: str = '/manipulator_target'
    command_topic: str = field(
        default_factory=lambda: os.getenv(
            'FRANKA_COMMAND_TOPIC', '/forward_position_controller/commands'
        )
    )
    ee_frame: str = field(default_factory=lambda: os.getenv('FRANKA_EE_FRAME', 'fr3_link8'))
    base_frame: str = field(default_factory=lambda: os.getenv('FRANKA_BASE_FRAME', 'world'))

    # ── Target sampling volume (spherical, robot base frame) ───────────────────
    # See DataHandler.set_random_target for the meaning of each pair.
    radius_bounds: Tuple[float, float] = (0.2, 0.75)
    azimuth_bounds: Tuple[float, float] = (-5 / 12, 5 / 12)
    elevation_bounds: Tuple[float, float] = (0.25, 0.78)

    def save(self, path: str) -> None:
        """Write the configuration to *path* as JSON."""
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'EnvConfig':
        """Read a configuration back from the JSON file at *path*."""
        with open(path, 'r') as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        tuple_fields = {
            'report_thresholds', 'radius_bounds', 'azimuth_bounds', 'elevation_bounds',
        }
        kwargs = {
            k: (tuple(v) if k in tuple_fields else v)
            for k, v in data.items() if k in known
        }
        return cls(**kwargs)
