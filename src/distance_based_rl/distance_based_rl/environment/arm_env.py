"""Env for the distance-based reward function."""

import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
from typing import Optional

from .data_handler import DataHandler
from .env_config import EnvConfig


class State:
    """
    Data structure to hold the current state of the environment, including the manipulator's position and the target position.

    This is used for constructing the observation that is returned to the RL agent.
    """

    def __init__(self, manipulator_position, target_position, joint_positions, joint_velocities, joint_efforts):
        self.manipulator_position = manipulator_position  # Tuple of (x, y, z)
        self.target_position = target_position  # Tuple of (x, y, z)
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_efforts = joint_efforts

    @staticmethod
    def _normalize_vector(values, size):
        if values is None:
            return tuple([0.0] * size)
        if isinstance(values, np.ndarray):
            values = values.tolist()
        values = tuple(float(v) for v in values)
        if len(values) < size:
            values = values + tuple([0.0] * (size - len(values)))
        return values[:size]

    @classmethod
    def from_sources(
        cls,
        manipulator_position=None,
        target_position=None,
        joint_positions=None,
        joint_velocities=None,
        joint_efforts=None,
    ):
        return cls(
            manipulator_position=cls._normalize_vector(manipulator_position, 3),
            target_position=cls._normalize_vector(target_position, 3),
            joint_positions=cls._normalize_vector(joint_positions, 7),
            joint_velocities=cls._normalize_vector(joint_velocities, 7),
            joint_efforts=cls._normalize_vector(joint_efforts, 7),
        )

    # Normalization constants — Franka workspace / joint limits
    _EE_SCALE   = 0.75   # metres: EE/target positions roughly in [0, 0.75]
    _JPOS_SCALE = np.pi  # rad
    _JVEL_SCALE = 2.5    # rad/s
    _JEFF_SCALE = 87.0   # N·m

    def as_vector(self) -> np.ndarray:
        ee_pos     = np.asarray(self.manipulator_position, dtype=np.float32)
        tgt_pos    = np.asarray(self.target_position,      dtype=np.float32)
        jpos       = np.asarray(self.joint_positions,      dtype=np.float32)
        jvel       = np.asarray(self.joint_velocities,     dtype=np.float32)
        jeff       = np.asarray(self.joint_efforts,        dtype=np.float32)
        distance   = np.linalg.norm(ee_pos - tgt_pos) / self._EE_SCALE
        return np.concatenate([
            ee_pos  / self._EE_SCALE,
            tgt_pos / self._EE_SCALE,
            jpos    / self._JPOS_SCALE,
            jvel    / self._JVEL_SCALE,
            jeff    / self._JEFF_SCALE,
            [distance],
        ]).astype(np.float32)

    @staticmethod
    def vector_dim() -> int:
        return 3 + 3 + 7 + 7 + 7 + 1  # +1 for normalised distance feature


class StateActionReward:
    """
    Data structure to hold a single transition of state, action, reward, next_state, and done flag.

    This is used for storing experiences in the replay buffer for training the RL agent.
    """

    def __init__(self, state: State, action: np.ndarray, reward: float,
                 next_state: State, done: bool):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def as_replay_tuple(self):
        return (
            self.state.as_vector(),
            np.asarray(self.action, dtype=np.float32),
            float(self.reward),
            self.next_state.as_vector(),
            bool(self.done),
        )


class ManipulatorEnv(gym.Env):
    """
    Class defining the ROS2 environment for distance-based reinforcement learning with a manipulator.

    It interfaces with ROS2 and provides the standard Gym API (reset, step) for training RL agents.

    The environment subscribes to the manipulator's current position and the target position,
    and publishes action commands to the manipulator.
    The reward is based on the negative distance between the manipulator and the target,
    encouraging the agent to minimize this distance.
    """

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        external_target: bool = False,
        home_on_reset: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize the ROS2 environment, including setting up the ROS2 node, subscribers, and publishers.

        The action space is defined as a continuous space with 7 dimensions (e.g., for 7 joints),
        and the observation space includes the current position of the manipulator and the target position.

        ``config`` holds the tunable environment parameters (thresholds, timeouts,
        topics, target sampling volume) and defaults to ``EnvConfig()``, which reads the
        same environment variables the shell scripts set.

        ``external_target``: if True, reset() does NOT publish a random target and instead
        waits for one published externally to /manipulator_target — used at inference so
        the arm tracks a user-provided target.

        ``home_on_reset``: if True (training default), reset() commands the arm back to
        its home configuration before each episode. Set False at inference so the arm
        moves toward the target from its current pose instead of homing first.

        ``seed`` seeds target sampling and the action space, making a run reproducible.
        """
        super(ManipulatorEnv, self).__init__()

        self.config = config if config is not None else EnvConfig()
        self._ros_ready = False
        self.executor = None
        self.spin_thread = None
        self.target_position = None
        self._external_target = bool(external_target)
        self._home_on_reset = bool(home_on_reset)

        # Track whether *this* env initialised rclpy, so close() only tears down what it
        # created and never pulls the context out from under another env in the process.
        self._owns_rclpy = False
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True

        # Interface node for ROS2 communication (subscribers and publishers)
        self.node = DataHandler(config=self.config, seed=seed)

        if rclpy.ok():
            try:
                self.executor = MultiThreadedExecutor()
                self.executor.add_node(self.node)

                # Run ROS2 spinning in a background thread
                self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
                self.spin_thread.start()
                self._ros_ready = True
            except Exception:
                self.executor = None
                self.spin_thread = None

        self._max_episode_steps = self.config.max_episode_steps
        self._step_count = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        if seed is not None:
            self.action_space.seed(seed)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(State.vector_dim(),),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        """
        Start a new episode and return the initial observation.

        Sets a new random target in the ROS2 environment and waits for valid state and
        target data before returning.
        """
        super().reset(seed=seed)

        # Propagate the seed to the node's private generator: without this, target
        # sampling would keep drawing from unseeded global state and no run would be
        # reproducible from its seed.
        if seed is not None:
            self.node.seed(seed)
            self.action_space.seed(seed)

        # Training: return the arm to a known home pose so every episode starts the same.
        # Inference (home_on_reset=False): leave the arm where it is and move toward target.
        if self._home_on_reset:
            if self.node.go_home():
                self._wait_until_settled()
            else:
                self.node.get_logger().warning("Failed to command home pose on reset")

        # Set a new random target in the ROS2 environment — skipped when an external target
        # is provided (inference), so a user-published /manipulator_target is preserved.
        if self._external_target:
            print("⏳ Waiting for externally published target on /manipulator_target")
        elif self.node.set_random_target():
            print("✓ New random target set successfully")
        else:
            print("✗ Failed to set new random target. Using the last target if available.")

        self.target_position = self.node.get_target_position()

        # Wait until we have valid data from subscribers.
        # Use elapsed wall time instead of loop iterations to avoid instant timeout.
        timeout_sec = self.config.state_wait_timeout_sec
        poll_interval_sec = self.config.state_wait_poll_sec
        deadline = time.monotonic() + max(0.0, timeout_sec)
        next_info_log = time.monotonic() + 0.5
        while rclpy.ok() and (self.node.get_manipulator_position() is None or self.node.get_target_position() is None):
            now = time.monotonic()
            if now >= deadline:
                self.node.get_logger().warning('Timeout waiting for robot state or target')
                break
            if now >= next_info_log:
                self.node.get_logger().info('Waiting for robot state and target...')
                next_info_log = now + 0.5
            time.sleep(max(0.0, poll_interval_sec))

        self._step_count = 0
        state = self._build_state()
        return state.as_vector(), {"state": state}

    def step(self, action):
        """
        Publish the action and compute the reward based on the distance to the target.

        The episode is terminated if the distance is below a threshold.
        """
        if not self.node.publish_command(action):

            self.node.get_logger().warning("ROS2 not ready, failed to publish action command")
        else:
            self.node.get_logger().debug(f"Published action command: {action}")

        # Wait for the controller to actually move the joints to the commanded increment
        # before measuring the resulting state, so the reward reflects the action's outcome
        # rather than a mid-motion transient.
        self._wait_until_settled()

        self._step_count += 1
        curr_pos = self.node.get_manipulator_position()
        target_pos = self.node.get_target_position()

        # Guard: if either position is unavailable (ROS2 data not yet received), skip
        # distance computation entirely.  Without this check, both positions would default
        # to (0, 0, 0), distance collapses to 0, and the episode terminates spuriously with
        # the success bonus.
        if curr_pos is None or target_pos is None:
            self.node.get_logger().warning(
                "Position data unavailable during step (curr=%s, target=%s); skipping reward computation",
                curr_pos, target_pos,
            )
            next_state = self._build_state()
            truncated = self._step_count >= self._max_episode_steps
            return next_state.as_vector(), 0.0, False, truncated, {"state": next_state, "success": False}

        distance = np.linalg.norm(np.array(curr_pos, dtype=np.float64) - np.array(target_pos, dtype=np.float64))

        terminated = bool(distance < self.config.min_distance_threshold)
        truncated  = (not terminated) and self._step_count >= self._max_episode_steps

        # Additive bonus: preserves distance-shaped gradient at the terminal step.
        reward = -float(distance) + (self.config.success_bonus if terminated else 0.0)

        next_state = self._build_state()
        return next_state.as_vector(), reward, terminated, truncated, {
            "state":    next_state,
            "success":  terminated,
            "distance": float(distance),
            # Success at several tolerances, so the reported rate is not tied to the
            # single (generous) threshold that ends the episode.
            "success_at": {
                f"{t:.2f}": bool(distance < t) for t in self.config.report_thresholds
            },
            "steps": self._step_count,
        }

    def _wait_until_settled(self) -> None:
        """
        Block until the arm's joints have (approximately) stopped moving.

        Returns early on timeout. This replaces a fixed sleep so the measured state reflects the commanded
        move's outcome. Tunable via environment variables.
        """
        min_dwell  = self.config.settle_min_dwell_sec

        timeout    = self.config.settle_timeout_sec
        poll       = self.config.settle_poll_sec
        vel_thresh = self.config.settle_vel_thresh  # rad/s

        # Always give the controller a brief head start before sampling velocities.
        time.sleep(max(0.0, min_dwell))

        deadline = time.monotonic() + max(0.0, timeout)
        while rclpy.ok() and time.monotonic() < deadline:
            vels = self.node.get_joint_velocities()
            if vels is not None and float(np.max(np.abs(np.asarray(vels, dtype=np.float64)))) < vel_thresh:
                return  # settled
            time.sleep(max(0.0, poll))

    def get_state(self) -> State:
        """Return the current environment state as a :class:`State` object."""
        return self._build_state()

    def _build_state(self) -> State:
        return State.from_sources(
            manipulator_position=self.node.get_manipulator_position(),
            target_position=self.node.get_target_position(),
            joint_positions=self.node.get_joint_positions(),
            joint_velocities=self.node.get_joint_velocities(),
            joint_efforts=self.node.get_joint_efforts(),
        )

    def close(self):
        """
        Clean up function to shut down ROS2 and join the spinning thread when the env is closed.

        Only shuts rclpy down if this env was the one that initialised it — otherwise a
        second env (or any other node) in the same process would lose its context.
        """
        if hasattr(self, 'node') and self.node is not None:

            self.node.destroy_node()
            self.node = None
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
            self._owns_rclpy = False
        if self.spin_thread is not None:
            self.spin_thread.join(timeout=5.0)
            self.spin_thread = None
