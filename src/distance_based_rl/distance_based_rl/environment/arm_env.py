"""Env for the distance-based reward function."""

import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time

from .data_handler import DataHandler

# Threshold for episode termination based on distance to target (in meters)
MIN_DISTANCE_THRESHOLD = 0.05

class State:
    """
    Data structure to hold the current state of the environment, including the manipulator's position and the target position.
    This is used for constructing the observation that is returned to the RL agent.
    """
    def __init__(self, manipulator_position, target_position, joint_positions, joint_velocities, joint_efforts):
        self.manipulator_position = manipulator_position # Tuple of (x, y, z)
        self.target_position = target_position # Tuple of (x, y, z)
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

    def as_vector(self) -> np.ndarray:
        return np.concatenate([
            np.asarray(self.manipulator_position, dtype=np.float32),
            np.asarray(self.target_position, dtype=np.float32),
            np.asarray(self.joint_positions, dtype=np.float32),
            np.asarray(self.joint_velocities, dtype=np.float32),
            np.asarray(self.joint_efforts, dtype=np.float32),
        ]).astype(np.float32)

    @staticmethod
    def vector_dim() -> int:
        return 3 + 3 + 7 + 7 + 7

class StateActionReward:
    """
    Data structure to hold a single transition of state, action, reward, next_state, and done flag.
    This is used for storing experiences in the replay buffer for training the RL agent.
    """
    def __init__(self, state : State, action : np.ndarray, reward : float, next_state : State, done : bool):
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
    def __init__(self):
        """
        Initialize the ROS2 environment, including setting up the ROS2 node, subscribers, and publishers.
        The action space is defined as a continuous space with 7 dimensions (e.g., for 7 joints),
        and the observation space includes the current position of the manipulator and the target position.
        """
        super(ManipulatorEnv, self).__init__()
        self._ros_ready = False
        self.executor = None
        self.spin_thread = None
        self.target_position = None
        
        if not rclpy.ok():
            rclpy.init()

        self.node = DataHandler() # Interface node for ROS2 communication (subscribers and publishers)

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

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(State.vector_dim(),),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        """
        Set a new random target position in the ROS2 env
        and wait for valid state and target data
        before returning the initial observation.
        """
        super().reset(seed=seed)
        
        # Set a new random target in the ROS2 environment

        if self.node.set_random_target(): print("✓ New random target set successfully")
        else:
            print("✗ Failed to set new random target. Using the last target if available.")
        
        self.target_position = self.node.get_target_position()
        
        # Wait until we have valid data from subscribers
        # Target should be updated by the service call, but we also need valid robot state
        timeout_counter = 0
        max_timeout = 100  # ~1 second at 100Hz
        while rclpy.ok() and (self.node.get_manipulator_position() is None or self.node.get_target_position() is None):
            timeout_counter += 1
            if timeout_counter > max_timeout:
                self.node.get_logger().warning('Timeout waiting for robot state or target')
                break
            if timeout_counter % 50 == 0:  # Log every 0.5s
                self.node.get_logger().info("Waiting for robot state and target...")

        state = self._build_state()
        return state.as_vector(), {"state": state}

    def step(self, action):
        """
        Publish the action and compute the reward based on the distance to the target.
        The episode is terminated if the distance is below a threshold.
        """
        if not self.node.publish_command(action): self.node.get_logger().warning("ROS2 not ready, failed to publish action command")
        else: self.node.get_logger().debug(f"Published action command: {action}")
        
        # Allow time for the controller to process and state to update if needed
        time.sleep(0.05)  # TODO: tune the delay or sync it better
        
        curr_pos = self.node.get_manipulator_position() or (0.0, 0.0, 0.0)
        target_pos = self.node.get_target_position() or (0.0, 0.0, 0.0)
        
        distance = np.linalg.norm(np.array(curr_pos) - np.array(target_pos))
        reward = -distance
        
        terminated = bool(distance < MIN_DISTANCE_THRESHOLD)
        truncated = False

        if terminated: reward = 10.0 # TODO: verify this works well or change the value
        
        next_state = self._build_state()
        return next_state.as_vector(), reward, terminated, truncated, {"state": next_state}

    def _build_state(self) -> State:
        return State.from_sources(
            manipulator_position=self.node.get_manipulator_position(),
            target_position=self.node.get_target_position(),
            joint_positions=self.node.get_joint_positions(),
            joint_velocities=self.node.get_joint_velocities(),
            joint_efforts=self.node.get_joint_efforts(),
        )

    def _get_obs(self):
        """
        Get the current observation, which includes the manipulator's position and the target position.
        If the positions are not yet available, it returns a default observation of zeros.
        """
        return self._build_state().as_vector()

    def close(self):
        """
        Clean up function to shut down ROS2 and join the spinning thread when the env is closed.
        """
        if hasattr(self, 'node') and self.node is not None:
            self.node.destroy_node()
        if self.executor is not None:
            self.executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        if self.spin_thread is not None:
            self.spin_thread.join()