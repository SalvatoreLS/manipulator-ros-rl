"""Env for the distance-based reward function."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..data_monitor import DataMonitor

MIN_DISTANCE_THRESHOLD : float = 0.05

class ManipulatorEnv(gym.Env):
    def __init__(self):
        super(ManipulatorEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32) # TODO: Define the observation space properly
        self.state = None
        self.target = None # Defined by the user input of the target position (ROS2 topic /manipulator_target)
        self.data_monitor = DataMonitor()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(7, dtype=np.float32)
        observation = np.concatenate([self.state, self.target])  # Combine state and target for observation
        return observation, {}

    def step(self, action):
        """
        The env receives the action and sees the new position of the manipulator hand in the space.
        The reward is calculated based on the distance between the manipulator hand and the target position.

        The position of the manipulator is read from the topic /franka_robot_state_broadcaster/robot_state (or prefixed topic if using ROS2 namespaces).
        The message is of type franka_msgs/RobotState, and the position of the manipulator hand can be obtained from the field robot_state.O_T_EE (the position of the end-effector in the world frame).

        The target position is read from the topic /manipulator_target (or prefixed topic if using ROS2 namespaces), and the message type is geometry_msgs/Point.
        """
        self.state = self.data_monitor.get_position_data().manipulator_position
        self.target = self.data_monitor.get_position_data().target_position

        # Calculate the distance between the manipulator hand and the target position
        distance = self.__compute_distance()
        reward = self.__compute_reward(distance)

        terminated = bool(distance < MIN_DISTANCE_THRESHOLD)
        truncated = False

        observation = np.concatenate([self.state, self.target])
        return observation, reward, terminated, truncated, {}

    def _get_ee_position(self, joint_angles):
        """
        Helper function to compute the end-effector position from the joint angles.
        Since I have a ROS2 topic that returns the position directly, I skip this step for now.
        """
        return self.state

    def render(self):
        """No rendering necessary because it interacts directly with the Gazebo environment."""
        pass  # Placeholder for rendering the environment

    def __compute_distance(self):
        """
        Compute the distance between the manipulator hand and the target position.
        Using a function because I might want to change the distance calculation method later.
        """
        return np.linalg.norm(np.array(self.state) - np.array([self.target.x, self.target.y, self.target.z]))

    def __compute_reward(self, distance : float):
        """
        Compute the reward based on the distance between the manipulator hand and the target position.
        The reward is negative of the distance, so the agent is incentivized to minimize the distance.
        """
        reward = -distance
        return reward