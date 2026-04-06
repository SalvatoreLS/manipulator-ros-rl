"""Unit tests for environment components."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from geometry_msgs.msg import Point


class TestDataHandlerMocked:
    """Test DataHandler with mocked ROS2 components."""

    @patch('distance_based_rl.env.data_handler.rclpy')
    def test_data_handler_initialization(self, mock_rclpy):
        """Test DataHandler initialization."""
        from distance_based_rl.env.data_handler import DataHandler
        
        # Skip actual ROS2 initialization
        handler = DataHandler()
        
        assert handler._manipulator_position is None
        assert handler._target is None

    def test_point_to_tuple_conversion(self):
        """Test conversion of Point to tuple."""
        from distance_based_rl.env.data_handler import DataHandler
        
        # Create mock handler
        with patch.object(DataHandler, '__init__', lambda x: None):
            handler = DataHandler()
            handler._manipulator_position = None
            handler._target = None
        
        # Test with real Point
        point = Point(x=1.0, y=2.0, z=3.0)
        # Manually set to test property
        handler._manipulator_position = point
        
        result = handler.manipulator_position
        assert result == (1.0, 2.0, 3.0)

    def test_state_callback_extracts_ee_position(self):
        """Test that state callback correctly extracts end-effector position."""
        from distance_based_rl.env.data_handler import DataHandler
        from franka_msgs.msg import RobotState
        
        with patch.object(DataHandler, '__init__', lambda x: None):
            handler = DataHandler()
            handler._manipulator_position = None
            handler._lock = __import__('threading').Lock()
        
        # Create mock RobotState with O_T_EE transformation matrix
        msg = RobotState()
        # O_T_EE is a flattened 4x4 matrix (16 elements)
        # Translation is at indices 12, 13, 14
        msg.O_T_EE = [0]*12 + [1.5, 2.5, 3.5] + [0]*3
        
        handler.state_callback(msg)
        
        assert handler._manipulator_position is not None
        assert handler._manipulator_position.x == 1.5
        assert handler._manipulator_position.y == 2.5
        assert handler._manipulator_position.z == 3.5

    def test_target_callback_updates_target(self):
        """Test that target callback updates target position."""
        from distance_based_rl.env.data_handler import DataHandler
        
        with patch.object(DataHandler, '__init__', lambda x: None):
            handler = DataHandler()
            handler._target = None
            handler._lock = __import__('threading').Lock()
        
        point = Point(x=0.5, y=0.6, z=0.7)
        handler.target_callback(point)
        
        assert handler._target is not None
        assert handler._target.x == 0.5
        assert handler._target.y == 0.6
        assert handler._target.z == 0.7


class TestManipulatorEnvIntegration:
    """Integration tests for ManipulatorEnv (without full ROS2)."""

    @patch('distance_based_rl.env.arm_env.rclpy')
    @patch('distance_based_rl.env.arm_env.DataHandler')
    def test_manipulator_env_observation_space(self, mock_handler, mock_rclpy):
        """Test that observation space is correctly configured."""
        from distance_based_rl.env.arm_env import ManipulatorEnv
        
        mock_rclpy.ok.return_value = False  # Skip actual init
        
        env = ManipulatorEnv()
        
        assert env.observation_space.shape == (6,)
        assert env.action_space.shape == (7,)

    @patch('distance_based_rl.env.arm_env.rclpy')
    @patch('distance_based_rl.env.arm_env.DataHandler')
    def test_manipulator_env_action_space(self, mock_handler, mock_rclpy):
        """Test that action space is correctly configured."""
        from distance_based_rl.env.arm_env import ManipulatorEnv
        
        mock_rclpy.ok.return_value = False
        
        env = ManipulatorEnv()
        
        # Check bounds
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0

    @patch('distance_based_rl.env.arm_env.rclpy')
    @patch('distance_based_rl.env.arm_env.DataHandler')
    def test_get_obs_formatting(self, mock_handler, mock_rclpy):
        """Test that observations are correctly formatted."""
        from distance_based_rl.env.arm_env import ManipulatorEnv
        
        mock_rclpy.ok.return_value = False
        
        env = ManipulatorEnv()
        env.node = MagicMock()
        env.node.manipulator_position = (1.0, 2.0, 3.0)
        env.node.target = (0.5, 0.6, 0.7)
        
        obs = env._get_obs()
        
        assert obs.shape == (6,)
        assert obs.dtype == np.float32
        np.testing.assert_array_almost_equal(
            obs, 
            np.array([1.0, 2.0, 3.0, 0.5, 0.6, 0.7], dtype=np.float32)
        )

    @patch('distance_based_rl.env.arm_env.rclpy')
    @patch('distance_based_rl.env.arm_env.DataHandler')
    def test_get_obs_with_none_values(self, mock_handler, mock_rclpy):
        """Test that _get_obs handles None values gracefully."""
        from distance_based_rl.env.arm_env import ManipulatorEnv
        
        mock_rclpy.ok.return_value = False
        
        env = ManipulatorEnv()
        env.node = MagicMock()
        env.node.manipulator_position = None
        env.node.target = None
        
        obs = env._get_obs()
        
        assert obs.shape == (6,)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(obs, expected)

    @patch('distance_based_rl.env.arm_env.rclpy')
    @patch('distance_based_rl.env.arm_env.DataHandler')
    def test_step_reward_calculation(self, mock_handler, mock_rclpy):
        """Test that reward is negative distance."""
        from distance_based_rl.env.arm_env import ManipulatorEnv
        
        mock_rclpy.ok.return_value = False
        
        env = ManipulatorEnv()
        env.node = MagicMock()
        env.node.manipulator_position = (0.0, 0.0, 0.0)
        env.node.target = (1.0, 0.0, 0.0)
        
        action = np.array([0.0]*7)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Distance = sqrt((1-0)^2) = 1.0, reward = -1.0
        assert reward == -1.0
        assert not terminated  # Distance > threshold

    @patch('distance_based_rl.env.arm_env.rclpy')
    @patch('distance_based_rl.env.arm_env.DataHandler')
    def test_step_termination_condition(self, mock_handler, mock_rclpy):
        """Test that episode terminates when close enough to target."""
        from distance_based_rl.env.arm_env import ManipulatorEnv
        
        mock_rclpy.ok.return_value = False
        
        env = ManipulatorEnv()
        env.node = MagicMock()
        
        # Very close to target (< 0.05 threshold)
        env.node.manipulator_position = (0.0, 0.0, 0.0)
        env.node.target = (0.02, 0.0, 0.0)
        
        action = np.array([0.0]*7)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert terminated is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
