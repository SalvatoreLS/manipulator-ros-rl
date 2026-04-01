"""
ROS2 node that monitors the data about the manipulator's state and the target position.
It is used to provide this information to the reinforcement learning agent.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from franka_msgs.msg import RobotState

class PositionData:
    """Class to store the position data of the manipulator and the target."""
    def __init__(self):
        self.manipulator_position = None
        self.target_position = None

class DataMonitor(Node):
    def __init__(self):
        super().__init__('data_monitor')
        self.state = None
        self.target = None
        self.position_data = PositionData()

        # Subscribe to the manipulator state topic
        self.create_subscription(RobotState, '/franka_robot_state_broadcaster/robot_state', self.state_callback, 10)

        # Subscribe to the target position topic
        self.create_subscription(Point, '/manipulator_target', self.target_callback, 10)

    def state_callback(self, msg):
        """Callback function to handle incoming manipulator state messages."""
        self.state = msg.O_T_EE  # Assuming O_T_EE is the position of the end-effector

    def target_callback(self, msg):
        """Callback function to handle incoming target position messages."""
        self.target = msg  # Assuming msg is of type geometry_msgs/Point and contains the target position

    def get_position_data(self):
        """Method to return the current position data of the manipulator and the target."""
        self.position_data.manipulator_position = self.state
        self.position_data.target_position = self.target
        return self.position_data