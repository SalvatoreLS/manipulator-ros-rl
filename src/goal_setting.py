"""
ROS2 node that publishes a coordinate to the topic /manipulator_goal.
This goal is read and the robot will move to that target once properly trained with reinforcement learning.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

class GoalSetting(Node):
    def __init__(self):
        super().__init__('goal_setting')
        self.subscriber_ = self.create_subscription(Point, '/manipulator_goal', 10)
        self.get_logger().info('GoalSetting node has been started. Ready to read goals from /manipulator_goal topic.')

    def read_new_goal(self, goal: Point):
        """Method to read a new goal position."""
        self.get_logger().info(f'Read new goal: {goal}')

def main(args=None):
    rclpy.init(args=args)
    goal_setting = GoalSetting()
    rclpy.spin(goal_setting)
    goal_setting.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()