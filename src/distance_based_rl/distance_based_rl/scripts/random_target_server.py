import random
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_srvs.srv import SetBool

class RandomTargetServer(Node):
    def __init__(self):
        super().__init__('random_target_server')
        self.publisher_ = self.create_publisher(Point, '/manipulator_target', 10)
        self.service = self.create_service(SetBool, '/set_random_target', self.set_random_target_callback)

    def set_random_target_callback(self, request, response):
        try:
            msg = Point()
            # TODO: Define ranges of values correctly
            msg.x = random.uniform(-1.0, 1.0)
            msg.y = random.uniform(-1.0, 1.0)
            msg.z = random.uniform(0.0, 1.0)
            self.publisher_.publish(msg)
            self.get_logger().info(f'Published new random target: ({msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f})')
            response.success = True
        except Exception as e:
            self.get_logger().error(f'Error generating random target: {str(e)}')
            response.success = False
        return response

def main(args=None):
    rclpy.init(args=args)
    random_target_server = RandomTargetServer()
    rclpy.spin(random_target_server)
    random_target_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()