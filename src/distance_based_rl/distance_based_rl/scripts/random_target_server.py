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
            self.get_logger().info(f'Published new random target: {msg}')
            response.success = True
            response.message = 'Random target published successfully'
        except Exception as e:
            response.success = False
            response.message = f'Error: {str(e)}'
        return response

def main(args=None):
    rclpy.init(args=args)
    random_target_server = RandomTargetServer()
    rclpy.spin(random_target_server)
    random_target_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()