"""
Monitor manipulator state and target position.

Provide the collected information to the reinforcement learning agent.
"""

import rclpy
import threading
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import Point
from typing import Optional, Tuple
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool
import numpy as np

# Handle imports for both direct execution and module import
try:
    from franka_msgs.msg import FrankaRobotState as _FrankaRobotState
except ImportError:
    _FrankaRobotState = None

try:
    from franka_msgs.msg import RobotState as RobotState
    _RobotStateType = RobotState
except ImportError:
    RobotState = None
    _RobotStateType = _FrankaRobotState

# Fallback definition for RobotState when unavailable
if RobotState is None:
    class RobotState:
        def __init__(self):
            self.O_T_EE = []

    if _FrankaRobotState is not None:
        import franka_msgs.msg as franka_msgs_msg
        franka_msgs_msg.RobotState = RobotState

if _RobotStateType is None:
    _RobotStateType = RobotState


class DataHandler(Node):
    """
    Interface node for ROS2 communication, handling subscriptions to the manipulator's state and target position,
    and providing methods to call services and publish commands.

    It maintains the current state of the manipulator and the target position, which can be accessed by the RL agent.
    """
    def __init__(self):
        """
        Initialize the ROS2 node, subscribers, and publishers.
        Set up thread-safe access to the manipulator state and target position.
        """
        self._lock = threading.Lock()
        self._manipulator_position: Optional[Point] = None
        self._target: Optional[Point] = None
        self._joint_positions: Optional[Tuple[float, ...]] = None
        self._joint_velocities: Optional[Tuple[float, ...]] = None
        self._joint_efforts: Optional[Tuple[float, ...]] = None
        self._ros_ready = False

        # Intiialization of ROS2 node
        try: super().__init__('data_monitor')
        except Exception: return

        # Callback group for thread safety
        cb_group = MutuallyExclusiveCallbackGroup()

        # Subscription for manipulator state
        self.create_subscription(
            _RobotStateType,
            '/franka_robot_state_broadcaster/robot_state',
            self.state_callback,
            10,
            callback_group=cb_group
        )

        # Subscription for target position (to compute reward in env)
        self.create_subscription(
            Point,
            '/manipulator_target',
            self.target_callback,
            10,
            callback_group=cb_group
        )

        # Publisher for target position (changed when resetting env)
        self.target_publisher_ = self.create_publisher(Point, '/manipulator_target', 10)

        # Publisher for action commands. It moves the joints of the manipulator (7D vector)
        self.commands_publisher_ = self.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)

        self._ros_ready = True

    @property
    def manipulator_position(self):
        """
        Thread-safe access to the current position of the manipulator.
        Returns a tuple (x, y, z) if the position is available, or None if not yet received.
        """
        lock = getattr(self, '_lock', None)
        if lock is None:
            position = getattr(self, '_manipulator_position', None)
        else:
            with lock:
                position = self._manipulator_position
        if position is None: return None
        return (position.x, position.y, position.z)

    @property
    def target(self):
        """
        Thread-safe access to the current target position.
        Same as manipulator_position, but for the target.
        Returns (x, y, z) or None.
        """
        lock = getattr(self, '_lock', None)
        if lock is None:
            target = getattr(self, '_target', None)
        else:
            with lock:
                target = self._target
        if target is None: return None
        return (target.x, target.y, target.z)

    def state_callback(self, msg):
        """
        Callback that processes messages for the manipulator state.
        It extracts the end-effector position field and updates the state inside the node.
        """
        if len(msg.O_T_EE) >= 15:
            p = Point()
            p.x = float(msg.O_T_EE[12])
            p.y = float(msg.O_T_EE[13])
            p.z = float(msg.O_T_EE[14])

            joint_positions = tuple(float(v) for v in getattr(msg, 'q', [])[:7]) or None
            joint_velocities = tuple(float(v) for v in getattr(msg, 'dq', [])[:7]) or None
            joint_efforts = tuple(float(v) for v in getattr(msg, 'tau_J', [])[:7]) or None

            lock = getattr(self, '_lock', None)
            if lock is None:
                self._manipulator_position = p
                self._joint_positions = joint_positions
                self._joint_velocities = joint_velocities
                self._joint_efforts = joint_efforts
            else:
                with lock:
                    self._manipulator_position = p
                    self._joint_positions = joint_positions
                    self._joint_velocities = joint_velocities
                    self._joint_efforts = joint_efforts

    def target_callback(self, msg):
        """
        Same as state_callback, but for the target position.
        It updates the target position inside the node.
        """
        lock = getattr(self, '_lock', None)
        if lock is None:
            self._target = msg
        else:
            with lock:
                self._target = msg

    def set_random_target(
        self,
        bounds : Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((0.1, 0.9), (0.1, 0.9), (0.1, 0.9))
    ) -> bool:
        """
        It sets a new random target position by publishing to /manipulator_target.
        It returns True if successful, False otherwise.

        The position is generated in polar coordinates (r, phi, theta) and converted to Cartesian (x, y, z).
        """
        if not getattr(self, '_ros_ready', False):
            return False
        
        # Generate a random target position within bounds
        r = np.random.uniform(bounds[0][0], bounds[0][1])
        phi = np.random.uniform(bounds[1][0], bounds[1][1]) * np.pi * 2
        theta = np.random.uniform(bounds[2][0], bounds[2][1]) * np.pi / 2

        # Convert to Cartesian coordinates
        x, y, z = self.__to_cartesian(r, phi, theta)

        # Publish the new target position
        target_msg = Point()
        target_msg.x = x
        target_msg.y = y
        target_msg.z = z
        self.target_publisher_.publish(target_msg)

        # Update internal target state immediately and return success or failure
        lock = getattr(self, '_lock', None)
        if lock is None:
            self._target = target_msg
        else:
            with lock:
                self._target = target_msg

        return True

    def get_target_position(self) -> Optional[Tuple[float, float, float]]:
        """
        Get the current target position as a tuple (x, y, z).
        Returns None if the target position is not yet available.
        """
        return self.target

    def get_manipulator_position(self) -> Optional[Tuple[float, float, float]]:
        """
        Get the current manipulator position as a tuple (x, y, z).
        Returns None if the manipulator position is not yet available.
        """
        return self.manipulator_position

    def get_joint_positions(self) -> Optional[Tuple[float, ...]]:
        """Get current joint positions as a tuple (7 values) if available."""
        lock = getattr(self, '_lock', None)
        if lock is None:
            return getattr(self, '_joint_positions', None)
        with lock:
            return self._joint_positions

    def get_joint_velocities(self) -> Optional[Tuple[float, ...]]:
        """Get current joint velocities as a tuple (7 values) if available."""
        lock = getattr(self, '_lock', None)
        if lock is None:
            return getattr(self, '_joint_velocities', None)
        with lock:
            return self._joint_velocities

    def get_joint_efforts(self) -> Optional[Tuple[float, ...]]:
        """Get current joint efforts as a tuple (7 values) if available."""
        lock = getattr(self, '_lock', None)
        if lock is None:
            return getattr(self, '_joint_efforts', None)
        with lock:
            return self._joint_efforts

    def __to_cartesian(self, r, phi, theta):
        """
        Utility function to convert polar coordinates (r, phi, theta) to Cartesian coordinates (x, y, z).
        """
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.cos(theta)
        z = r * np.sin(theta)
        return x, y, z

    def publish_command(self, action: np.ndarray) -> bool:
        """
        Publish message in the /forward_position_controller/commands topic to move the manipulator joints.
        The action is expected to be a 7D vector of joint positions.
        """
        if not getattr(self, '_ros_ready', False):
            return False
        msg = Float64MultiArray()
        msg.data = action.tolist()
        self.commands_publisher_.publish(msg)
        return True