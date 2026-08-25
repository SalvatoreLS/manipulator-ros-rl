"""
Monitor manipulator state and target position.

Provide the collected information to the reinforcement learning agent.
"""

import rclpy
import rclpy.time
import threading
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import Point
from typing import Optional, Tuple
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import tf2_ros
import numpy as np

from .env_config import EnvConfig

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
    Interface node for ROS2 communication.

    Handles subscriptions to the manipulator's state and target position, and provides
    methods to call services and publish commands.

    It maintains the current state of the manipulator and the target position, which can be accessed by the RL agent.
    """

    def __init__(self, config: Optional[EnvConfig] = None, seed: Optional[int] = None):
        """
        Initialize the ROS2 node, subscribers, and publishers.

        Set up thread-safe access to the manipulator state and target position.

        ``config`` holds the tunable environment parameters and defaults to
        ``EnvConfig()``, which reads the same environment variables the shell scripts
        already set.  ``seed`` seeds this node's private random generator, which drives
        target sampling: a dedicated Generator (rather than the global ``np.random``
        state) is what makes a run reproducible from its seed alone.
        """
        self.config = config if config is not None else EnvConfig()

        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self._manipulator_position: Optional[Point] = None
        self._target: Optional[Point] = None
        self._joint_positions: Optional[Tuple[float, ...]] = None
        self._joint_velocities: Optional[Tuple[float, ...]] = None
        self._joint_efforts: Optional[Tuple[float, ...]] = None
        self._ros_ready = False
        # True once state_callback (franka_robot_state_broadcaster) fires — suppresses TF-based EE updates
        self._ee_from_state_cb = False

        # Intiialization of ROS2 node
        try:
            super().__init__('data_monitor')
        except Exception:
            return

        # Callback group for thread safety
        cb_group = MutuallyExclusiveCallbackGroup()

        robot_state_topic = self.config.robot_state_topic

        # Subscription for manipulator state
        self.create_subscription(
            _RobotStateType,
            robot_state_topic,
            self.state_callback,
            10,
            callback_group=cb_group
        )
        self.get_logger().info(f'Subscribed to robot state topic: {robot_state_topic}')

        # Subscription for target position (to compute reward in env)
        self.create_subscription(
            Point,
            self.config.target_topic,
            self.target_callback,
            10,
            callback_group=cb_group
        )

        # TF2 buffer + listener: used to compute EE position when franka_robot_state_broadcaster
        # is unavailable (e.g. Gazebo, where 'fr3/robot_state' hardware interface doesn't exist).
        # robot_state_publisher reads /joint_states and publishes the full TF tree, so the EE
        # frame (fr3_link8 in world) is available as long as joint_state_broadcaster is running.
        self._ee_frame = self.config.ee_frame
        self._base_frame = self.config.base_frame
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Subscription for joint states — fallback for joint data and EE position in Gazebo
        js_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            JointState,
            self.config.joint_states_topic,
            self._joint_state_callback,
            10,
            callback_group=js_cb_group,
        )

        # Publisher for target position (changed when resetting env)
        self.target_publisher_ = self.create_publisher(Point, self.config.target_topic, 10)

        # Publisher for action commands. It moves the joints of the manipulator (7D vector)
        self.commands_publisher_ = self.create_publisher(
            Float64MultiArray, self.config.command_topic, 10
        )

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
        if position is None:
            return None
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
        if target is None:
            return None
        return (target.x, target.y, target.z)

    def state_callback(self, msg):
        """
        Process messages from franka_robot_state_broadcaster.

        It extracts the end-effector position field and updates the state inside the node.
        This is the primary source of EE position on real hardware; it is NOT available in
        Gazebo (the 'fr3/robot_state' hardware interface doesn't exist there).
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
                self._ee_from_state_cb = True
            else:
                with lock:
                    self._manipulator_position = p
                    self._joint_positions = joint_positions
                    self._joint_velocities = joint_velocities
                    self._joint_efforts = joint_efforts
                    self._ee_from_state_cb = True

    def _joint_state_callback(self, msg):
        """
        Handle /joint_states messages from joint_state_broadcaster.

        Always updates joint positions/velocities/efforts for the 7 arm joints.
        When franka_robot_state_broadcaster is not available (Gazebo), also looks up the EE
        position via TF2 using transforms published by robot_state_publisher.
        """
        _JOINT_ORDER = (

            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7',
        )

        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        if not all(j in name_to_idx for j in _JOINT_ORDER):
            return  # Expected joints not in this message

        indices = [name_to_idx[j] for j in _JOINT_ORDER]

        positions = tuple(msg.position[i] for i in indices) if msg.position else None
        velocities = (
            tuple(msg.velocity[i] for i in indices)
            if msg.velocity and len(msg.velocity) > max(indices) else None
        )
        efforts = (
            tuple(msg.effort[i] for i in indices)
            if msg.effort and len(msg.effort) > max(indices) else None
        )

        # Look up EE position via TF only when franka_robot_state_broadcaster hasn't fired
        ee_position = None
        with self._lock:
            needs_ee = not self._ee_from_state_cb
        if needs_ee:
            try:
                t = self._tf_buffer.lookup_transform(
                    self._base_frame,
                    self._ee_frame,
                    rclpy.time.Time(),
                )
                p = Point()
                p.x = t.transform.translation.x
                p.y = t.transform.translation.y
                p.z = t.transform.translation.z
                ee_position = p
            except Exception:
                pass  # TF not yet available; will retry on next callback

        with self._lock:
            if positions is not None:
                self._joint_positions = positions
            if velocities is not None:
                self._joint_velocities = velocities
            if efforts is not None:
                self._joint_efforts = efforts
            if ee_position is not None and not self._ee_from_state_cb:
                self._manipulator_position = ee_position

    def target_callback(self, msg):
        """
        Update the target position inside the node.

        Same as state_callback, but for the target published on /manipulator_target.
        """
        lock = getattr(self, '_lock', None)

        if lock is None:
            self._target = msg
        else:
            with lock:
                self._target = msg

    def seed(self, seed: Optional[int]) -> None:
        """Reseed the private generator used for target sampling."""
        self._rng = np.random.default_rng(seed)

    def set_random_target(
        self,
        bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
    ) -> bool:
        """
        Set a new random target position by publishing to /manipulator_target.

        Returns True if successful, False otherwise.

        The position is generated in spherical coordinates and converted to Cartesian (x, y, z).
        Sampling uses this node's private ``np.random.Generator`` (see ``seed``), not the
        global numpy state, so a seeded run replays the same target sequence.

        bounds: defaults to the radius / azimuth / elevation bounds in ``EnvConfig``.
            bounds[0] = (r_min, r_max)            — radial distance in metres.
                        Kept within [0.2, 0.75] to stay inside the FR3 workspace
                        (~0.855 m max reach) and away from the near-base singularity.
                        Sampled via cube-root transform for uniform volume coverage.
            bounds[1] = (phi_scale_min, phi_scale_max) — azimuth as fraction of 2π.
                        (-5/12, 5/12) → ±150° centred at phi=0° (robot forward / +x axis),
                        excluding ±30° behind the robot to avoid joint1 limits (±166°)
                        and rear singularities. Negative values wrap correctly.
            bounds[2] = (theta_scale_min, theta_scale_max) — elevation as fraction of π/2.
                        (0.25, 0.78) → 22.5°–70° above the horizontal plane, keeping targets
                        off the floor and away from the overhead singularity.
                        Sampled via arcsin transform for uniform solid-angle coverage.
                        Guaranteed floor clearance is r_min·sin(22.5°) ≈ 0.077 m (at the
                        smallest radius); a target at r = 0.5 clears ≈ 0.19 m.
        """
        if not getattr(self, '_ros_ready', False):

            return False

        if bounds is None:
            bounds = (
                self.config.radius_bounds,
                self.config.azimuth_bounds,
                self.config.elevation_bounds,
            )

        rng = self._rng

        # Uniform volume: P(r) ∝ r² → sample r³ uniformly then take cube root.
        r = float(rng.uniform(bounds[0][0] ** 3, bounds[0][1] ** 3) ** (1.0 / 3.0))

        # Azimuth centred at phi=0° (robot forward); negative values handled by numpy.
        phi = rng.uniform(bounds[1][0], bounds[1][1]) * 2.0 * np.pi

        # Uniform solid angle: P(θ_elev) ∝ cos(θ_elev) → sample sin uniformly then arcsin.
        sin_lo = np.sin(bounds[2][0] * np.pi / 2.0)
        sin_hi = np.sin(bounds[2][1] * np.pi / 2.0)
        theta = float(np.arcsin(rng.uniform(sin_lo, sin_hi)))

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
        """Convert spherical coordinates (r, phi, theta) to Cartesian (x, y, z)."""
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.cos(theta)
        z = r * np.sin(theta)
        return x, y, z

    # Franka Panda joint position limits [low, high] in radians (7 joints).
    _JOINT_LOW  = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
    _JOINT_HIGH = np.array([2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], dtype=np.float64)

    # Franka "ready" home configuration (rad) — safe, singularity-free starting pose.
    _HOME_POSITION = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)

    def publish_command(self, action: np.ndarray) -> bool:
        """
        Publish a command to /forward_position_controller/commands to move the manipulator joints.

        Action is a 7D vector in [-1, 1] interpreted as an *incremental* (delta) joint
        move: joint_target = current_joint_positions + action * config.max_joint_delta,
        clipped to the Franka joint limits. Incremental control (vs absolute mapping) keeps
        per-step motion small and physically reachable, which makes the reward consistent
        with the action and the dynamics smooth.

        Returns False if ROS2 is not ready or current joint feedback is unavailable.
        """
        if not getattr(self, '_ros_ready', False):

            return False
        current = self.get_joint_positions()
        if current is None:
            # Cannot apply incremental control without knowing the current configuration.
            return False
        current = np.asarray(current, dtype=np.float64)
        action_clipped = np.clip(action, -1.0, 1.0)
        joint_target = current + action_clipped * self.config.max_joint_delta
        joint_target = np.clip(joint_target, self._JOINT_LOW, self._JOINT_HIGH)
        msg = Float64MultiArray()
        msg.data = joint_target.tolist()
        self.commands_publisher_.publish(msg)
        return True

    def go_home(self) -> bool:
        """
        Publish the fixed home joint configuration so the arm returns to a known pose.

        Used to reset the arm at the start of each *training* episode (not at inference).
        Returns True if the command was published, False if ROS2 is not ready.
        """
        if not getattr(self, '_ros_ready', False):

            return False
        msg = Float64MultiArray()
        msg.data = self._HOME_POSITION.tolist()
        self.commands_publisher_.publish(msg)
        return True
