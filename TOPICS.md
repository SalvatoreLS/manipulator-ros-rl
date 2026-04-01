# ROS2 Topics and Types

This document lists the ROS2 topics, services, and actions used or provided by the various components in this workspace.

## 1. Franka Robot State Broadcaster
Default Node Namespace: `/franka_robot_state_broadcaster`

| Topic | Type | Description |
|---|---|---|
| `~/robot_state` | `franka_msgs/msg/FrankaRobotState` | Comprehensive state of the Franka robot (contains EE pose, joint states, etc.) |
| `~/current_pose` | `geometry_msgs/msg/PoseStamped` | Current end-effector pose (O_T_EE) |
| `~/last_desired_pose` | `geometry_msgs/msg/PoseStamped` | Last desired end-effector pose (O_T_EE_d) |
| `~/desired_end_effector_twist` | `geometry_msgs/msg/TwistStamped` | Desired end-effector twist |
| `~/measured_joint_states` | `sensor_msgs/msg/JointState` | Measured joint positions, velocities, and efforts |
| `~/external_wrench_in_stiffness_frame` | `geometry_msgs/msg/WrenchStamped` | Estimated external wrench at the stiffness frame |
| `~/external_wrench_in_base_frame` | `geometry_msgs/msg/WrenchStamped` | Estimated external wrench at the base frame |
| `~/external_joint_torques` | `sensor_msgs/msg/JointState` | Estimated external joint torques |
| `~/desired_joint_states` | `sensor_msgs/msg/JointState` | Desired joint positions, velocities, and efforts |

## 2. Franka Gripper
Default Node Name: `/franka_gripper_node` (often namespaced as `/franka_gripper`)

| Interface Name | Interface Type | Message/Service/Action Type | Description |
|---|---|---|---|
| `~/joint_states` | Topic | `sensor_msgs/msg/JointState` | Gripper joint states |
| `~/stop` | Service | `std_srvs/srv/Trigger` | Stops the current gripper command |
| `~/homing` | Action | `franka_msgs/action/Homing` | Performs gripper homing |
| `~/move` | Action | `franka_msgs/action/Move` | Moves gripper to a specific width |
| `~/grasp` | Action | `franka_msgs/action/Grasp` | Grasps an object with specified force |
| `~/gripper_action` | Action | `control_msgs/action/GripperCommand` | Standard ROS gripper command interface |

## 3. Franka Hardware & General Management
Default Node Namespace: `/franka_hardware` or similar

| Interface Name | Interface Type | Message/Service/Action Type | Description |
|---|---|---|---|
| `~/set_joint_stiffness` | Service | `franka_msgs/srv/SetJointStiffness` | Sets the joint stiffness for impedance control |
| `~/set_cartesian_stiffness` | Service | `franka_msgs/srv/SetCartesianStiffness` | Sets the Cartesian stiffness for impedance control |
| `~/set_tcp_frame` | Service | `franka_msgs/srv/SetTCPFrame` | Sets the Tool Center Point (TCP) frame |
| `~/set_stiffness_frame` | Service | `franka_msgs/srv/SetStiffnessFrame` | Sets the stiffness frame |
| `~/set_force_torque_collision_behavior` | Service | `franka_msgs/srv/SetForceTorqueCollisionBehavior` | Sets the collision behavior for force/torque |
| `~/set_full_collision_behavior` | Service | `franka_msgs/srv/SetFullCollisionBehavior` | Sets the full collision behavior |
| `~/set_load` | Service | `franka_msgs/srv/SetLoad` | Sets the mass and inertia of the load |
| `~/error_recovery` | Action | `franka_msgs/action/ErrorRecovery` | Recovers the robot from an error state |

## 4. Reinforcement Learning (RL) & Custom Topics
Topics defined or used specifically for RL training and manipulator movement.

| Topic | Type | Description |
|---|---|---|
| `/manipulator_target` | `geometry_msgs/msg/Point` | The target position for the end-effector (Global) |
| `/forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | Input commands for the position controller (Joint positions) |

## 5. Standard ROS2 & Simulation Topics
Common topics used by the framework and simulated sensors.

| Topic | Type | Description |
|---|---|---|
| `/joint_states` | `sensor_msgs/msg/JointState` | Aggregated joint states of the entire system |
| `/dynamic_joint_states` | `control_msgs/msg/DynamicJointState` | Joint states including non-standard interfaces |
| `/robot_description` | `std_msgs/msg/String` | Robot URDF description |
| `/joy` | `sensor_msgs/msg/Joy` | Joystick input (if used for teleoperation) |
| `/cmd_vel` | `geometry_msgs/msg/TwistStamped` | Velocity commands (if using mobile platform or cartesian velocity) |

## 6. Simulated Sensors (Gazebo)
Topics provided by simulated sensors in Gazebo.

| Topic | Type | Description |
|---|---|---|
| `/camera_front/image_raw` | `sensor_msgs/msg/Image` | Front camera feed |
| `/camera_rear/image_raw` | `sensor_msgs/msg/Image` | Rear camera feed |
| `/camera_left/image_raw` | `sensor_msgs/msg/Image` | Left camera feed |
| `/camera_right/image_raw` | `sensor_msgs/msg/Image` | Right camera feed |
| `/imu/data` | `sensor_msgs/msg/Imu` | IMU sensor data |
| `/{robot_name}/scan` | `sensor_msgs/msg/LaserScan` | Lidar/Laser scan data |
