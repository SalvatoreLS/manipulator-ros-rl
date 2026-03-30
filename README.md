# Reinforcement Learning Manipulator ROS2

This project uses a ROS2 Gazebo environment to simulate and train a manipulator (Franka arm) using reinforcement learning.

## Quick Start

### 1. Launch the Docker Environment
```bash
docker-compose build
docker-compose run franka-rl
```

### 2. Start the Gazebo Simulation
```bash
ros2 launch franka_gazebo_bringup gazebo_joint_position_controller_example.launch.py load_gripper:=true
```

### 3. Send Control Commands
Open a new terminal and enter the container:
```bash
docker-compose exec franka-rl bash
```

Then send a command to move the arm:
```bash
ros2 topic pub /panda_arm_controller/commands std_msgs/Float64MultiArray "{layout: {dim: [], data_offset: 0}, data: [0.0, 0.0, 0.0, -2.0, 0.0, 2.0, 0.0]}"
```

Watch the arm move in the Gazebo window!

## Documentation

- **[VISUALIZATION.md](./VISUALIZATION.md)** - Complete guide to launching and controlling the robot in simulation
- **[DOCKER.md](./DOCKER.md)** - Docker setup and configuration
- **[TODO.md](./TODO.md)** - Project roadmap

## Project Overview

This is a collaborative robotics simulation environment for reinforcement learning research. The system combines:

- **Gazebo**: Physics engine for realistic robot dynamics
- **ROS 2**: Middleware for robot communication and control
- **Franka Arm**: Industrial collaborative manipulator
- **RViz**: Visualization of robot state
- **RL Framework**: Gymnasium + Stable-Baselines3 for training agents

## Key Features

- Full Gazebo physics simulation with gravity
- 7-DOF Franka arm with optional gripper
- Real-time ROS 2 control interface
- Multiple controller options (position, velocity, impedance)
- Pre-configured Docker environment with all dependencies
- Support for reinforcement learning training