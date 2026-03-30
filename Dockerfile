FROM osrf/ros:humble-desktop-full

# Install Gazebo/Franka and build dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-ros-gz \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-franka-description \
    ros-humble-pinocchio \
    libpoco-dev \
    python3-vcstool \
    python3-pip \
    ros-humble-ament-cmake-clang-format \
    ros-humble-ament-cmake-clang-tidy \
    ros-humble-moveit \
    ros-humble-moveit-core \
    ros-humble-moveit-ros-planning \
    ros-humble-moveit-ros-planning-interface \
    ros-humble-moveit-kinematics \
    ros-humble-moveit-planners-ompl \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-ros-visualization \
    ros-humble-gz-ros2-control \
    ros-humble-controller-manager \
    ros-humble-moveit-msgs \
    ros-humble-moveit-core \
    ros-humble-moveit-ros-planning \
    ros-humble-moveit-ros-planning-interface \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-core \
    ros-humble-moveit-ros-planning \
    ros-humble-moveit-ros-planning-interface \
    ros-humble-moveit-kinematics \
    ros-humble-moveit-planners-ompl \
    ros-humble-moveit-simple-controller-manager \
    ros-humble-moveit-ros-visualization \
    ros-humble-joint-trajectory-controller \
    && rm -rf /var/lib/apt/lists/*

# Install RL Python stack
RUN pip3 install \
    gymnasium \
    stable-baselines3 \
    shimmy \
    tensorboard

WORKDIR /ros2_rl_ws

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]