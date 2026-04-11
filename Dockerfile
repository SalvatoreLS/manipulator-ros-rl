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

# Install PyTorch for CUDA 12.8 explicitly before other packages so that
# stable-baselines3 does not pull in a newer wheel compiled for CUDA 13.0+
# (which exceeds what the host NVIDIA driver supports).
# T2000 (Turing/sm_75) is fully supported by CUDA 12.x.
RUN pip3 install \
    "torch==2.7.0+cu128" \
    "torchvision==0.22.0+cu128" \
    "torchaudio==2.7.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128

# Install RL Python stack
RUN pip3 install \
    gymnasium \
    stable-baselines3 \
    shimmy \
    tensorboard \
    tqdm \
    keyboard \
    pynput

WORKDIR /ros2_rl_ws

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]