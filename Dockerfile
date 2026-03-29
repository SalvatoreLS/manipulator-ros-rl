FROM osrf/ros:humble-desktop-full

# I install Gazebo Fortress and Franka dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-ros-gz \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-franka-description \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# I install the RL stack
RUN pip3 install \
    gymnasium \
    stable-baselines3 \
    shimmy \
    tensorboard

# I set up the workspace directory
WORKDIR /ros2_rl_ws

# I copy the entrypoint script into the container
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# I set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# I default to bash if no command is provided
CMD ["bash"]