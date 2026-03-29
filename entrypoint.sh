#!/bin/bash
set -e

# I source the main ROS 2 Humble installation
source "/opt/ros/humble/setup.bash"

# I source the local workspace if the install directory exists
if [ -f "/ros2_rl_ws/install/setup.bash" ]; then
  source "/ros2_rl_ws/install/setup.bash"
fi

# I execute the command passed to the docker container (e.g., bash or a python script)
exec "$@"