#!/bin/bash

# Configuration
CONTAINER_NAME="franka_ros2_rl"
COMPOSE_FILE="docker-compose.yaml"
WORKDIR="/ros2_rl_ws"
WRAPPER="/entrypoint.sh"

# Colors for output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}=== Franka ROS2 Simulation Setup ===${NC}"

# Cleanup function to be called on Ctrl+C (SIGINT)
cleanup() {
    echo -e "\n${RED}Caught signal! Shutting down all nodes...${NC}"
    
    # 1. Stop processes inside the container
    echo -e "${CYAN}Stopping ROS nodes...${NC}"
    docker exec $CONTAINER_NAME pkill -SIGINT -f "ros_gz_bridge"
    docker exec $CONTAINER_NAME pkill -SIGINT -f "ros2 launch franka_gazebo_bringup"
    docker exec $CONTAINER_NAME pkill -SIGINT -f "move_keyboard"
    
    # 2. Kill the host processes (docker exec commands)
    [ -n "$BRIDGE_PID" ] && kill $BRIDGE_PID 2>/dev/null
    [ -n "$LAUNCH_PID" ] && kill $LAUNCH_PID 2>/dev/null
    
    # 3. Wait for graceful shutdown of nodes
    sleep 2
    
    # 4. Stop the container
    echo -e "${CYAN}Stopping Docker container...${NC}"
    docker compose -f $COMPOSE_FILE stop
    
    echo -e "${GREEN}Cleanup complete. Goodbye!${NC}"
    exit 0
}

# Trap Ctrl+C (SIGINT)
trap cleanup SIGINT

# 1. Start Docker container
echo -e "${CYAN}1. Launching Docker container...${NC}"
docker compose -f $COMPOSE_FILE up -d

# Check if the container is running
if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo -e "${RED}Error: Failed to start container $CONTAINER_NAME.${NC}"
    exit 1
fi
echo -e "${GREEN}Container is up.${NC}"

# 2. Run the bridge
echo -e "${CYAN}2. Starting ROS-GZ Bridge...${NC}"
docker exec -i $CONTAINER_NAME $WRAPPER ros2 run ros_gz_bridge parameter_bridge \
     --ros-args -p config_file:=$WORKDIR/config/bridge.yaml &
BRIDGE_PID=$!

# Wait for bridge initialization
sleep 2

# 3. Run the simulation
echo -e "${CYAN}3. Starting Simulation (RViz + Gazebo)...${NC}"
docker exec -i $CONTAINER_NAME $WRAPPER ros2 launch franka_gazebo_bringup gazebo_franka_arm_example_controller.launch.py \
     load_gripper:=true \
     controller:=forward_position_controller \
     rviz:=true &
LAUNCH_PID=$!

# Wait for simulation to come up
echo -e "${CYAN}Waiting for simulation to initialize...${NC}"
sleep 5

# 4. Run the keyboard movement node (interactive)
echo -e "${CYAN}4. Starting Keyboard Controller...${NC}"
echo -e "${GREEN}Nodes are running! Logs are visible. Focus this window to use keyboard controls.${NC}"
echo -e "${GREEN}Press Ctrl+C to stop all nodes and the container.${NC}"

# We use -it for the interactive keyboard node
docker exec -it $CONTAINER_NAME $WRAPPER ros2 run keyboard_movement move_keyboard

# Clean up will be triggered by trap on Ctrl+C or when the above command finishes
cleanup
