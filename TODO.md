## 1. System Requirements & Versions
I use the following stack for stability on Jammy Jellyfish:

| Component | Version |
| :--- | :--- |
| **OS** | Ubuntu 22.04 (Jammy Jellyfish) |
| **ROS 2** | Humble Hawksbill |
| **Simulator** | Gazebo Fortress (Ignition) |
| **RL Library** | Stable Baselines3 (SB3) |
| **Interface** | Gymnasium |

---

## 2. Environment Installation
I install the necessary packages to bridge the simulation and control layers.

```bash
# Update and install ROS 2 Humble
sudo apt update && sudo apt install ros-humble-desktop -y

# Install Gazebo Fortress and ROS-GZ Bridge
sudo apt install ros-humble-ros-gz -y

# Install ros2_control and Franka description
sudo apt install ros-humble-ros2-control \
                 ros-humble-ros2-controllers \
                 ros-humble-franka-description -y

# Install RL dependencies
pip install gymnasium stable-baselines3 shimmy
```

---

## 3. Workspace Setup
I create a workspace and clone the necessary simulation logic.

```bash
mkdir -p ~/ros2_rl_ws/src
cd ~/ros2_rl_ws/src
# Clone a compatible panda_gazebo package if not using the built-in description
# For this guide, I assume you have a URDF configured for ros2_control and gz_ros2_control.
```

### The `ros_gz_bridge` Configuration
The bridge allows Python to "talk" to Gazebo directly or through ROS topics. I use a `bridge.yaml` file to map these:

```yaml
# bridge.yaml
- ros_topic_name: "/joint_states"
  gz_topic_name: "/world/default/model/panda/joint_state"
  ros_type_name: "sensor_msgs/msg/JointState"
  gz_type_name: "gz.msgs.Model"
  direction: GZ_TO_ROS
```

---

## 4. Communication Architecture
The RL loop follows a specific data flow: The Agent sends actions to ROS 2 controllers, Gazebo physics steps forward, and the bridge returns the new state.



---

## 5. Gymnasium Environment Implementation
I wrap the ROS 2 node inside a Gymnasium class. I use `MultiDiscrete` for simple tasks or `Box` for continuous joint control.

```python
import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from gymnasium import spaces

class PandaRobotEnv(gym.Env):
    def __init__(self):
        super(PandaRobotEnv, self).__init__()
        if not rclpy.ok(): rclpy.init()
        self.node = Node('panda_rl_node')
        
        # 7 Joints for Panda
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        # Observations: 7 positions + 7 velocities
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        
        self.pub = self.node.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)
        self.sub = self.node.create_subscription(JointState, '/joint_states', self._obs_callback, 10)
        
        self.state = np.zeros(14)
        self.goal = np.array([0.5, 0.0, 0.5]) # Example Cartesian goal

    def _obs_callback(self, msg):
        self.state = np.array(msg.position + msg.velocity)

    def step(self, action):
        # I scale action to robot joint limits
        cmd = Float64MultiArray()
        cmd.data = action.tolist()
        self.pub.publish(cmd)
        
        # I step the simulation (if using Gazebo stepping plugin) or wait for real-time
        rclpy.spin_once(self.node, timeout_sec=0.05)
        
        reward = self._compute_reward()
        terminated = self._is_done()
        
        return self.state, reward, terminated, False, {}

    def _compute_reward(self):
        # I use negative Euclidean distance
        dist = np.linalg.norm(self.state[:3] - self.goal) # Simplified
        return -dist

    def reset(self, seed=None, options=None):
        # I call the Gazebo reset service here
        return self.state, {}
```

---

## 6. Training the Agent
I use PPO (Proximal Policy Optimization) for training.

```python
from stable_baselines3 import PPO

env = PandaRobotEnv()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_panda_tensorboard/")

# I train for 100k steps
model.learn(total_timesteps=100000)

# I save the weights
model.save("ppo_panda_manipulator")
```

---

## 7. Testing and Validation
To test, I load the saved model and run the environment loop without the `model.learn()` call.

1.  **Verify Control:** Run `ros2 topic list` to ensure `/forward_position_controller/commands` is available.
2.  **Monitor Rewards:** Use Tensorboard to check convergence:
    `tensorboard --logdir ./ppo_panda_tensorboard/`
3.  **Visual Check:** I observe the robot in Gazebo. If it jitters, I decrease the `learning_rate` or increase the controller frequency in the `ros2_control` URDF tags.

---

## 8. Reward Function Math
I use a dense reward to prevent the "sparse reward problem" where the robot moves randomly without learning.

$$R = -(w_{dist} \cdot d(ee, goal)) - (w_{effort} \cdot \|\tau\|^2) + C_{success}$$

Where:
* $d(ee, goal)$ is the distance from end-effector to target.
* $\|\tau\|^2$ penalizes high torques to encourage smooth motion.
* $C_{success}$ is a large positive constant given only when the goal is reached.