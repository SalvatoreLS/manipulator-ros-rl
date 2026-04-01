import gymnasium as gym
from arm_env import ManipulatorEnv
from sac_agent import SACAgent

NUM_EPISODES = 1000
OUTPUT_DIR : str = "output/"
# TODO: see if I should add a finite number of steps per episode to prevent infinite loops.
# TODO: synchronize training with target updates to make the new target position available at the beginning of each episode. This can be done by adding a ROS2 service that requests a new random target position at the beginning of each episode, and the service can be called from the reset() method of the environment.

def train():
    env = ManipulatorEnv()
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=256)

    num_episodes = NUM_EPISODES

    for episode in range(num_episodes):
        state, _ = env.reset() # new random target position
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)  # Replace with your RL agent's action selection
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    train()