import gymnasium as gym
from arm_env import ManipulatorEnv

NUM_EPISODES = 1000

def train():
    env = ManipulatorEnv()
    num_episodes = NUM_EPISODES

    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()  # Replace with your RL agent's action selection
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    train()