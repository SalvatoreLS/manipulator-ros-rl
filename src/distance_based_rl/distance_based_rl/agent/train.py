"""Train the SAC agent with the ROS2 environment."""

import argparse
import json
import os
import sys
from datetime import datetime

# Check for torch and gym dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. Install with: pip install tensorboard")

import gymnasium as gym

# Import config (no environment dependencies)
try:
    from config import TrainingConfig
except ImportError:
    # If not found, use local definition (for direct execution)
    from distance_based_rl.agent.config import TrainingConfig

# Handle imports for both direct execution and module import
try:
    from distance_based_rl.environment.arm_env import ManipulatorEnv, State, StateActionReward
    from distance_based_rl.agent.sac_agent import SACAgent
except ImportError:
    from arm_env import ManipulatorEnv, State, StateActionReward
    from sac_agent import SACAgent


def setup_logging(output_dir):
    """Set up tensorboard logging."""
    if not TENSORBOARD_AVAILABLE:
        return None

    log_dir = os.path.join(output_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to {log_dir}")
    print(f"View with: tensorboard --logdir {log_dir}")
    return writer


def train(config=None):
    """Train the SAC agent with ROS2 environment."""
    if config is None:
        config = TrainingConfig()

    print("=" * 60)
    print("Distance-Based RL Training")
    print("=" * 60)
    print(config)
    print()

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(config.output_dir, 'config.json')
    config.save(config_path)

    # Setup tensorboard
    writer = setup_logging(config.output_dir) if config.use_tensorboard else None

    # Initialize environment and agent
    print("Initializing environment and agent...")
    env = ManipulatorEnv()
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=config.hidden_dim,
        lr=config.learning_rate,
        buffer_size=config.buffer_size
    )
    print(f"✓ Environment initialized: observation_space={env.observation_space.shape}, "
          f"action_space={env.action_space.shape}")
    print(f"✓ Agent initialized: hidden_dim={config.hidden_dim}, "
          f"learning_rate={config.learning_rate}")
    print()

    # Training loop
    print("Starting training...")
    print("-" * 60)

    episode_rewards = []

    try:
        for episode in range(config.num_episodes):
            state, info = env.reset()  # [ROS] New random target position in ROS2 environment
            state_obj = info.get('state')
            done = False
            total_reward = 0
            steps = 0

            while not done and steps < config.max_steps_per_episode:
                # Select and execute action
                action = agent.select_action(state_obj if state_obj is not None else state)
                next_state, reward, terminated, truncated, step_info = env.step(action) # [ROS] Step in ROS2 environment
                next_state_obj = step_info.get('state')
                done = terminated or truncated

                # Store transition in replay buffer
                if state_obj is not None and next_state_obj is not None:
                    transition = StateActionReward(
                        state=state_obj,
                        action=action,
                        reward=reward,
                        next_state=next_state_obj,
                        done=done,
                    )
                    agent.replay_buffer.push(transition)
                else:
                    agent.replay_buffer.push(state, action, reward, next_state, done)

                # Optimize if buffer has enough samples
                if len(agent.replay_buffer) >= config.batch_size:
                    agent.optimize(batch_size=config.batch_size)

                total_reward += reward
                state = next_state
                state_obj = next_state_obj
                steps += 1

            # Track episode metrics
            episode_rewards.append(total_reward)

            # Log to tensorboard
            if writer is not None:
                writer.add_scalar('rewards/episode', total_reward, episode)
                writer.add_scalar('buffer_size', agent.get_buffer_size(), episode)
                writer.add_scalar('episode_steps', steps, episode)

            # Print progress
            if (episode + 1) % max(1, config.num_episodes // 10) == 0:
                avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
                print(f"Episode {episode + 1}/{config.num_episodes} | "
                      f"Total Reward: {total_reward:.2f} | "
                      f"Avg Reward (last 10): {avg_reward:.2f} | "
                      f"Steps: {steps} | "
                      f"Buffer Size: {agent.get_buffer_size()}")

            # Save checkpoint
            if (episode + 1) % config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    config.output_dir,
                    f'checkpoint_episode_{episode + 1}.pt'
                )
                agent.save_model(checkpoint_path)

    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")

    finally:
        # Save final model
        final_model_path = os.path.join(config.output_dir, 'final_model.pt')
        agent.save_model(final_model_path)

        # Close environment
        env.close()

        # Close tensorboard writer
        if writer is not None:
            writer.close()
            print(f"TensorBoard writer closed")

    print("-" * 60)
    print(f"Training completed!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Results saved to: {config.output_dir}")
    print("=" * 60)


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description='Train SAC agent with distance-based reward in ROS2 environment'
    )
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Number of training episodes (default: 1000)')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode (default: 500)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for optimization (default: 256)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate for optimizer (default: 3e-4)')
    parser.add_argument('--buffer-size', type=int, default=10000,
                        help='Replay buffer capacity (default: 10000)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden layer dimension for policy network (default: 256)')
    parser.add_argument('--output-dir', type=str, default='output/',
                        help='Output directory for results (default: output/)')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                        help='Save checkpoint every N episodes (default: 50)')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Disable tensorboard logging')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to model to load before training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON file to load')

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        print(f"Loading configuration from {args.config}...")
        config = TrainingConfig.load(args.config)
    else:
        config = TrainingConfig(
            num_episodes=args.num_episodes,
            max_steps_per_episode=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            hidden_dim=args.hidden_dim,
            output_dir=args.output_dir,
            checkpoint_interval=args.checkpoint_interval,
            use_tensorboard=not args.no_tensorboard,
        )

    # Train agent
    train(config)

    # Optionally load model for evaluation
    if args.load_model:
        print(f"\nLoading model from {args.load_model}...")
        agent = SACAgent(state_dim=State.vector_dim(), action_dim=7)
        agent.load_model(args.load_model)
        print("Model loaded successfully")


if __name__ == "__main__":
    main()