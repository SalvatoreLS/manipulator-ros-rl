"""Train the SAC agent with the ROS2 environment."""

import argparse
import os
import warnings
from collections import deque
from datetime import datetime

# PyTorch's autograd engine probes for CUDA devices during every backward pass,
# even when all tensors live on CPU.  When the host NVIDIA driver version doesn't
# match the CUDA runtime bundled in the PyTorch wheel this produces a noisy
# UserWarning that is harmless for CPU-only training.  Filter it here, before any
# torch import, so it never reaches the user's terminal.
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='CUDA initialization',
)

# Optional progress-bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: install tqdm for richer progress bars  →  pip install tqdm")

# Optional tensorboard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. Install with: pip install tensorboard")

import gymnasium as gym  # noqa: F401 – keep for env registration side-effects

# Import config (no environment dependencies)
try:
    from config import TrainingConfig
except ImportError:
    from distance_based_rl.agent.config import TrainingConfig

# Handle imports for both direct execution and module import
try:
    from distance_based_rl.environment.arm_env import ManipulatorEnv, State, StateActionReward
    from distance_based_rl.agent.sac_agent import SACAgent
except ImportError:
    from arm_env import ManipulatorEnv, State, StateActionReward
    from sac_agent import SACAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Print *msg* without breaking active tqdm bars."""
    if TQDM_AVAILABLE:
        tqdm.write(msg)
    else:
        print(msg)


def setup_logging(output_dir: str):
    """Set up tensorboard logging and return a SummaryWriter (or None)."""
    if not TENSORBOARD_AVAILABLE:
        return None

    log_dir = os.path.join(output_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    _log(f"TensorBoard logs  → {log_dir}")
    _log(f"  tensorboard --logdir {log_dir}")
    return writer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config=None):
    """Train the SAC agent with ROS2 environment."""
    if config is None:
        config = TrainingConfig()

    print("=" * 60)
    print("Distance-Based RL Training")
    print("=" * 60)
    print(config)
    print()

    # Persist configuration
    os.makedirs(config.output_dir, exist_ok=True)
    config_path = os.path.join(config.output_dir, 'config.json')
    config.save(config_path)

    # Tensorboard
    writer = setup_logging(config.output_dir) if config.use_tensorboard else None

    # Resolve compute device
    import torch
    _cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    _cuda_disabled = _cuda_env is not None and _cuda_env.strip() == ''
    device = 'cpu' if (_cuda_disabled or not torch.cuda.is_available()) else 'cuda'

    # Build environment and agent
    print("Initializing environment and agent...")
    env = ManipulatorEnv()
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=config.hidden_dim,
        lr=config.learning_rate,
        buffer_size=config.buffer_size,
        device=device,
    )
    print(f"  device       : {device}")
    print(f"  obs / act    : {env.observation_space.shape} / {env.action_space.shape}")
    print(f"  hidden_dim   : {config.hidden_dim}")
    print(f"  learning_rate: {config.learning_rate}")
    print()
    print("Starting training ...")
    print("-" * 60)

    # Running statistics
    episode_rewards: list[float] = []
    recent_rewards: deque[float] = deque(maxlen=10)
    best_reward = float('-inf')

    # ── outer bar: one tick per episode ─────────────────────────────────────
    ep_bar = (
        tqdm(
            range(config.num_episodes),
            desc="Training",
            unit="ep",
            dynamic_ncols=True,
            colour="cyan",
        )
        if TQDM_AVAILABLE
        else range(config.num_episodes)
    )

    try:
        for episode in ep_bar:
            state, info = env.reset()          # [ROS] new random target in ROS2
            state_obj = info.get('state')
            done = False
            total_reward = 0.0
            steps = 0
            updates = 0

            # ── inner bar: one tick per environment step ─────────────────────
            step_bar = (
                tqdm(
                    total=config.max_steps_per_episode,
                    desc=f"  ep {episode + 1:>{len(str(config.num_episodes))}}",
                    unit="step",
                    leave=False,
                    dynamic_ncols=True,
                    bar_format=(
                        "{l_bar}{bar}| {n_fmt}/{total_fmt}"
                        " [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                    ),
                )
                if TQDM_AVAILABLE
                else None
            )

            # ── episode rollout ──────────────────────────────────────────────
            while not done and steps < config.max_steps_per_episode:
                action = agent.select_action(state_obj if state_obj is not None else state)
                next_state, reward, terminated, truncated, step_info = env.step(action)  # [ROS]
                next_state_obj = step_info.get('state')
                done = terminated or truncated

                # Store transition
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

                # Gradient updates — perform gradient_steps updates per env step
                # to saturate the GPU while the slow ROS2/Gazebo env is the bottleneck.
                if len(agent.replay_buffer) >= config.batch_size:
                    for _ in range(config.gradient_steps):
                        agent.optimize(batch_size=config.batch_size)
                    updates += config.gradient_steps

                total_reward += reward
                state = next_state
                state_obj = next_state_obj
                steps += 1

                # Refresh inner bar
                if step_bar is not None:
                    step_bar.update(1)
                    step_bar.set_postfix(
                        r=f"{total_reward:+.2f}",
                        buf=agent.get_buffer_size(),
                        refresh=False,
                    )

            if step_bar is not None:
                step_bar.close()

            # ── episode bookkeeping ──────────────────────────────────────────
            episode_rewards.append(total_reward)
            recent_rewards.append(total_reward)
            avg10 = sum(recent_rewards) / len(recent_rewards)
            best_reward = max(best_reward, total_reward)

            # Update outer bar postfix
            if TQDM_AVAILABLE:
                ep_bar.set_postfix(
                    reward=f"{total_reward:+.2f}",
                    avg10=f"{avg10:+.2f}",
                    best=f"{best_reward:+.2f}",
                    buf=agent.get_buffer_size(),
                    upd=updates,
                    refresh=True,
                )

            # Per-episode log line — always printed so the user has a scrolling
            # record even when tqdm bars are active (tqdm.write routes above bars).
            ep_width = len(str(config.num_episodes))
            _log(
                f"[ep {episode + 1:>{ep_width}}/{config.num_episodes}]"
                f"  reward={total_reward:+8.2f}"
                f"  avg10={avg10:+8.2f}"
                f"  best={best_reward:+8.2f}"
                f"  steps={steps:>4}"
                f"  buf={agent.get_buffer_size():>6}"
                f"  upd={updates:>4}"
            )

            # Tensorboard
            if writer is not None:
                writer.add_scalar('rewards/episode', total_reward, episode)
                writer.add_scalar('rewards/avg10',   avg10,        episode)
                writer.add_scalar('train/updates',   updates,      episode)
                writer.add_scalar('train/updates_per_step', updates / max(steps, 1), episode)
                writer.add_scalar('buffer_size',     agent.get_buffer_size(), episode)
                writer.add_scalar('episode_steps',   steps,        episode)

            # Checkpoint
            if (episode + 1) % config.checkpoint_interval == 0:
                ckpt_path = os.path.join(
                    config.output_dir,
                    f'checkpoint_episode_{episode + 1}.pt',
                )
                agent.save_model(ckpt_path)
                _log(f"  checkpoint saved → {ckpt_path}")

    except KeyboardInterrupt:
        _log("\n⚠  Training interrupted by user")

    finally:
        final_model_path = os.path.join(config.output_dir, 'final_model.pt')
        agent.save_model(final_model_path)
        env.close()
        if writer is not None:
            writer.close()

    print("-" * 60)
    print("Training completed!")
    print(f"Final model → {final_model_path}")
    print(f"Results     → {config.output_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description='Train SAC agent with distance-based reward in ROS2 environment'
    )
    parser.add_argument('--num-episodes',        type=int,   default=1000,    help='Number of training episodes (default: 1000)')
    parser.add_argument('--max-steps',           type=int,   default=500,     help='Maximum steps per episode (default: 500)')
    parser.add_argument('--batch-size',          type=int,   default=256,     help='Batch size for optimization (default: 256)')
    parser.add_argument('--learning-rate',       type=float, default=3e-4,    help='Learning rate for optimizer (default: 3e-4)')
    parser.add_argument('--buffer-size',         type=int,   default=10000,   help='Replay buffer capacity (default: 10000)')
    parser.add_argument('--hidden-dim',          type=int,   default=256,     help='Hidden layer dimension for policy network (default: 256)')
    parser.add_argument('--output-dir',          type=str,   default='output/', help='Output directory for results (default: output/)')
    parser.add_argument('--checkpoint-interval', type=int,   default=50,      help='Save checkpoint every N episodes (default: 50)')
    parser.add_argument('--gradient-steps',      type=int,   default=10,      help='Gradient updates per environment step — increases GPU utilization (default: 10)')
    parser.add_argument('--no-tensorboard',      action='store_true',         help='Disable tensorboard logging')
    parser.add_argument('--load-model',          type=str,   default=None,    help='Path to model to load before training')
    parser.add_argument('--config',              type=str,   default=None,    help='Path to config JSON file to load')

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
            gradient_steps=args.gradient_steps,
        )

    train(config)

    # Optionally load model for evaluation
    if args.load_model:
        print(f"\nLoading model from {args.load_model}...")
        agent = SACAgent(state_dim=State.vector_dim(), action_dim=7)
        agent.load_model(args.load_model)
        print("Model loaded successfully")


if __name__ == "__main__":
    main()
