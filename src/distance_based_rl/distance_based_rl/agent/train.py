"""Train the SAC agent with the ROS2 environment."""

import argparse
import os
import warnings
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Optional

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

from distance_based_rl.agent.config import TrainingConfig
from distance_based_rl.agent.sac_agent import SACAgent
from distance_based_rl.environment.arm_env import ManipulatorEnv, StateActionReward
from distance_based_rl.environment.env_config import EnvConfig, _env_int


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


def _run_gradient_updates(agent: SACAgent, batch_size: int, gradient_steps: int) -> dict:
    """
    Run *gradient_steps* SAC optimize calls in a background thread.

    Returns averaged loss metrics over all gradient steps.
    The CUDA synchronize at the end ensures all GPU kernels dispatched by
    optimizer.step() are fully written to device memory before the future
    resolves.  Without this, the main thread could read partially-updated
    parameters via select_action() immediately after result() returns.
    """
    totals: dict = {}
    for _ in range(gradient_steps):
        metrics = agent.optimize(batch_size=batch_size)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
    # Flush all asynchronous CUDA work from this thread's stream so the main
    # thread sees fully-updated parameters when pending_opt.result() unblocks.
    if agent.device.type == 'cuda':
        import torch as _torch
        _torch.cuda.synchronize(device=agent.device)
    n = max(gradient_steps, 1)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config=None, resume_from=None):
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

    if device == 'cuda':
        # Allow cuDNN to auto-select the fastest kernels for fixed-size inputs.
        torch.backends.cudnn.benchmark = True

    # Torch defaults to one CPU thread per physical core (6 on this machine), but the
    # networks here are two 256-wide layers running on the GPU — there is no CPU work
    # worth parallelising.  Under scripts/run_experiments.sh several trainings run
    # concurrently, each alongside its own Gazebo, so the default would put ~18 torch
    # threads on 6 physical cores and steal time from the simulators, which ARE the
    # bottleneck.  One thread per trainer leaves the cores to physics.
    # Override with TORCH_NUM_THREADS if a CPU-only run ever needs more.
    torch.set_num_threads(max(1, _env_int('TORCH_NUM_THREADS', 1)))

    # Seed every source of randomness in the process. The env's own generator is seeded
    # through the constructor below.
    if config.seed is not None:
        import numpy as _np
        torch.manual_seed(config.seed)
        _np.random.seed(config.seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(config.seed)

    # Build environment and agent
    print("Initializing environment and agent...")
    env_config = EnvConfig(max_episode_steps=config.max_steps_per_episode)
    env_config.save(os.path.join(config.output_dir, 'env_config.json'))
    env = ManipulatorEnv(config=env_config, seed=config.seed)
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=config.hidden_dim,
        lr=config.learning_rate,
        buffer_size=config.buffer_size,
        device=device,
    )
    if resume_from:
        if os.path.exists(resume_from):
            agent.load_model(resume_from)
            print(f"  resumed from : {resume_from}")
        else:
            print(f"  --load-model '{resume_from}' not found — starting fresh")

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
    recent_successes: deque[bool] = deque(maxlen=10)
    recent_distances: deque[float] = deque(maxlen=10)
    # One rolling window per reporting tolerance, so success is quoted at 10/5/2 cm
    # rather than only at the (generous) threshold that ends an episode.
    recent_success_at: dict[str, deque] = {}
    best_reward = float('-inf')
    best_model_path = os.path.join(config.output_dir, 'best_model.pt')
    global_step = 0  # counts env steps across all episodes (drives warm-up)

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

    # Single background worker that runs gradient updates concurrently with env.step().
    # env.step() sleeps for ~50 ms (ROS2 controller settling time), which releases the
    # Python GIL and lets the worker thread run PyTorch / CUDA kernels freely.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sac_opt')
    pending_opt: Optional[Future] = None

    try:
        for episode in ep_bar:
            # Drain any update that was still running at the end of the previous episode
            if pending_opt is not None:
                pending_opt.result()
                pending_opt = None

            # Seed only the first reset: reseeding every episode would replay the same
            # target over and over instead of a reproducible *sequence* of targets.
            state, info = env.reset(seed=config.seed if episode == 0 else None)
            state_obj = info.get('state')
            done = False
            total_reward = 0.0
            steps = 0
            updates = 0
            episode_success = False
            final_distance = float('nan')
            steps_to_goal = None
            success_at: dict = {}
            last_metrics: dict = {}

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
                # 1. Select action. During warm-up use uniform random actions to seed the
                #    replay buffer; afterwards read the policy network (stable: pending_opt
                #    drained after env.step).
                if global_step < config.warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state_obj if state_obj is not None else state)

                # 3. Submit gradient updates to a background thread so they run
                #    concurrently with env.step()'s ~50 ms ROS2/sleep overhead.
                #    The GIL is released during sleep() and CUDA kernels, so the
                #    worker thread gets real CPU+GPU time while the main thread waits.
                if len(agent.replay_buffer) >= config.batch_size:
                    pending_opt = executor.submit(
                        _run_gradient_updates,
                        agent,
                        config.batch_size,
                        config.gradient_steps,
                    )
                    updates += config.gradient_steps
                else:
                    pending_opt = None

                # 4. Env step — gradient updates overlap this blocking call.  [ROS]
                next_state, reward, terminated, truncated, step_info = env.step(action)
                next_state_obj = step_info.get('state')
                done = terminated or truncated
                if step_info.get('distance') is not None:
                    final_distance = float(step_info['distance'])
                    success_at = step_info.get('success_at', {})
                if terminated:
                    episode_success = True
                    if steps_to_goal is None:
                        steps_to_goal = steps + 1
                if pending_opt is not None:
                    last_metrics = pending_opt.result()
                    pending_opt = None

                # 5. Store transition (push() is lock-protected inside ReplayBuffer).
                # Use `terminated` (not `done`) so truncated episodes still bootstrap:
                # done=True cuts the Bellman backup; timeout is not a true terminal state.
                if state_obj is not None and next_state_obj is not None:
                    transition = StateActionReward(
                        state=state_obj,
                        action=action,
                        reward=reward,
                        next_state=next_state_obj,
                        done=terminated,
                    )
                    agent.replay_buffer.push(transition)
                else:
                    agent.replay_buffer.push(state, action, reward, next_state, terminated)

                total_reward += reward
                state     = next_state
                state_obj = next_state_obj
                steps += 1
                global_step += 1

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
            recent_successes.append(episode_success)
            avg10 = sum(recent_rewards) / len(recent_rewards)
            success_rate = sum(recent_successes) / len(recent_successes)

            if final_distance == final_distance:  # not NaN
                recent_distances.append(final_distance)
            for thresh, reached in success_at.items():
                recent_success_at.setdefault(thresh, deque(maxlen=10)).append(bool(reached))
            avg_distance = (
                sum(recent_distances) / len(recent_distances) if recent_distances else float('nan')
            )

            if total_reward > best_reward:
                best_reward = total_reward
                agent.save_model(best_model_path)

            # Update outer bar postfix
            if TQDM_AVAILABLE:
                ep_bar.set_postfix(
                    reward=f"{total_reward:+.2f}",
                    avg10=f"{avg10:+.2f}",
                    best=f"{best_reward:+.2f}",
                    sr=f"{success_rate:.0%}",
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
                f"  sr10={success_rate:.0%}"
                f"  dist={final_distance:.3f}m"
                f"  steps={steps:>4}"
                f"  buf={agent.get_buffer_size():>6}"
                f"  upd={updates:>4}"
            )

            # Tensorboard
            if writer is not None:
                writer.add_scalar('rewards/episode',        total_reward,                    episode)
                writer.add_scalar('rewards/avg10',          avg10,                           episode)
                writer.add_scalar('rewards/best',           best_reward,                     episode)
                writer.add_scalar('train/success_rate_10',  success_rate,                    episode)
                writer.add_scalar('train/updates',          updates,                         episode)
                writer.add_scalar('train/updates_per_step', updates / max(steps, 1),         episode)
                writer.add_scalar('buffer_size',            agent.get_buffer_size(),         episode)
                writer.add_scalar('episode_steps',          steps,                           episode)
                if final_distance == final_distance:  # not NaN
                    writer.add_scalar('train/final_distance',     final_distance, episode)
                    writer.add_scalar('train/final_distance_10',  avg_distance,   episode)
                if steps_to_goal is not None:
                    writer.add_scalar('train/steps_to_goal', steps_to_goal, episode)
                for thresh, window in recent_success_at.items():
                    writer.add_scalar(
                        f'train/success_rate_10_at_{thresh}m',
                        sum(window) / len(window),
                        episode,
                    )
                if last_metrics:
                    writer.add_scalar('losses/critic', last_metrics.get('critic_loss', 0.0), episode)
                    writer.add_scalar('losses/policy', last_metrics.get('policy_loss', 0.0), episode)
                    writer.add_scalar('losses/alpha',  last_metrics.get('alpha_loss',  0.0), episode)
                    writer.add_scalar('train/alpha',   last_metrics.get('alpha',       0.0), episode)

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
        # Drain any pending gradient update before saving the model
        if pending_opt is not None:
            try:
                pending_opt.result()
            except Exception as exc:
                _log(f"  warning: gradient update failed during shutdown: {exc}")
        executor.shutdown(wait=False)

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
    parser.add_argument('--hidden-dim',          type=int,   default=256,
                        help='Hidden layer dimension for policy network (default: 256)')
    parser.add_argument('--output-dir',          type=str,   default='output/',
                        help='Output directory for results (default: output/)')
    parser.add_argument('--checkpoint-interval', type=int,   default=50,
                        help='Save checkpoint every N episodes (default: 50)')
    parser.add_argument('--gradient-steps',      type=int,   default=1,
                        help='Gradient updates per environment step. Raising this costs far more '
                             'wall clock than the GPU work implies (the updates hold the GIL and '
                             'starve the ROS executor): 6 measured 5x slower end to end than 1. '
                             'See agent/config.py and scripts/bench_optimize.py (default: 1)')
    parser.add_argument('--warmup-steps',        type=int,   default=1000,
                        help='Initial env steps using random actions before the policy (default: 1000)')
    parser.add_argument('--seed',                type=int,   default=None,
                        help='Master seed for torch, numpy, the action space and target sampling (default: unseeded)')
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
            warmup_steps=args.warmup_steps,
            seed=args.seed,
        )

    train(config, resume_from=args.load_model)


if __name__ == "__main__":
    main()
