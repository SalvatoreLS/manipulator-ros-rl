#!/usr/bin/env python3
"""Stable-Baselines3 SAC baseline on the same ROS2/Gazebo environment.

Purpose is *not* to get a better policy — it is a correctness check on the from-scratch
SAC in ``distance_based_rl/agent/sac_agent.py``. Both agents see the identical
environment, the identical hyperparameters and the identical step budget, so a matching
learning curve is evidence the hand-written implementation is right, and a diverging one
is a bug worth finding.

Run it exactly like the training entry point, against a live simulation:

    ros2 launch franka_gazebo_bringup gazebo_franka_arm_example_controller.launch.py \
        controller:=forward_position_controller rviz:=false gz_args:="-s -r empty.sdf"
    python3 scripts/sb3_baseline.py --total-steps 50000 --seed 0 \
        --output-dir output/runs/sb3/seed0
"""

import argparse
import os

import gymnasium as gym

# Importing the package registers "ManipulatorReach-v0".
import distance_based_rl.environment  # noqa: F401
from distance_based_rl.environment import ENV_ID
from distance_based_rl.environment.env_config import EnvConfig


def main():
    """Parse arguments and train an SB3 SAC agent on the manipulator env."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total-steps',   type=int,   default=50_000, help='Total environment steps (default: 50000)')
    parser.add_argument('--max-steps',     type=int,   default=500,    help='Max steps per episode (default: 500)')
    parser.add_argument('--batch-size',    type=int,   default=256,    help='Batch size (default: 256)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,   help='Learning rate (default: 3e-4)')
    parser.add_argument('--buffer-size',   type=int,   default=50_000, help='Replay buffer capacity (default: 50000)')
    parser.add_argument('--hidden-dim',    type=int,   default=256,    help='Hidden width of both nets (default: 256)')
    parser.add_argument('--warmup-steps',  type=int,   default=1000,   help='Random-action warm-up steps (default: 1000)')
    parser.add_argument('--gradient-steps', type=int,  default=6,      help='Gradient updates per env step (default: 6)')
    parser.add_argument('--seed',          type=int,   default=0,      help='Seed (default: 0)')
    parser.add_argument('--output-dir',    type=str,   default='output/runs/sb3', help='Where to write logs and the model')
    args = parser.parse_args()

    # Imported late: pulling in torch/SB3 before argparse makes --help slow.
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')

    env_config = EnvConfig(max_episode_steps=args.max_steps)
    env_config.save(os.path.join(args.output_dir, 'env_config.json'))

    # disable_env_checker: the checker calls reset()/step() an extra time, which would
    # command the real arm before training starts.
    env = gym.make(
        ENV_ID,
        config=env_config,
        seed=args.seed,
        disable_env_checker=True,
    )
    env = Monitor(env, filename=os.path.join(args.output_dir, 'monitor.csv'))

    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.warmup_steps,
        gradient_steps=args.gradient_steps,
        train_freq=1,
        ent_coef='auto',          # matches the automatic temperature tuning in SACAgent
        policy_kwargs=dict(net_arch=[args.hidden_dim, args.hidden_dim]),
        tensorboard_log=log_dir,
        seed=args.seed,
        verbose=1,
    )

    print('=' * 60)
    print(f'SB3 SAC baseline on {ENV_ID}')
    print(f'  total steps : {args.total_steps}')
    print(f'  seed        : {args.seed}')
    print(f'  logs        : {log_dir}')
    print('=' * 60)

    try:
        model.learn(total_timesteps=args.total_steps, log_interval=1)
    except KeyboardInterrupt:
        print('\n⚠  Baseline interrupted by user')
    finally:
        model_path = os.path.join(args.output_dir, 'sb3_sac.zip')
        model.save(model_path)
        env.close()
        print(f'Model → {model_path}')


if __name__ == '__main__':
    main()
