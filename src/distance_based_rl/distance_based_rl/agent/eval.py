"""
Deploy a trained SAC agent: drive the arm to a published target and hold.

Unlike training, this entry point:
  * does NOT home the arm on reset — it moves toward the target from the current pose;
  * does NOT randomize the target — it consumes the target published to
    /manipulator_target;
  * runs an open-ended control loop — once the target is reached the policy keeps the
    end-effector there, and when a new target is published the arm tracks it.

Usage (inside the container):
    ros2 run distance_based_rl eval_agent --load-model output/best_model.pt
Then publish a target:
    ros2 topic pub --once /manipulator_target geometry_msgs/msg/Point "{x: 0.4, y: 0.0, z: 0.5}"
"""

import argparse
import os
import time

from distance_based_rl.agent.sac_agent import SACAgent
from distance_based_rl.environment.arm_env import ManipulatorEnv


def evaluate(model_path: str, hidden_dim: int = 256, max_target_wait_sec: float = 60.0):
    """Load *model_path* and continuously drive the arm toward the published target."""
    print("=" * 60)
    print("Distance-Based RL — Deploy / Inference")
    print("=" * 60)

    import torch
    _cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    _cuda_disabled = _cuda_env is not None and _cuda_env.strip() == ''
    device = 'cpu' if (_cuda_disabled or not torch.cuda.is_available()) else 'cuda'

    # Inference env: keep the arm where it is, track an externally published target.
    env = ManipulatorEnv(external_target=True, home_on_reset=False)
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=hidden_dim,
        device=device,
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    agent.load_model(model_path)
    print(f"  loaded model : {model_path}")
    print(f"  device       : {device}")
    print("  Publish a target to /manipulator_target to move the arm. Ctrl+C to stop.")
    print("-" * 60)

    try:
        # reset() waits for state; with external_target it does not publish a target.
        _, info = env.reset()
        state_obj = info.get('state')

        # Wait until a target has actually been published before commanding motion.
        deadline = time.monotonic() + max_target_wait_sec
        while env.node.get_target_position() is None and time.monotonic() < deadline:
            print("  waiting for target on /manipulator_target ...")
            time.sleep(1.0)

        last_logged = 0.0
        while True:
            obs = state_obj if state_obj is not None else env.get_state()
            action = agent.select_action(obs, evaluate=True)
            _, reward, terminated, truncated, step_info = env.step(action)
            state_obj = step_info.get('state')

            # Open-ended hold: ignore terminated/truncated and keep tracking the target.
            now = time.monotonic()
            if now - last_logged >= 1.0:
                dist = step_info.get('distance')
                reached = step_info.get('success')
                if dist is not None:
                    print(f"  distance={dist:.3f} m  reached={bool(reached)}")
                last_logged = now

    except KeyboardInterrupt:
        print("\n⚠  Deployment stopped by user")
    finally:
        env.close()


def main():
    """Parse arguments and run deployment."""
    parser = argparse.ArgumentParser(
        description='Deploy a trained SAC agent to track a published target with the arm'
    )
    parser.add_argument('--load-model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--hidden-dim',  type=int, default=256,  help='Hidden dim — must match the trained model (default: 256)')
    parser.add_argument('--target-wait', type=float, default=60.0, help='Seconds to wait for the first target (default: 60)')
    args = parser.parse_args()

    evaluate(args.load_model, hidden_dim=args.hidden_dim, max_target_wait_sec=args.target_wait)


if __name__ == "__main__":
    main()
