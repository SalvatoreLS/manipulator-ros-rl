"""
ROS2/Gazebo Gymnasium environment for distance-based manipulator control.

Registering the env under an id lets any standard tool construct it — in particular the
Stable-Baselines3 baseline used to cross-check the from-scratch SAC implementation:

    import gymnasium as gym
    import distance_based_rl.environment  # noqa: F401  (registers the id)
    env = gym.make("ManipulatorReach-v0")
"""

from gymnasium.envs.registration import register, registry

from .arm_env import ManipulatorEnv, State, StateActionReward
from .data_handler import DataHandler
from .env_config import EnvConfig

ENV_ID = "ManipulatorReach-v0"

# Guard against re-registration: this module is imported from several entry points, and
# gymnasium raises if an id is registered twice.
if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="distance_based_rl.environment.arm_env:ManipulatorEnv",
        # Episode truncation is handled inside the env (it needs the step count to build
        # the info dict), so no TimeLimit wrapper is applied here.
        max_episode_steps=None,
        order_enforce=False,
    )

__all__ = [
    "ENV_ID",
    "DataHandler",
    "EnvConfig",
    "ManipulatorEnv",
    "State",
    "StateActionReward",
]
