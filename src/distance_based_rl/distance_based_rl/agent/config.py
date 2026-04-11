"""Training configuration management."""

import json
import os


class TrainingConfig:
    """Configuration for training."""

    def __init__(self, **kwargs):
        """Initialize configuration with defaults and overrides."""
        self.num_episodes = kwargs.get('num_episodes', 1000)
        self.max_steps_per_episode = kwargs.get('max_steps_per_episode', 500)
        self.batch_size = kwargs.get('batch_size', 256)
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.buffer_size = kwargs.get('buffer_size', 10000)
        self.hidden_dim = kwargs.get('hidden_dim', 256)
        self.output_dir = kwargs.get('output_dir', 'output/')
        self.checkpoint_interval = kwargs.get('checkpoint_interval', 50)
        self.use_tensorboard = kwargs.get('use_tensorboard', True)
        # Number of gradient updates to perform per environment step.
        # The ROS2/Gazebo step blocks for ~50-100 ms while GPU updates take ~1 ms,
        # so a UTD ratio > 1 is needed to saturate the GPU.
        self.gradient_steps = kwargs.get('gradient_steps', 10)

    def save(self, path):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        config_dict = {
            'num_episodes': self.num_episodes,
            'max_steps_per_episode': self.max_steps_per_episode,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'buffer_size': self.buffer_size,
            'hidden_dim': self.hidden_dim,
            'checkpoint_interval': self.checkpoint_interval,
            'gradient_steps': self.gradient_steps,
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path):
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def __repr__(self):
        """Represent the configuration as a string."""
        items = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"TrainingConfig({', '.join(items)})"
