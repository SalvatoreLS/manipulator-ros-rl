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
        # Each optimize() call costs ~7-8 ms of Python dispatch overhead on top of
        # actual GPU compute (~1 ms).  The env.step() blocks for ~50 ms (ROS2 sleep),
        # which is the async window available for gradient work.  Keeping
        # gradient_steps * ~8 ms ≤ 50 ms ensures updates finish before the next
        # action selection — eliminating the per-step wait entirely.
        # On this hardware: 6 × 8 ms = 48 ms < 50 ms (fits with a small margin).
        # Increase if env.step() is slower (e.g. --state-wait-timeout > 5 s).
        self.gradient_steps = kwargs.get('gradient_steps', 6)
        # Number of initial environment steps that use uniformly random actions before the
        # policy takes over. Seeds the replay buffer with diverse transitions and avoids
        # early policy collapse — standard SAC warm-up.
        self.warmup_steps = kwargs.get('warmup_steps', 1000)
        # Master seed for the run: seeds torch, numpy, the action space and the
        # environment's target sampler. None means "not reproducible".
        self.seed = kwargs.get('seed', None)

    def save(self, path):
        """Save configuration to JSON file."""
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        config_dict = {
            'num_episodes':          self.num_episodes,
            'max_steps_per_episode': self.max_steps_per_episode,
            'batch_size':            self.batch_size,
            'learning_rate':         self.learning_rate,
            'buffer_size':           self.buffer_size,
            'hidden_dim':            self.hidden_dim,
            'output_dir':            self.output_dir,
            'checkpoint_interval':   self.checkpoint_interval,
            'use_tensorboard':       self.use_tensorboard,
            'gradient_steps':        self.gradient_steps,
            'warmup_steps':          self.warmup_steps,
            'seed':                  self.seed,
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
