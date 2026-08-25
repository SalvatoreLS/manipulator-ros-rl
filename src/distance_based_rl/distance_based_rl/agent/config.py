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
        #
        # This was 6, sized on the theory that 6 x ~8 ms of gradient work hides inside
        # env.step()'s ~50 ms of ROS2 waiting.  The per-call cost is right (measured
        # 7.87 ms mean; see scripts/bench_optimize.py) but the conclusion was not:
        # the updates run on a background thread, and between CUDA dispatches they hold
        # the GIL, starving the ROS executor that _wait_until_settled() polls.  The
        # settle detection then rides its timeout on every step.  Measured end to end:
        #
        #   gradient_steps=6 ->  1.94 step/s  (516 ms/step)
        #   gradient_steps=1 ->  9.76 step/s  (102 ms/step)
        #
        # Each extra update costs ~83 ms of wall clock in situ, ~10x its standalone
        # cost.  One update per env step is also the SAC default, and because the
        # environment now runs 5x faster the number of updates per second of wall clock
        # is unchanged — the same learning, against five times as much fresh data.
        # Raise this only after re-measuring; it is not a free knob.
        self.gradient_steps = kwargs.get('gradient_steps', 1)
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
