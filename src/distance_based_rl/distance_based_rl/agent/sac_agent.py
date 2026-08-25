"""Define the Soft Actor-Critic (SAC) agent for distance-based RL."""

import copy
import threading
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os


class GaussianPolicy(nn.Module):
    """
    Class defining the Gaussian policy network for the SAC agent.

    It outputs both the mean and log standard deviation of the action distribution.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(GaussianPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 2)
        return mean, log_std

    def sample(self, state):
        """Sample an action from the policy given a state."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class QNetwork(nn.Module):
    """
    Class defining the Q-network for the SAC agent.

    It takes both state and action as input and outputs a single Q-value.
    It is the critic network used to estimate the value of state-action pairs,
    and is trained to minimize the Bellman error.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """
    Circular replay buffer backed by pre-allocated contiguous NumPy arrays.

    Compared to a Python-list buffer this gives:
      - O(1) insertion (no Python object allocation per transition)
      - Fast batch sampling via np.random.randint + fancy indexing instead of
        random.sample on a list of tuples followed by zip(*batch) and repeated
        np.asarray calls.
      - Thread-safe push / sample via a threading.Lock so that gradient updates
        can run concurrently with the environment step.

    Arrays are allocated lazily on the first push() call so the constructor does
    not need the state / action dimensions.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._lock = threading.Lock()
        self._arrays = None   # allocated on first push
        self._ptr  = 0
        self._size = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_vector(state) -> np.ndarray:
        if hasattr(state, 'as_vector') and callable(state.as_vector):
            return np.asarray(state.as_vector(), dtype=np.float32).reshape(-1)
        return np.asarray(state, dtype=np.float32).reshape(-1)

    def _alloc(self, state_dim: int, action_dim: int) -> None:
        """Pre-allocate contiguous float32 arrays (called once, under lock)."""
        c = self.capacity
        self._arrays = dict(
            states      = np.zeros((c, state_dim),  dtype=np.float32),
            actions     = np.zeros((c, action_dim), dtype=np.float32),
            rewards     = np.zeros(c,               dtype=np.float32),
            next_states = np.zeros((c, state_dim),  dtype=np.float32),
            dones       = np.zeros(c,               dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, state, action=None, reward=None, next_state=None, done=None):
        # Unwrap StateActionReward / any object with as_replay_tuple()
        if action is None and hasattr(state, 'as_replay_tuple') and callable(state.as_replay_tuple):
            state, action, reward, next_state, done = state.as_replay_tuple()

        s  = self._to_vector(state)
        ns = self._to_vector(next_state)
        a  = np.asarray(action, dtype=np.float32).reshape(-1)

        with self._lock:
            if self._arrays is None:
                self._alloc(s.shape[0], a.shape[0])
            arr = self._arrays
            p   = self._ptr
            arr['states'][p]      = s
            arr['actions'][p]     = a
            arr['rewards'][p]     = float(reward)
            arr['next_states'][p] = ns
            arr['dones'][p]       = float(done)
            self._ptr  = (p + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Return a random batch as five NumPy arrays.

        Fancy indexing always produces a copy, so the returned arrays are
        independent of the buffer and safe to use outside the lock.
        """
        with self._lock:

            size = self._size
            if size < batch_size:
                raise ValueError(
                    f"Not enough samples in replay buffer: "
                    f"requested {batch_size}, available {size}"
                )
            idxs = np.random.randint(0, size, size=batch_size)
            arr  = self._arrays
            return (
                arr['states'][idxs],
                arr['actions'][idxs],
                arr['rewards'][idxs],
                arr['next_states'][idxs],
                arr['dones'][idxs],
            )

    def __len__(self) -> int:
        return self._size


class SACAgent:
    """
    Class defining the Soft Actor-Critic (SAC) agent for distance-based reinforcement learning.

    It includes the policy network, two Q-networks, target networks, and optimizers for each component.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        buffer_size=10000,
        device=None,
    ):
        self.state_dim  = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = gamma
        self.tau   = tau
        self.alpha = alpha
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Resolve device: honour explicit argument, then check CUDA availability.
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=lr
        )

        self.target_entropy  = -action_dim
        # Initialise log_alpha at log(alpha) rather than 0.  Starting at 0 means alpha=1.0
        # on the first optimize() call — and since the shaped per-step reward is O(0.1-1),
        # an entropy coefficient of 1.0 lets the entropy term dominate the Q term for the
        # first few thousand updates, keeping the policy near-random.  Auto-tuning still
        # moves alpha from here; this only fixes where it starts.
        self.log_alpha       = torch.full(
            (1,), float(np.log(alpha)), requires_grad=True, device=self.device
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.optimizer       = self.policy_optimizer

    def _state_to_vector(self, state):
        if hasattr(state, 'as_vector') and callable(state.as_vector):
            state_vec = np.asarray(state.as_vector(), dtype=np.float32)
        else:
            state_vec = np.asarray(state, dtype=np.float32)
        return state_vec.reshape(-1)

    def select_action(self, state, evaluate=False):
        # torch.from_numpy: zero-copy from numpy; unsqueeze for batch dim
        state_np = self._state_to_vector(state)
        state_t  = torch.from_numpy(state_np).unsqueeze(0).to(self.device, non_blocking=True)
        if evaluate:
            mean, _ = self.policy(state_t)
            action  = torch.tanh(mean)
        else:
            action, _ = self.policy.sample(state_t)
        return action.detach().cpu().numpy()[0].astype(np.float32)

    def optimize(self, batch_size=256):
        s_np, a_np, r_np, ns_np, d_np = self.replay_buffer.sample(batch_size)

        # torch.from_numpy: zero-copy on CPU; .to() does async H2D copy on CUDA.
        states      = torch.from_numpy(s_np).to(self.device,  non_blocking=True)
        actions     = torch.from_numpy(a_np).to(self.device,  non_blocking=True)
        rewards     = torch.from_numpy(r_np).unsqueeze(1).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(ns_np).to(self.device, non_blocking=True)
        dones       = torch.from_numpy(d_np).unsqueeze(1).to(self.device, non_blocking=True)

        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * min_q_next

        # Critic update
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # set_to_none=True: skips the zero-fill allocation, freeing memory sooner
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), max_norm=1.0
        )
        self.critic_optimizer.step()

        # Policy update
        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_probs - min_q_new).mean()

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # Temperature update
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = float(self.log_alpha.exp().detach().item())

        self._soft_update_targets()

        return {
            'critic_loss': critic_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss':  alpha_loss.item(),
            'alpha':       self.alpha,
        }

    def _soft_update_targets(self):
        """Soft-update target networks using in-place lerp_ (faster than explicit formula)."""
        for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            tp.data.lerp_(p.data, self.tau)
        for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            tp.data.lerp_(p.data, self.tau)

    def save_model(self, model_path):
        """Save model checkpoints and optimizer states."""
        dirname = os.path.dirname(model_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        tmp_path = model_path + '.tmp'
        torch.save(
            {
                'policy':           self.policy.state_dict(),
                'critic1':          self.critic1.state_dict(),
                'critic2':          self.critic2.state_dict(),
                'critic1_target':   self.critic1_target.state_dict(),
                'critic2_target':   self.critic2_target.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'alpha_optimizer':  self.alpha_optimizer.state_dict(),
                'log_alpha':        self.log_alpha.detach().cpu(),
                'alpha':            self.alpha,
            },
            tmp_path,
        )
        os.replace(tmp_path, model_path)

    def load_model(self, model_path):
        """Load model checkpoints and optimizer states when available."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])

        if 'policy_optimizer' in checkpoint:
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        if 'critic_optimizer' in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if 'alpha_optimizer' in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        if 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha'].to(self.device).clone().detach().requires_grad_(True)
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha],
                lr=self.alpha_optimizer.param_groups[0]['lr'],
            )
        self.alpha = float(checkpoint.get('alpha', self.log_alpha.exp().detach().item()))

    def get_buffer_size(self):
        """Return the number of stored transitions in replay buffer."""
        return len(self.replay_buffer)
