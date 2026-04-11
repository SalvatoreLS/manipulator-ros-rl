"""Define the Soft Actor-Critic (SAC) agent for distance-based RL."""

import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
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
    """Simple replay buffer for storing transitions."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    @staticmethod
    def _to_vector(state):
        if hasattr(state, 'as_vector') and callable(state.as_vector):
            return np.asarray(state.as_vector(), dtype=np.float32).reshape(-1)
        return np.asarray(state, dtype=np.float32).reshape(-1)

    def push(self, state, action=None, reward=None, next_state=None, done=None):
        if action is None and hasattr(state, 'as_replay_tuple') and callable(state.as_replay_tuple):
            state, action, reward, next_state, done = state.as_replay_tuple()

        state_vec = self._to_vector(state)
        next_state_vec = self._to_vector(next_state)
        action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
        reward_val = float(reward)
        done_val = bool(done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state_vec, action_vec, reward_val, next_state_vec, done_val)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough samples in replay buffer: requested {batch_size}, available {len(self.buffer)}"
            )
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

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
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = gamma
        self.tau = tau
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

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.optimizer = self.policy_optimizer

    def _state_to_vector(self, state):
        if hasattr(state, 'as_vector') and callable(state.as_vector):
            state_vec = np.asarray(state.as_vector(), dtype=np.float32)
        else:
            state_vec = np.asarray(state, dtype=np.float32)
        return state_vec.reshape(-1)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(self._state_to_vector(state)).unsqueeze(0).to(self.device)
        if evaluate:
            mean, _ = self.policy(state)
            action = torch.tanh(mean)
        else:
            action, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0].astype(np.float32)

    def optimize(self, batch_size=256):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

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
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Policy update
        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - min_q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Temperature update
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = float(self.log_alpha.exp().detach().item())

        self._soft_update_targets()

    def _soft_update_targets(self):
        """Helper function to perform soft updates of the target networks."""
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, model_path):
        """Save model checkpoints and optimizer states."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(
            {
                'policy': self.policy.state_dict(),
                'critic1': self.critic1.state_dict(),
                'critic2': self.critic2.state_dict(),
                'critic1_target': self.critic1_target.state_dict(),
                'critic2_target': self.critic2_target.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'alpha_optimizer': self.alpha_optimizer.state_dict(),
                'log_alpha': self.log_alpha.detach().cpu(),
                'alpha': self.alpha,
            },
            model_path,
        )

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
            self.log_alpha = checkpoint['log_alpha'].clone().detach().requires_grad_(True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_optimizer.param_groups[0]['lr'])
        self.alpha = float(checkpoint.get('alpha', self.log_alpha.exp().detach().item()))

    def get_buffer_size(self):
        """Return the number of stored transitions in replay buffer."""
        return len(self.replay_buffer)


# Backward-compatible alias expected by tests.
FCGP = GaussianPolicy