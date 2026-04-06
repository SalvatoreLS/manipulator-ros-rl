"""Define the Soft Actor-Critic (SAC) agent for distance-based RL."""

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import json

# TODO: check correctness of these classes

class FCGP(nn.Module):
    """Fully Connected Gaussian Policy Network."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(FCGP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 2)  # Clamp log_std for numerical stability
        return mean, log_std
    

class ReplayBuffer:
    """Simple replay buffer for storing transitions."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, buffer_size=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.policy = FCGP(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.buffer_size = buffer_size

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.policy(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # Reparameterization trick
        action = torch.tanh(z)  # Squash action to [-1, 1]
        return action.detach().cpu().numpy()[0]

    def optimize(self, batch_size=256):
        # Sample a batch of transitions from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute target Q-values and update the policy network
        q_values = self.policy(states)[0]  # Get mean action from policy
        target_q_values = rewards + (1 - dones) * 0.99 * self.policy(next_states)[0].max(1, keepdim=True)[0]  # Compute target Q-values
        loss = F.mse_loss(q_values, target_q_values.detach())  # Compute MSE loss between current Q-values and target Q-values
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def save_model(self, path):
        """Save agent model and hyperparameters to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'lr': self.lr,
            'buffer_size': self.buffer_size,
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load agent model and hyperparameters from disk."""
        checkpoint = torch.load(path)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {path}")
        print(f"Loaded model: state_dim={checkpoint['state_dim']}, "
              f"action_dim={checkpoint['action_dim']}, "
              f"hidden_dim={checkpoint['hidden_dim']}")

    def get_buffer_size(self):
        """Return current size of replay buffer."""
        return len(self.replay_buffer)