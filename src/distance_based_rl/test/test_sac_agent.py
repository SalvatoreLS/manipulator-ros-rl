"""Unit tests for SAC agent components."""

import pytest
import sys
import os
import numpy as np
import torch
import tempfile
import json

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from distance_based_rl.agent.sac_agent import FCGP, ReplayBuffer, SACAgent
from distance_based_rl.agent.config import TrainingConfig


class TestFCGP:
    """Test Fully Connected Gaussian Policy network."""

    def test_fcgp_initialization(self):
        """Test that FCGP network initializes correctly."""
        state_dim = 6
        action_dim = 7
        network = FCGP(state_dim, action_dim, hidden_dim=256)
        
        assert network.fc1.in_features == state_dim
        assert network.fc1.out_features == 256
        assert network.mean_layer.out_features == action_dim
        assert network.log_std_layer.out_features == action_dim

    def test_fcgp_forward_pass(self):
        """Test that FCGP forward pass produces correct output shapes."""
        state_dim = 6
        action_dim = 7
        batch_size = 4
        
        network = FCGP(state_dim, action_dim, hidden_dim=256)
        state = torch.randn(batch_size, state_dim)
        
        mean, log_std = network(state)
        
        assert mean.shape == (batch_size, action_dim)
        assert log_std.shape == (batch_size, action_dim)
        assert torch.all(log_std >= -20) and torch.all(log_std <= 2)  # Clamp check

    def test_fcgp_single_state(self):
        """Test FCGP with single state vector."""
        state_dim = 6
        action_dim = 7
        
        network = FCGP(state_dim, action_dim)
        state = torch.randn(1, state_dim)
        
        mean, log_std = network(state)
        
        assert mean.shape == (1, action_dim)
        assert log_std.shape == (1, action_dim)


class TestReplayBuffer:
    """Test replay buffer for storing transitions."""

    def test_replay_buffer_initialization(self):
        """Test replay buffer initialization."""
        capacity = 100
        buffer = ReplayBuffer(capacity)
        
        assert buffer.capacity == capacity
        assert len(buffer) == 0

    def test_replay_buffer_push(self):
        """Test pushing transitions to buffer."""
        buffer = ReplayBuffer(10)
        state = np.array([1, 2, 3])
        action = np.array([0.5, 0.6])
        reward = 1.0
        next_state = np.array([1.1, 2.1, 3.1])
        done = False
        
        buffer.push(state, action, reward, next_state, done)
        assert len(buffer) == 1

    def test_replay_buffer_circular_overwrite(self):
        """Test that buffer overwrites old transitions when full."""
        buffer = ReplayBuffer(3)
        
        for i in range(5):
            buffer.push(
                np.array([i]), 
                np.array([i]), 
                float(i), 
                np.array([i]), 
                False
            )
        
        # Buffer should only hold last 3 items
        assert len(buffer) == 3

    def test_replay_buffer_sample(self):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(100)
        
        for i in range(10):
            buffer.push(
                np.array([i, i+1, i+2]),
                np.array([i*0.1]),
                float(i),
                np.array([i, i+1, i+2]),
                i % 2 == 0
            )
        
        batch_size = 4
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        assert states.shape[0] == batch_size
        assert actions.shape[0] == batch_size
        assert rewards.shape[0] == batch_size
        assert next_states.shape[0] == batch_size
        assert dones.shape[0] == batch_size

    def test_replay_buffer_sample_raises_when_empty(self):
        """Test that sampling from empty buffer raises error."""
        buffer = ReplayBuffer(10)
        
        with pytest.raises(ValueError):
            buffer.sample(4)


class TestSACAgent:
    """Test SAC agent."""

    def test_sac_agent_initialization(self):
        """Test SAC agent initialization."""
        state_dim = 6
        action_dim = 7
        
        agent = SACAgent(state_dim, action_dim)
        
        assert agent.policy is not None
        assert agent.optimizer is not None
        assert agent.replay_buffer is not None

    def test_sac_agent_select_action(self):
        """Test action selection from SAC agent."""
        state_dim = 6
        action_dim = 7
        
        agent = SACAgent(state_dim, action_dim)
        state = np.random.randn(state_dim)
        
        action = agent.select_action(state)
        
        assert action.shape == (action_dim,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)  # Tanh squashing

    def test_sac_agent_action_range(self):
        """Test that selected actions are in [-1, 1] range."""
        state_dim = 6
        action_dim = 7
        
        agent = SACAgent(state_dim, action_dim)
        
        for _ in range(10):
            state = np.random.randn(state_dim)
            action = agent.select_action(state)
            
            assert np.all(action >= -1.0), "Action has values < -1"
            assert np.all(action <= 1.0), "Action has values > 1"

    def test_sac_agent_store_transition(self):
        """Test storing transitions in agent's replay buffer."""
        state_dim = 6
        action_dim = 7
        
        agent = SACAgent(state_dim, action_dim)
        
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = 1.0
        next_state = np.random.randn(state_dim)
        done = False
        
        agent.replay_buffer.push(state, action, reward, next_state, done)
        
        assert len(agent.replay_buffer) == 1

    def test_sac_agent_optimize(self):
        """Test that optimize step runs without error."""
        state_dim = 6
        action_dim = 7
        agent = SACAgent(state_dim, action_dim, buffer_size=100)
        
        # Fill buffer with random transitions
        for _ in range(50):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            reward = np.random.randn(1)[0]
            next_state = np.random.randn(state_dim)
            done = np.random.rand() > 0.5
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
        
        # Should not raise error
        agent.optimize(batch_size=16)
        assert True  # If no exception, test passes

    def test_sac_agent_save_load_model(self):
        """Test saving and loading agent model."""
        state_dim = 6
        action_dim = 7
        
        agent = SACAgent(state_dim, action_dim, hidden_dim=256, lr=3e-4)
        
        # Fill buffer with some data before saving
        for _ in range(10):
            agent.replay_buffer.push(
                np.random.randn(state_dim),
                np.random.randn(action_dim),
                1.0,
                np.random.randn(state_dim),
                False
            )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pt')
            
            # Save model
            agent.save_model(model_path)
            assert os.path.exists(model_path), "Model file not created"
            
            # Create new agent and load model
            agent2 = SACAgent(state_dim, action_dim)
            agent2.load_model(model_path)
            
            # Verify loaded weights match
            for p1, p2 in zip(agent.policy.parameters(), agent2.policy.parameters()):
                assert torch.allclose(p1, p2), "Loaded model weights don't match"

    def test_sac_agent_get_buffer_size(self):
        """Test buffer size tracking."""
        agent = SACAgent(6, 7, buffer_size=100)
        
        assert agent.get_buffer_size() == 0
        
        # Add some transitions
        for i in range(5):
            agent.replay_buffer.push(
                np.random.randn(6),
                np.random.randn(7),
                1.0,
                np.random.randn(6),
                False
            )
        
        assert agent.get_buffer_size() == 5


class TestTrainingConfig:
    """Test training configuration."""

    def test_config_initialization(self):
        """Test configuration initialization with defaults."""
        config = TrainingConfig()
        
        assert config.num_episodes == 1000
        assert config.max_steps_per_episode == 500
        assert config.batch_size == 256
        assert config.learning_rate == 3e-4
        assert config.buffer_size == 10000

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = TrainingConfig(
            num_episodes=500,
            batch_size=128,
            learning_rate=1e-3,
            hidden_dim=512
        )
        
        assert config.num_episodes == 500
        assert config.batch_size == 128
        assert config.learning_rate == 1e-3
        assert config.hidden_dim == 512

    def test_config_save_load(self):
        """Test saving and loading configuration."""
        config1 = TrainingConfig(
            num_episodes=100,
            batch_size=64,
            learning_rate=5e-4,
            hidden_dim=128
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.json')
            
            # Save config
            config1.save(config_path)
            assert os.path.exists(config_path), "Config file not created"
            
            # Load config
            config2 = TrainingConfig.load(config_path)
            
            # Verify values match
            assert config2.num_episodes == 100
            assert config2.batch_size == 64
            assert config2.learning_rate == 5e-4
            assert config2.hidden_dim == 128

    def test_config_string_representation(self):
        """Test configuration string representation."""
        config = TrainingConfig(num_episodes=50)
        config_str = str(config)
        
        assert 'TrainingConfig' in config_str
        assert 'num_episodes=50' in config_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
