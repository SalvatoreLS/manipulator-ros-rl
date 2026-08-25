"""
Unit tests for the ROS2 Gymnasium environment.

Everything here runs without a live ROS2 graph: ``DataHandler`` is built with its
``__init__`` bypassed and its publishers mocked, and ``ManipulatorEnv`` is built with
``rclpy``, the executor and the node patched out. That keeps the suite runnable from
``colcon test`` without launching Gazebo.
"""

import threading

import numpy as np
import pytest
from geometry_msgs.msg import Point
from unittest.mock import MagicMock, patch

from distance_based_rl.environment.arm_env import (
    ManipulatorEnv,
    State,
    StateActionReward,
)
from distance_based_rl.environment.data_handler import DataHandler
from distance_based_rl.environment.env_config import EnvConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_handler(config=None, seed=None):
    """Build a DataHandler without touching ROS2, wired for offline testing."""
    with patch.object(DataHandler, '__init__', lambda self: None):
        handler = DataHandler()
    handler.config = config if config is not None else EnvConfig()
    handler._rng = np.random.default_rng(seed)
    handler._lock = threading.Lock()
    handler._manipulator_position = None
    handler._target = None
    handler._joint_positions = None
    handler._joint_velocities = None
    handler._joint_efforts = None
    handler._ee_from_state_cb = False
    handler._ros_ready = True
    handler.target_publisher_ = MagicMock()
    handler.commands_publisher_ = MagicMock()
    # Normally created in __init__; the TF lookup is expected to fail offline, which is
    # the same path taken before the transform is first published.
    handler._tf_buffer = MagicMock()
    handler._ee_frame = handler.config.ee_frame
    handler._base_frame = handler.config.base_frame
    return handler


class FakeRobotState:
    """Stand-in for franka_msgs/FrankaRobotState (not importable without the msgs pkg)."""

    def __init__(self, o_t_ee, q=(), dq=(), tau_j=()):
        self.O_T_EE = list(o_t_ee)
        self.q = list(q)
        self.dq = list(dq)
        self.tau_J = list(tau_j)


def make_env(config=None):
    """Build a ManipulatorEnv with ROS2 fully patched out and a mock node attached."""
    with patch('distance_based_rl.environment.arm_env.rclpy') as mock_rclpy, \
         patch('distance_based_rl.environment.arm_env.MultiThreadedExecutor'), \
         patch('distance_based_rl.environment.arm_env.DataHandler'):
        mock_rclpy.ok.return_value = False  # skip executor/spin-thread setup
        env = ManipulatorEnv(config=config)
    env.node = MagicMock()
    return env


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TestState:
    """The observation vector built from raw ROS2 data."""

    def test_vector_dim_matches_layout(self):
        # 3 EE + 3 target + 7 positions + 7 velocities + 7 efforts + 1 distance
        assert State.vector_dim() == 28

    def test_normalize_vector_pads_and_truncates(self):
        assert State._normalize_vector(None, 3) == (0.0, 0.0, 0.0)
        assert State._normalize_vector((1.0,), 3) == (1.0, 0.0, 0.0)
        assert State._normalize_vector(tuple(range(10)), 7) == tuple(float(i) for i in range(7))
        assert State._normalize_vector(np.array([1.0, 2.0]), 2) == (1.0, 2.0)

    def test_from_sources_defaults_to_zeros(self):
        state = State.from_sources()
        vec = state.as_vector()
        assert vec.shape == (State.vector_dim(),)
        assert vec.dtype == np.float32
        np.testing.assert_array_equal(vec, np.zeros(State.vector_dim(), dtype=np.float32))

    def test_as_vector_normalises_each_block(self):
        state = State.from_sources(
            manipulator_position=(0.75, 0.0, 0.0),
            target_position=(0.0, 0.75, 0.0),
            joint_positions=[np.pi] * 7,
            joint_velocities=[2.5] * 7,
            joint_efforts=[87.0] * 7,
        )
        vec = state.as_vector()

        np.testing.assert_allclose(vec[0:3], [1.0, 0.0, 0.0], atol=1e-6)   # EE / 0.75
        np.testing.assert_allclose(vec[3:6], [0.0, 1.0, 0.0], atol=1e-6)   # target / 0.75
        np.testing.assert_allclose(vec[6:13], [1.0] * 7, atol=1e-6)        # q / pi
        np.testing.assert_allclose(vec[13:20], [1.0] * 7, atol=1e-6)       # dq / 2.5
        np.testing.assert_allclose(vec[20:27], [1.0] * 7, atol=1e-6)       # tau / 87

        # Distance feature: |EE - target| / 0.75
        expected = np.linalg.norm(np.array([0.75, 0.0, 0.0]) - np.array([0.0, 0.75, 0.0])) / 0.75
        np.testing.assert_allclose(vec[27], expected, rtol=1e-6)

    def test_replay_tuple_round_trip(self):
        s = State.from_sources(manipulator_position=(0.1, 0.2, 0.3))
        ns = State.from_sources(manipulator_position=(0.2, 0.2, 0.3))
        transition = StateActionReward(s, np.ones(7), 1.5, ns, True)

        state_v, action_v, reward, next_v, done = transition.as_replay_tuple()

        assert state_v.shape == (State.vector_dim(),)
        assert next_v.shape == (State.vector_dim(),)
        assert action_v.shape == (7,)
        assert reward == 1.5
        assert done is True


# ---------------------------------------------------------------------------
# DataHandler — callbacks
# ---------------------------------------------------------------------------

class TestDataHandlerCallbacks:
    """State ingestion from the two possible ROS2 sources."""

    def test_manipulator_position_property_converts_point(self):
        handler = make_handler()
        handler._manipulator_position = Point(x=1.0, y=2.0, z=3.0)
        assert handler.manipulator_position == (1.0, 2.0, 3.0)

    def test_manipulator_position_none_before_first_message(self):
        assert make_handler().manipulator_position is None

    def test_state_callback_extracts_ee_and_joints(self):
        handler = make_handler()
        # O_T_EE is a column-major 4x4 transform; translation sits at indices 12..14.
        msg = FakeRobotState(
            o_t_ee=[0.0] * 12 + [1.5, 2.5, 3.5] + [1.0],
            q=[0.1] * 7,
            dq=[0.2] * 7,
            tau_j=[0.3] * 7,
        )

        handler.state_callback(msg)

        assert handler.manipulator_position == (1.5, 2.5, 3.5)
        assert handler.get_joint_positions() == tuple([0.1] * 7)
        assert handler.get_joint_velocities() == tuple([0.2] * 7)
        assert handler.get_joint_efforts() == tuple([0.3] * 7)
        # Flags the hardware source as live, which suppresses the TF fallback.
        assert handler._ee_from_state_cb is True

    def test_state_callback_ignores_short_transform(self):
        handler = make_handler()
        handler.state_callback(FakeRobotState(o_t_ee=[0.0] * 4))
        assert handler.manipulator_position is None

    def test_target_callback_updates_target(self):
        handler = make_handler()
        handler.target_callback(Point(x=0.5, y=0.6, z=0.7))
        assert handler.get_target_position() == (0.5, 0.6, 0.7)

    def test_joint_state_callback_reorders_by_name(self):
        """/joint_states ordering is not guaranteed — values must be picked by name."""
        handler = make_handler()
        names = [f'fr3_joint{i}' for i in range(7, 0, -1)]  # reversed on purpose
        msg = MagicMock()
        msg.name = names
        msg.position = [float(i) for i in range(7, 0, -1)]  # 7..1
        msg.velocity = []
        msg.effort = []

        handler._joint_state_callback(msg)

        # joint1..joint7 in canonical order
        assert handler.get_joint_positions() == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)

    def test_joint_state_callback_ignores_foreign_messages(self):
        """A gripper-only /joint_states message must not clobber arm state."""
        handler = make_handler()
        msg = MagicMock()
        msg.name = ['fr3_finger_joint1', 'fr3_finger_joint2']
        msg.position = [0.01, 0.01]
        msg.velocity = []
        msg.effort = []

        handler._joint_state_callback(msg)

        assert handler.get_joint_positions() is None


# ---------------------------------------------------------------------------
# DataHandler — commands
# ---------------------------------------------------------------------------

class TestDataHandlerCommands:
    """The action interface: incremental joint deltas, clipped to hardware limits."""

    def test_publish_command_is_incremental(self):
        handler = make_handler()
        handler._joint_positions = tuple([0.0] * 7)

        handler.publish_command(np.array([1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0]))

        sent = handler.commands_publisher_.publish.call_args[0][0].data
        delta = handler.config.max_joint_delta
        np.testing.assert_allclose(
            sent[:3], [delta, -delta, 0.5 * delta], atol=1e-9
        )

    def test_publish_command_clips_action_before_scaling(self):
        """An out-of-range action must saturate at one delta, not scale past it."""
        handler = make_handler()
        # Start from the home pose: unlike an all-zeros vector it is a valid FR3
        # configuration (joint 4 is limited to [-3.0718, -0.0698], so 0.0 is illegal).
        home = DataHandler._HOME_POSITION
        handler._joint_positions = tuple(home.tolist())

        handler.publish_command(np.array([5.0] * 7))  # far outside [-1, 1]

        sent = np.asarray(handler.commands_publisher_.publish.call_args[0][0].data)
        expected = np.clip(
            home + handler.config.max_joint_delta,
            DataHandler._JOINT_LOW,
            DataHandler._JOINT_HIGH,
        )
        np.testing.assert_allclose(sent, expected, atol=1e-9)

    def test_publish_command_respects_joint_limits(self):
        handler = make_handler()
        # Start at the upper limit and push further in the same direction.
        handler._joint_positions = tuple(DataHandler._JOINT_HIGH.tolist())

        handler.publish_command(np.ones(7))

        sent = np.asarray(handler.commands_publisher_.publish.call_args[0][0].data)
        assert np.all(sent <= DataHandler._JOINT_HIGH + 1e-9)
        assert np.all(sent >= DataHandler._JOINT_LOW - 1e-9)

    def test_publish_command_without_joint_feedback_fails(self):
        """Incremental control is impossible without knowing the current pose."""
        handler = make_handler()
        handler._joint_positions = None

        assert handler.publish_command(np.zeros(7)) is False
        handler.commands_publisher_.publish.assert_not_called()

    def test_publish_command_when_ros_not_ready(self):
        handler = make_handler()
        handler._ros_ready = False
        assert handler.publish_command(np.zeros(7)) is False

    def test_go_home_publishes_home_configuration(self):
        handler = make_handler()
        assert handler.go_home() is True
        sent = handler.commands_publisher_.publish.call_args[0][0].data
        np.testing.assert_allclose(sent, DataHandler._HOME_POSITION, atol=1e-9)


# ---------------------------------------------------------------------------
# DataHandler — target sampling
# ---------------------------------------------------------------------------

class TestTargetSampling:
    """Targets must land inside the reachable, singularity-free workspace."""

    @staticmethod
    def _spherical(target):
        x, y, z = target
        r = float(np.linalg.norm([x, y, z]))
        elevation = float(np.arcsin(z / r))
        azimuth = float(np.arctan2(y, x))
        return r, azimuth, elevation

    def test_samples_stay_inside_configured_volume(self):
        handler = make_handler(seed=0)
        cfg = handler.config
        r_lo, r_hi = cfg.radius_bounds
        az_lo, az_hi = (b * 2.0 * np.pi for b in cfg.azimuth_bounds)
        el_lo, el_hi = (b * np.pi / 2.0 for b in cfg.elevation_bounds)

        for _ in range(200):
            assert handler.set_random_target() is True
            r, azimuth, elevation = self._spherical(handler.get_target_position())

            assert r_lo - 1e-9 <= r <= r_hi + 1e-9
            assert az_lo - 1e-9 <= azimuth <= az_hi + 1e-9
            assert el_lo - 1e-9 <= elevation <= el_hi + 1e-9

    def test_samples_are_above_the_floor(self):
        """
        Targets must clear the table.

        The guaranteed clearance is set by the *smallest* radius:
        r_min · sin(22.5°) ≈ 0.077 m.
        """
        handler = make_handler(seed=1)
        cfg = handler.config
        floor = cfg.radius_bounds[0] * np.sin(cfg.elevation_bounds[0] * np.pi / 2.0)

        for _ in range(200):
            handler.set_random_target()
            assert handler.get_target_position()[2] >= floor - 1e-9

    def test_seeded_sampling_is_reproducible(self):
        a = make_handler(seed=42)
        b = make_handler(seed=42)
        c = make_handler(seed=43)

        a.set_random_target()
        b.set_random_target()
        c.set_random_target()

        assert a.get_target_position() == b.get_target_position()
        assert a.get_target_position() != c.get_target_position()

    def test_reseeding_replays_the_same_sequence(self):
        handler = make_handler(seed=7)
        first = [(handler.set_random_target(), handler.get_target_position())[1] for _ in range(3)]

        handler.seed(7)
        again = [(handler.set_random_target(), handler.get_target_position())[1] for _ in range(3)]

        assert first == again

    def test_does_not_touch_global_numpy_state(self):
        """Sampling must not perturb the global RNG other components rely on."""
        handler = make_handler(seed=3)
        np.random.seed(123)
        before = np.random.rand()

        np.random.seed(123)
        handler.set_random_target()
        after = np.random.rand()

        assert before == after

    def test_target_is_published_and_cached(self):
        handler = make_handler(seed=0)
        handler.set_random_target()

        handler.target_publisher_.publish.assert_called_once()
        published = handler.target_publisher_.publish.call_args[0][0]
        assert handler.get_target_position() == (published.x, published.y, published.z)

    def test_returns_false_when_ros_not_ready(self):
        handler = make_handler()
        handler._ros_ready = False
        assert handler.set_random_target() is False


# ---------------------------------------------------------------------------
# ManipulatorEnv
# ---------------------------------------------------------------------------

class TestManipulatorEnvSpaces:
    """Gymnasium API surface."""

    def test_observation_space_matches_state_layout(self):
        env = make_env()
        assert env.observation_space.shape == (State.vector_dim(),)
        assert env.observation_space.dtype == np.float32

    def test_action_space_is_seven_normalised_joints(self):
        env = make_env()
        assert env.action_space.shape == (7,)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0


class TestManipulatorEnvStep:
    """Reward, termination and the guard against missing ROS2 data."""

    @staticmethod
    def _env_with_positions(ee, target, config=None):
        env = make_env(config=config)
        env.node.get_manipulator_position.return_value = ee
        env.node.get_target_position.return_value = target
        env.node.get_joint_positions.return_value = tuple([0.0] * 7)
        env.node.get_joint_velocities.return_value = tuple([0.0] * 7)
        env.node.get_joint_efforts.return_value = tuple([0.0] * 7)
        env.node.publish_command.return_value = True
        # Settling is a wall-clock wait against a real controller; irrelevant offline.
        env._wait_until_settled = lambda: None
        return env

    def test_reward_is_negative_distance(self):
        env = self._env_with_positions((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

        _, reward, terminated, truncated, info = env.step(np.zeros(7))

        assert reward == pytest.approx(-1.0)
        assert terminated is False
        assert truncated is False
        assert info['distance'] == pytest.approx(1.0)

    def test_reaching_target_terminates_with_bonus(self):
        env = self._env_with_positions((0.0, 0.0, 0.0), (0.02, 0.0, 0.0))

        _, reward, terminated, _, info = env.step(np.zeros(7))

        assert terminated is True
        assert info['success'] is True
        # Bonus is additive, so the distance-shaped gradient survives the terminal step.
        assert reward == pytest.approx(env.config.success_bonus - 0.02)

    def test_success_reported_at_every_threshold(self):
        env = self._env_with_positions((0.0, 0.0, 0.0), (0.07, 0.0, 0.0))

        _, _, terminated, _, info = env.step(np.zeros(7))

        assert terminated is True                 # inside the 10 cm terminating threshold
        assert info['success_at']['0.10'] is True
        assert info['success_at']['0.05'] is False
        assert info['success_at']['0.02'] is False

    def test_truncates_at_step_budget(self):
        env = self._env_with_positions(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), config=EnvConfig(max_episode_steps=3)
        )

        for _ in range(2):
            _, _, terminated, truncated, _ = env.step(np.zeros(7))
            assert not terminated and not truncated

        _, _, terminated, truncated, _ = env.step(np.zeros(7))
        assert truncated is True
        assert terminated is False

    def test_missing_position_does_not_fake_success(self):
        """
        Missing data must not be read as reaching the target.

        Without the guard both positions default to (0,0,0), distance collapses to 0 and
        the episode would terminate with the success bonus.
        """
        env = self._env_with_positions(None, (1.0, 0.0, 0.0))

        _, reward, terminated, _, info = env.step(np.zeros(7))

        assert terminated is False
        assert reward == 0.0
        assert info['success'] is False

    def test_step_publishes_the_action(self):
        env = self._env_with_positions((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        action = np.full(7, 0.25)

        env.step(action)

        np.testing.assert_array_equal(env.node.publish_command.call_args[0][0], action)


class TestManipulatorEnvLifecycle:
    """Resource handling around rclpy."""

    def test_close_does_not_shutdown_rclpy_it_does_not_own(self):
        env = make_env()
        env._owns_rclpy = False
        env.executor = MagicMock()
        env.spin_thread = None

        with patch('distance_based_rl.environment.arm_env.rclpy') as mock_rclpy:
            mock_rclpy.ok.return_value = True
            env.close()
            mock_rclpy.shutdown.assert_not_called()

    def test_close_shuts_down_rclpy_it_owns(self):
        env = make_env()
        env._owns_rclpy = True
        env.executor = MagicMock()
        env.spin_thread = None

        with patch('distance_based_rl.environment.arm_env.rclpy') as mock_rclpy:
            mock_rclpy.ok.return_value = True
            env.close()
            mock_rclpy.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# EnvConfig
# ---------------------------------------------------------------------------

class TestEnvConfig:
    """Configuration is the serialisable description of a run."""

    def test_environment_variables_are_honoured(self, monkeypatch):
        monkeypatch.setenv('FRANKA_MAX_JOINT_DELTA', '0.25')
        monkeypatch.setenv('MAX_STEPS_PER_EPISODE', '123')
        config = EnvConfig()
        assert config.max_joint_delta == 0.25
        assert config.max_episode_steps == 123

    def test_malformed_environment_variable_falls_back(self, monkeypatch):
        monkeypatch.setenv('FRANKA_MAX_JOINT_DELTA', 'not-a-number')
        assert EnvConfig().max_joint_delta == 0.1

    def test_explicit_arguments_win_over_environment(self, monkeypatch):
        monkeypatch.setenv('MAX_STEPS_PER_EPISODE', '123')
        assert EnvConfig(max_episode_steps=7).max_episode_steps == 7

    def test_save_load_round_trip(self, tmp_path):
        config = EnvConfig(max_episode_steps=42, min_distance_threshold=0.03)
        path = str(tmp_path / 'env_config.json')

        config.save(path)
        loaded = EnvConfig.load(path)

        assert loaded.max_episode_steps == 42
        assert loaded.min_distance_threshold == 0.03
        assert loaded.report_thresholds == config.report_thresholds


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
