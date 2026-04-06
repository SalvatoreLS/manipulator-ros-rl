#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Float64MultiArray
from franka_msgs.action import Grasp
from franka_msgs.action import Move
from pynput import keyboard
import time
import threading


# ---------------------------------------------------------------------------
# Key → (joint_index, direction)  mapping
# ---------------------------------------------------------------------------
KEY_MAP = {
    # Joint 0 – base rotation
    keyboard.Key.up:    (0, +1),
    keyboard.Key.down:  (0, -1),

    # Joint 1 – shoulder
    keyboard.Key.left:  (1, +1),
    keyboard.Key.right: (1, -1),

    # Joint 2 – elbow
    'w': (2, +1),
    's': (2, -1),

    # Joint 3 – forearm roll
    'a': (3, +1),
    'd': (3, -1),

    # Joint 4 – wrist pitch
    'q': (4, +1),
    'e': (4, -1),

    # Joint 5 – wrist roll
    'z': (5, +1),
    'x': (5, -1),

    # Joint 6 – flange / end-effector rotation
    'i': (6, +1),
    'k': (6, -1),
}

RESET_KEY   = 'r'   # Reset all joints to 0.0
GRIPPER_KEY = keyboard.Key.space  # Hold to close gripper

KEY_LABELS = {
    keyboard.Key.up:    '↑',
    keyboard.Key.down:  '↓',
    keyboard.Key.left:  '←',
    keyboard.Key.right: '→',
    keyboard.Key.space: 'SPACE',
}

JOINT_NAMES = [
    'Joint 0  – Base rotation     ',
    'Joint 1  – Shoulder          ',
    'Joint 2  – Elbow             ',
    'Joint 3  – Forearm roll      ',
    'Joint 4  – Wrist pitch       ',
    'Joint 5  – Wrist roll        ',
    'Joint 6  – Flange rotation   ',
]

# ---------------------------------------------------------------------------
# Gripper constants
# ---------------------------------------------------------------------------
GRIPPER_MAX_WIDTH   = 0.08   # metres – fully open (Franka Hand default)
GRIPPER_MIN_WIDTH   = 0.0    # metres – fully closed
GRIPPER_CLOSE_SPEED = 0.04   # m/s    – closing speed while SPACE held
GRIPPER_OPEN_SPEED  = 0.08   # m/s    – re-opening speed when SPACE released
GRIPPER_FORCE       = 10.0   # N      – used by Grasp fallback
GRIPPER_EPSILON     = 0.005  # m      – used by Grasp fallback
GRIPPER_CMD_EPS     = 0.001  # m      – avoid sending near-identical move goals
GRIPPER_CMD_PERIOD  = 0.05   # s      – max 20 Hz goal update rate


def _key_label(k) -> str:
    """Return a human-readable label for a key."""
    if k in KEY_LABELS:
        return KEY_LABELS[k]
    if isinstance(k, str):
        return k.upper()
    return str(k)


def _normalize_key(key):
    """
    Normalise a pynput key to the same type used in KEY_MAP / RESET_KEY.
    Special keys stay as keyboard.Key enums; character keys become lowercase str.
    """
    if isinstance(key, keyboard.KeyCode):
        return key.char.lower() if key.char else None
    return key  # already a keyboard.Key enum


class KeyboardArmController(Node):

    MAX_VALUE    = 2.0    # rad  – per-joint saturation limit
    STEP_SPEED   = 0.8    # rad/s – how fast a joint moves while key held
    LOOP_PERIOD  = 0.01   # s    – update-loop tick (100 Hz)

    def __init__(self):
        super().__init__('keyboard_arm_controller')

        # Declare ROS2 parameters for flexibility
        self.declare_parameter(
            'gripper_move_action_name',
            '/fr3_gripper/move'
        )
        self.declare_parameter(
            'gripper_grasp_action_name',
            ''
        )
        self.declare_parameter(
            'gripper_action_name',
            ''
        )
        self.declare_parameter(
            'joint_command_topic',
            '/forward_position_controller/commands'
        )
        
        # Get parameter values
        gripper_move_action_name = self.get_parameter('gripper_move_action_name').get_parameter_value().string_value
        gripper_grasp_action_name = self.get_parameter('gripper_grasp_action_name').get_parameter_value().string_value
        legacy_gripper_action_name = self.get_parameter('gripper_action_name').get_parameter_value().string_value
        joint_command_topic = self.get_parameter('joint_command_topic').get_parameter_value().string_value

        # Backward compatibility with old single gripper action parameter.
        if legacy_gripper_action_name:
            if legacy_gripper_action_name.endswith('/grasp') and not gripper_grasp_action_name:
                gripper_grasp_action_name = legacy_gripper_action_name
            if legacy_gripper_action_name.endswith('/grasp') and gripper_move_action_name == '/fr3_gripper/move':
                gripper_move_action_name = legacy_gripper_action_name[:-len('/grasp')] + '/move'
            self.get_logger().warn(
                'Parameter "gripper_action_name" is deprecated; use "gripper_move_action_name" and/or "gripper_grasp_action_name".'
            )

        if not gripper_grasp_action_name:
            if gripper_move_action_name.endswith('/move'):
                gripper_grasp_action_name = gripper_move_action_name[:-len('/move')] + '/grasp'
            else:
                gripper_grasp_action_name = '/fr3_gripper/grasp'

        # Joint position publisher
        self.publisher = self.create_publisher(
            Float64MultiArray,
            joint_command_topic,
            10,
        )

        # Gripper action clients. Prefer Move; fall back to Grasp if Move is unavailable.
        self._gripper_move_client = ActionClient(
            self, Move, gripper_move_action_name
        )
        self._gripper_grasp_client = ActionClient(
            self, Grasp, gripper_grasp_action_name
        )
        self._gripper_move_action_name = gripper_move_action_name
        self._gripper_grasp_action_name = gripper_grasp_action_name
        self.get_logger().info(f'Gripper move action client: {gripper_move_action_name}')
        self.get_logger().info(f'Gripper grasp action client (fallback): {gripper_grasp_action_name}')
        self.get_logger().info(f'Joint command topic: {joint_command_topic}')

        self.joint_positions: list[float] = [0.0] * 7
        self.active_keys: set = set()
        self._lock = threading.Lock()

        # Gripper state
        self._gripper_width   = GRIPPER_MAX_WIDTH   # starts fully open
        self._space_pressed   = False
        self._gripper_goal_handle = None
        self._last_gripper_cmd_width = None
        self._last_gripper_cmd_time = 0.0
        self._gripper_ready_log_emitted = False
        self._gripper_initialized = False
        self._gripper_mode = None  # 'move' or 'grasp'
        self._last_gripper_discovery_time = 0.0

        self.listener     = None
        self._loop_thread = threading.Thread(
            target=self._update_loop, daemon=True
        )

    # ------------------------------------------------------------------
    # Key callbacks
    # ------------------------------------------------------------------

    def on_key_press(self, key):
        norm = _normalize_key(key)
        if norm is None:
            return

        # Reset joints
        if norm == RESET_KEY:
            with self._lock:
                self.joint_positions = [0.0] * 7
            self.get_logger().info('All joints reset to 0.0')
            self._publish()
            return

        # Gripper close (spacebar)
        if norm == GRIPPER_KEY:
            with self._lock:
                self._space_pressed = True
            return

        if norm in KEY_MAP:
            with self._lock:
                self.active_keys.add(norm)

    def on_key_release(self, key):
        norm = _normalize_key(key)
        if norm is None:
            return

        # Gripper release → start re-opening
        if norm == GRIPPER_KEY:
            with self._lock:
                self._space_pressed = False
            return

        with self._lock:
            self.active_keys.discard(norm)

    # ------------------------------------------------------------------
    # Single update loop (runs in its own daemon thread)
    # ------------------------------------------------------------------

    def _update_loop(self):
        last_time = time.time()
        while rclpy.ok():
            now  = time.time()
            dt   = now - last_time
            last_time = now

            with self._lock:
                active_snapshot = set(self.active_keys)
                space = self._space_pressed

            # ---- Joint control ----------------------------------------
            moved = False
            for k in active_snapshot:
                if k not in KEY_MAP:
                    continue
                joint_idx, direction = KEY_MAP[k]
                delta = direction * self.STEP_SPEED * dt

                with self._lock:
                    new_val = self.joint_positions[joint_idx] + delta
                    new_val = max(-self.MAX_VALUE, min(self.MAX_VALUE, new_val))
                    self.joint_positions[joint_idx] = new_val
                moved = True

            if moved:
                self._publish()

            # ---- Gripper control (Progressive closing with spacebar) ----
            prev_width = self._gripper_width
            if space:
                # Progressively close while spacebar is held
                self._gripper_width = max(
                    GRIPPER_MIN_WIDTH,
                    self._gripper_width - GRIPPER_CLOSE_SPEED * dt,
                )
            else:
                # Re-open when spacebar released
                self._gripper_width = min(
                    GRIPPER_MAX_WIDTH,
                    self._gripper_width + GRIPPER_OPEN_SPEED * dt,
                )

            # Only send a new goal if width changed meaningfully
            width_changed = abs(self._gripper_width - prev_width) > 1e-4
            if width_changed:
                self._send_gripper_goal(
                    self._gripper_width,
                    GRIPPER_CLOSE_SPEED if space else GRIPPER_OPEN_SPEED,
                )
            elif not self._gripper_initialized:
                # Ensure gripper opens once at startup even without key input.
                self._send_gripper_goal(
                    GRIPPER_MAX_WIDTH,
                    GRIPPER_OPEN_SPEED,
                    force_send=True,
                )

            time.sleep(self.LOOP_PERIOD)

    # ------------------------------------------------------------------
    # Gripper action
    # ------------------------------------------------------------------

    def _discover_gripper_action_names(self):
        """Find gripper move/grasp action names from ROS graph services."""
        now = time.time()
        if (now - self._last_gripper_discovery_time) < 1.0:
            return None, None
        self._last_gripper_discovery_time = now

        move_candidates = []
        grasp_candidates = []
        for service_name, _service_types in self.get_service_names_and_types():
            if not service_name.endswith('/_action/send_goal'):
                continue
            action_name = service_name[:-len('/_action/send_goal')]
            if action_name.endswith('/move'):
                move_candidates.append(action_name)
            elif action_name.endswith('/grasp'):
                grasp_candidates.append(action_name)

        def pick_best(candidates):
            if not candidates:
                return None
            fr3 = [c for c in candidates if '/fr3_gripper/' in c]
            if fr3:
                return sorted(fr3)[0]
            return sorted(candidates)[0]

        return pick_best(move_candidates), pick_best(grasp_candidates)

    def _rebind_gripper_clients(self, move_action_name: str | None, grasp_action_name: str | None):
        """Recreate action clients for newly discovered names."""
        changed = False
        if move_action_name and move_action_name != self._gripper_move_action_name:
            self._gripper_move_action_name = move_action_name
            self._gripper_move_client = ActionClient(self, Move, move_action_name)
            changed = True
        if grasp_action_name and grasp_action_name != self._gripper_grasp_action_name:
            self._gripper_grasp_action_name = grasp_action_name
            self._gripper_grasp_client = ActionClient(self, Grasp, grasp_action_name)
            changed = True
        if changed:
            self._gripper_mode = None
            self.get_logger().warn(
                'Discovered gripper action names from ROS graph: '
                f'move={self._gripper_move_action_name}, grasp={self._gripper_grasp_action_name}'
            )

    def _select_gripper_client(self):
        """Choose active gripper interface. Prefer Move, fallback to Grasp."""
        if self._gripper_mode == 'move' and self._gripper_move_client.server_is_ready():
            return 'move', self._gripper_move_client
        if self._gripper_mode == 'grasp' and self._gripper_grasp_client.server_is_ready():
            return 'grasp', self._gripper_grasp_client

        discovered_move, discovered_grasp = self._discover_gripper_action_names()
        self._rebind_gripper_clients(discovered_move, discovered_grasp)

        if self._gripper_move_client.server_is_ready():
            if self._gripper_mode != 'move':
                self.get_logger().info(f'Using Move action: {self._gripper_move_action_name}')
            self._gripper_mode = 'move'
            return 'move', self._gripper_move_client

        if self._gripper_grasp_client.server_is_ready():
            if self._gripper_mode != 'grasp':
                self.get_logger().warn(
                    f'Move action not available, falling back to Grasp action: {self._gripper_grasp_action_name}'
                )
            self._gripper_mode = 'grasp'
            return 'grasp', self._gripper_grasp_client

        return None, None

    def _send_gripper_goal(self, width: float, speed: float, force_send: bool = False):
        """Send a gripper goal using the first available action interface."""
        mode, client = self._select_gripper_client()
        if client is None:
            if not self._gripper_ready_log_emitted:
                self.get_logger().warn(
                    'No gripper action server is ready yet; waiting... '
                    f'(move: {self._gripper_move_action_name}, grasp: {self._gripper_grasp_action_name})'
                )
                self._gripper_ready_log_emitted = True
            return
        if self._gripper_ready_log_emitted:
            self.get_logger().info('A gripper action server is now ready.')
            self._gripper_ready_log_emitted = False

        now = time.time()
        if not force_send:
            if self._last_gripper_cmd_width is not None:
                if abs(width - self._last_gripper_cmd_width) < GRIPPER_CMD_EPS:
                    return
            if (now - self._last_gripper_cmd_time) < GRIPPER_CMD_PERIOD:
                return

        if mode == 'move':
            goal = Move.Goal()
            goal.width = float(width)
            goal.speed = float(speed)
        else:
            goal = Grasp.Goal()
            goal.width = float(width)
            goal.speed = float(speed)
            goal.force = GRIPPER_FORCE
            goal.epsilon.inner = GRIPPER_EPSILON
            goal.epsilon.outer = GRIPPER_EPSILON

        # Fire-and-forget; cancel previous if still active
        if self._gripper_goal_handle is not None:
            try:
                self._gripper_goal_handle.cancel_goal_async()
            except Exception:
                pass

        future = client.send_goal_async(goal)
        future.add_done_callback(self._on_gripper_goal_response)
        self._last_gripper_cmd_width = float(width)
        self._last_gripper_cmd_time = now

    def _on_gripper_goal_response(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn('Gripper goal was rejected by server.')
                return
            self._gripper_goal_handle = goal_handle
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self._on_gripper_result)
        except Exception as exc:
            self.get_logger().warn(f'Failed to send gripper goal: {exc}')

    def _on_gripper_result(self, future):
        try:
            result = future.result().result
            if result.success:
                self._gripper_initialized = True
            else:
                self.get_logger().warn(f'Gripper action failed: {result.error}')
        except Exception as exc:
            self.get_logger().warn(f'Failed to read gripper action result: {exc}')

    # ------------------------------------------------------------------
    # Joint publisher
    # ------------------------------------------------------------------

    def _publish(self):
        msg = Float64MultiArray()
        with self._lock:
            msg.data = list(self.joint_positions)
        self.publisher.publish(msg)

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start(self):
        self._print_keybindings()

        self._loop_thread.start()

        self.listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release,
        )
        self.listener.start()

    # ------------------------------------------------------------------
    # Key-binding summary printed at startup
    # ------------------------------------------------------------------

    def _print_keybindings(self):
        rows: dict[int, list[str]] = {i: [] for i in range(7)}
        for k, (idx, direction) in KEY_MAP.items():
            arrow = '+' if direction > 0 else '-'
            rows[idx].append(f'{_key_label(k)} ({arrow})')

        sep   = '─' * 60
        lines = [
            '',
            sep,
            '  Keyboard Arm Controller — key bindings',
            sep,
        ]
        for i, name in enumerate(JOINT_NAMES):
            keys_str = '  /  '.join(rows[i])
            lines.append(f'  {name}  {keys_str}')

        lines += [
            '',
            f'  {RESET_KEY.upper()}              Reset all joints to 0.0',
            f'  SPACE (hold)   Close gripper progressively',
            f'  SPACE (release) Re-open gripper',
            f'  Ctrl-C         Shutdown',
            '',
            f'  MAX_VALUE      = ±{self.MAX_VALUE} rad',
            f'  STEP_SPEED     =  {self.STEP_SPEED} rad/s',
            f'  GRIPPER range  =  {GRIPPER_MIN_WIDTH*100:.0f} – {GRIPPER_MAX_WIDTH*100:.0f} cm',
            f'  GRIPPER speed  =  {GRIPPER_CLOSE_SPEED*100:.0f} cm/s (close) / {GRIPPER_OPEN_SPEED*100:.0f} cm/s (open)',
            sep,
            '',
        ]
        print('\n'.join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    controller = KeyboardArmController()
    controller.start()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down…')
    finally:
        if controller.listener:
            controller.listener.stop()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
