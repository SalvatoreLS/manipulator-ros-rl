#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
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

RESET_KEY = 'r'   # Reset all joints to 0.0

KEY_LABELS = {
    keyboard.Key.up:    '↑',
    keyboard.Key.down:  '↓',
    keyboard.Key.left:  '←',
    keyboard.Key.right: '→',
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

    MAX_VALUE       = 2.0    # rad  – per-joint saturation limit
    STEP_SPEED      = 0.8    # rad/s – how fast a joint moves while key held
    LOOP_PERIOD     = 0.01   # s    – update-loop tick (100 Hz)

    def __init__(self):
        super().__init__('keyboard_arm_controller')

        self.publisher = self.create_publisher(
            Float64MultiArray,
            '/forward_position_controller/commands',
            10,
        )

        self.joint_positions: list[float] = [0.0] * 7
        self.active_keys: set = set()
        self._lock = threading.Lock()

        self.listener   = None
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

        if norm == RESET_KEY:
            with self._lock:
                self.joint_positions = [0.0] * 7
            self.get_logger().info('All joints reset to 0.0')
            self._publish()
            return

        if norm in KEY_MAP:
            with self._lock:
                self.active_keys.add(norm)

    def on_key_release(self, key):
        norm = _normalize_key(key)
        if norm is None:
            return
        with self._lock:
            self.active_keys.discard(norm)

    # ------------------------------------------------------------------
    # Single update loop (runs in its own daemon thread)
    # ------------------------------------------------------------------

    def _update_loop(self):
        last_time = time.time()
        while rclpy.ok():
            now       = time.time()
            dt        = now - last_time
            last_time = now

            with self._lock:
                active_snapshot = set(self.active_keys)

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

            time.sleep(self.LOOP_PERIOD)

    # ------------------------------------------------------------------
    # Publisher
    # ------------------------------------------------------------------

    def _publish(self):
        msg = Float64MultiArray()
        with self._lock:
            msg.data = list(self.joint_positions)
        self.publisher.publish(msg)
        #self.get_logger().info(
        #    'Joints: ' + '  '.join(f'J{i}={v:+.3f}' for i, v in enumerate(msg.data))
        #)

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
        # Build one row per joint showing both keys for that joint
        rows: dict[int, list[str]] = {i: [] for i in range(7)}
        for k, (idx, direction) in KEY_MAP.items():
            arrow = '+' if direction > 0 else '-'
            rows[idx].append(f'{_key_label(k)} ({arrow})')

        sep   = '─' * 56
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
            f'  Ctrl-C         Shutdown',
            '',
            f'  MAX_VALUE    = ±{self.MAX_VALUE} rad',
            f'  STEP_SPEED   =  {self.STEP_SPEED} rad/s',
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
        if controller.listener: controller.listener.stop()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()