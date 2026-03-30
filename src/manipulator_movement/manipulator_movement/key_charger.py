import keyboard
import time
import threading

class KeyCharger:
    def __init__(self):
        self.MAX_VALUE = 1  # Define the maximum value for the charge
        self.SATURATION_SPEED = 0.5  # Define how fast the value saturates (units per second)
        self.current_value = 0  # Current charge value
        self.is_pressed = False  # Track if a key is currently pressed
        self.last_update_time = time.time()  # Track the last update time

        # Register keyboard event handlers
        keyboard.on_press(self.on_key_press)
        keyboard.on_release(self.on_key_release)

    def on_key_press(self, event):
        if event.name in ['up', 'down'] and not self.is_pressed:
            self.is_pressed = True
            self.last_update_time = time.time()
            threading.Thread(target=self.update_charge, args=(event.name,)).start()

    def on_key_release(self, event):
        if event.name in ['up', 'down'] and self.is_pressed:
            self.is_pressed = False
            self.current_value = 0  # Reset charge on release

    def update_charge(self, key):
        while self.is_pressed:
            current_time = time.time()
            delta_time = current_time - self.last_update_time
            self.last_update_time = current_time

            if key == 'up':
                self.current_value += self.SATURATION_SPEED * delta_time
                if self.current_value > self.MAX_VALUE:
                    self.current_value = self.MAX_VALUE
            elif key == 'down':
                self.current_value -= self.SATURATION_SPEED * delta_time
                if self.current_value < -self.MAX_VALUE:
                    self.current_value = -self.MAX_VALUE

            print(f"Current charge: {self.current_value:.2f}")
            time.sleep(0.01)  # Small delay to prevent CPU overload

    def start(self):
        print("KeyCharger started. Press 'up' or 'down' to charge.")
        keyboard.wait()  # Block and wait for keyboard events

# Example usage
if __name__ == "__main__":
    charger = KeyCharger()
    charger.start()