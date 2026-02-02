"""Input device handlers: keyboard and joystick (optional).

Provides lightweight controllers that expose a get_direction() method
returning an (dx, dy) unit vector (or (0,0) when idle).
"""
from pynput.keyboard import Key, Listener


class KeyboardController:
    """Maintain currently pressed arrow keys and expose direction."""

    def __init__(self):
        self._pressed = set()
        self._listener = Listener(on_press=self._on_press, on_release=self._on_release)

    def _on_press(self, key):
        try:
            if key in (Key.up, Key.down, Key.left, Key.right):
                self._pressed.add(key)
        except Exception:
            pass

    def _on_release(self, key):
        try:
            if key in (Key.up, Key.down, Key.left, Key.right):
                self._pressed.discard(key)
        except Exception:
            pass

    def start(self):
        self._listener.start()

    def stop(self):
        try:
            self._listener.stop()
        except Exception:
            pass

    def get_direction(self):
        """Return (dx, dy) where dx/dy are in [-1, 1]."""
        dx = 0
        dy = 0
        if Key.right in self._pressed:
            dx += 1
        if Key.left in self._pressed:
            dx -= 1
        if Key.up in self._pressed:
            dy += 1
        if Key.down in self._pressed:
            dy -= 1

        if dx == 0 and dy == 0:
            return 0.0, 0.0

        # Normalize so diagonal isn't larger
        mag = (dx ** 2 + dy ** 2) ** 0.5
        return dx / mag, dy / mag


# Joystick support via pygame (optional)
class JoystickController:
    """Read the first joystick axes (x, y) and return normalized direction.

    Falls back with zero direction when no joystick available.
    """

    def __init__(self, deadzone=0.2):
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            self.pygame = pygame
            if pygame.joystick.get_count() == 0:
                raise RuntimeError("No joystick found")
            self.j = pygame.joystick.Joystick(0)
            self.j.init()
            self.deadzone = deadzone
        except Exception as e:
            raise RuntimeError(f"Joystick initialization failed: {e}")

    def get_direction(self):
        self.pygame.event.pump()
        # Common mapping: axis 0 -> x, axis 1 -> y
        x = self.j.get_axis(0)
        y = -self.j.get_axis(1)  # invert Y so up is positive
        # Apply deadzone
        if abs(x) < self.deadzone and abs(y) < self.deadzone:
            return 0.0, 0.0
        mag = (x ** 2 + y ** 2) ** 0.5
        return x / mag, y / mag
