"""Video recording with circular pre-buffer and GPIO controls"""

import time
import threading
from collections import deque
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
from gpiozero import Button, LED

from moosedetector.config import DataCollectionConfig


class VideoRecorder:
    """Manages video recording with pre-buffer, GPIO button, and LED indicator.

    Thread safety model:
    - feed_frame() is called from the processing thread only
    - _on_button_press() is called from gpiozero's callback thread
    - The button callback sets a _toggle_requested flag; the processing thread
      acts on it in feed_frame() to avoid VideoWriter cross-thread issues
    """

    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self._lock = threading.Lock()

        # Circular buffers for pre-recording (raw + overlay + timestamps)
        maxlen = config.buffer_maxlen  # ~90 frames for 10s at 9fps
        self._raw_buffer = deque(maxlen=maxlen)
        self._overlay_buffer = deque(maxlen=maxlen)
        self._timestamp_buffer = deque(maxlen=maxlen)

        # Recording state
        self._recording = False
        self._toggle_requested = False  # Set by button callback, consumed by processing thread
        self._raw_writer = None
        self._overlay_writer = None
        self._frame_size = None  # (width, height), set on first frame
        self._record_frame_count = 0
        self._record_start_time = 0.0

        # GPIO
        self._button = None
        self._led = None

        # Create save directory
        config.save_directory.mkdir(parents=True, exist_ok=True)

        # Initialize GPIO
        if config.enabled:
            self._init_gpio()

    def _init_gpio(self):
        """Initialize button and LED GPIO devices"""
        try:
            self._button = Button(self.config.button_pin, bounce_time=0.3)
            self._button.when_pressed = self._on_button_press
            print(f"Recording button initialized on GPIO {self.config.button_pin}")
        except Exception as e:
            print(f"WARNING: Button GPIO init failed: {e}")
            self._button = None

        try:
            self._led = LED(self.config.led_pin)
            print(f"Recording LED initialized on GPIO {self.config.led_pin}")
        except Exception as e:
            print(f"WARNING: LED GPIO init failed: {e}")
            self._led = None

    def _on_button_press(self):
        """Button callback - runs in gpiozero's callback thread.

        Only sets a flag. The processing thread acts on it in feed_frame()
        to keep all VideoWriter operations on a single thread.
        """
        with self._lock:
            self._toggle_requested = True

    def feed_frame(self, raw_frame: np.ndarray, overlay_frame: np.ndarray):
        """Feed a frame pair into the recorder. Called every frame from processing thread.

        Args:
            raw_frame: RGB uint8 array (h, w, 3) - raw thermal image
            overlay_frame: RGB uint8 array (h, w, 3) - with detection overlays
        """
        # Capture frame size on first frame
        if self._frame_size is None:
            h, w = raw_frame.shape[:2]
            self._frame_size = (w, h)

        # Check for toggle request (from button press)
        toggle = False
        with self._lock:
            if self._toggle_requested:
                toggle = True
                self._toggle_requested = False

        if toggle:
            if self._recording:
                self._stop_recording()
            else:
                self._start_recording()

        # Feed frames
        if self._recording:
            self._write_frame(raw_frame, overlay_frame)
            self._record_frame_count += 1
        else:
            # Append to circular pre-buffer (with timestamp)
            self._raw_buffer.append(raw_frame.copy())
            self._overlay_buffer.append(overlay_frame.copy())
            self._timestamp_buffer.append(time.time())

    def _calculate_actual_fps(self) -> float:
        """Calculate actual FPS from buffered frame timestamps.

        Uses the same approach as the Seek SDK recording example:
        actual_fps = frame_count / elapsed_time
        """
        if len(self._timestamp_buffer) >= 2:
            elapsed = self._timestamp_buffer[-1] - self._timestamp_buffer[0]
            if elapsed > 0:
                actual_fps = (len(self._timestamp_buffer) - 1) / elapsed
                return actual_fps

        # Fallback to configured FPS if not enough data
        return self.config.fps

    def _start_recording(self):
        """Start recording: flush pre-buffer into new video files, then record live."""
        if self._frame_size is None:
            print("WARNING: Cannot start recording - no frames received yet")
            return

        # Calculate actual FPS from buffered timestamps
        actual_fps = self._calculate_actual_fps()

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = self.config.save_directory / f"raw_{timestamp_str}.avi"
        overlay_path = self.config.save_directory / f"overlay_{timestamp_str}.avi"

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        w, h = self._frame_size

        self._raw_writer = cv2.VideoWriter(str(raw_path), fourcc, actual_fps, (w, h))
        self._overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, actual_fps, (w, h))

        if not self._raw_writer.isOpened() or not self._overlay_writer.isOpened():
            print("ERROR: Failed to open VideoWriter")
            self._raw_writer = None
            self._overlay_writer = None
            return

        # Flush pre-buffer into the video files
        buffered_count = len(self._raw_buffer)
        for raw, overlay in zip(self._raw_buffer, self._overlay_buffer):
            self._write_frame(raw, overlay)

        # Clear buffers after flushing
        self._raw_buffer.clear()
        self._overlay_buffer.clear()
        self._timestamp_buffer.clear()

        self._recording = True
        self._record_frame_count = buffered_count
        self._record_start_time = time.time()

        # Start LED blinking (non-blocking)
        if self._led:
            self._led.blink(on_time=0.5, off_time=0.5)

        print(f"Recording started: {raw_path.name} "
              f"(pre-buffer: {buffered_count} frames, fps: {actual_fps:.1f})")

    def _stop_recording(self):
        """Stop recording and close video files."""
        self._recording = False

        # Calculate recording duration
        elapsed = time.time() - self._record_start_time
        total_frames = self._record_frame_count

        if self._raw_writer:
            self._raw_writer.release()
            self._raw_writer = None
        if self._overlay_writer:
            self._overlay_writer.release()
            self._overlay_writer = None

        # Stop LED
        if self._led:
            self._led.off()

        print(f"Recording stopped ({total_frames} frames, {elapsed:.1f}s)")

    def _write_frame(self, raw_frame: np.ndarray, overlay_frame: np.ndarray):
        """Write a single frame pair to video files. Converts RGB→BGR for OpenCV."""
        if self._raw_writer:
            self._raw_writer.write(cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR))
        if self._overlay_writer:
            self._overlay_writer.write(cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR))

    @property
    def is_recording(self) -> bool:
        """Whether recording is currently active"""
        return self._recording

    def cleanup(self):
        """Clean shutdown: stop recording, release GPIO"""
        if self._recording:
            self._stop_recording()

        if self._button:
            self._button.close()
            self._button = None
        if self._led:
            self._led.off()
            self._led.close()
            self._led = None

        print("VideoRecorder cleaned up")
