"""GPIO alert management with object persistence and tracking"""

from gpiozero import OutputDevice
import atexit
from typing import List, Dict
from moosedetector.config import AlertConfig



class AlertManager:
    """Manages GPIO alerts based on tracked object detections

    Features:
    - Drives GPIO pin HIGH when alert-worthy objects detected
    - Object persistence: keeps alert active for N frames after object disappears
    - Class filtering: only specified classes trigger alerts
    - Graceful cleanup with atexit
    """

    def __init__(self, config: AlertConfig):
        """Initialize alert manager

        Args:
            config: AlertConfig with GPIO pin, alert classes, persistence settings
        """
        self.config = config
        self._persistence_counters = {}  # track_id -> remaining_frames
        self._gpio = None
        self._alert_active = False

        # Initialize GPIO if enabled
        if config.enabled:
            try:
                self._gpio = OutputDevice(config.gpio_pin, initial_value=False)
                print(f"✓ GPIO alert system initialized on pin {config.gpio_pin}")

                # Register cleanup to ensure GPIO is released on exit
                atexit.register(self.cleanup)
            except Exception as e:
                print(f"⚠ WARNING: GPIO initialization failed: {e}")
                print("  Running without GPIO alerts")
                self._gpio = None
        else:
            print("ℹ GPIO alerts disabled in config")

    def update(self, detections: List[Dict]):
        """Update alert state based on current frame detections

        Args:
            detections: List of detection dicts with keys:
                - 'track_id': int, unique object ID
                - 'class_name': str, detected class
                - 'confidence': float, detection confidence
                - 'bbox': array, bounding box coordinates
        """
        # Find alert-worthy objects in current frame
        # (objects that match alert classes and meet confidence threshold)
        current_ids = set()
        for det in detections:
            if (det['class_name'] in self.config.alert_classes and
                det['confidence'] >= 0.5):  # Using fixed threshold for now
                current_ids.add(det['track_id'])

        # Update persistence counters
        # Reset counter for objects currently visible
        for track_id in current_ids:
            self._persistence_counters[track_id] = self.config.object_persistence_frames

        # Decrement counters for objects not currently visible
        for track_id in list(self._persistence_counters.keys()):
            if track_id not in current_ids:
                self._persistence_counters[track_id] -= 1
                # Remove if counter expired
                if self._persistence_counters[track_id] <= 0:
                    del self._persistence_counters[track_id]

        # Determine alert state: active if any tracked objects remain
        should_alert = len(self._persistence_counters) > 0

        # Update GPIO state
        if self._gpio is not None:
            self._gpio.value = should_alert

        self._alert_active = should_alert

    def get_alert_state(self) -> bool:
        """Get current alert state

        Returns:
            True if alert is active, False otherwise
        """
        return self._alert_active

    def get_tracked_objects(self) -> List[int]:
        """Get list of currently tracked object IDs

        Returns:
            List of track IDs that are maintaining the alert
        """
        return list(self._persistence_counters.keys())

    def cleanup(self):
        """Clean up GPIO resources

        CRITICAL: Must be called before program exit to release GPIO pin.
        Without cleanup, next run may fail with "resource busy" errors.
        """
        if self._gpio is not None:
            self._gpio.off()  # Ensure pin is LOW
            self._gpio.close()  # Release GPIO resource
            self._gpio = None
            print("✓ GPIO alert system cleaned up")
