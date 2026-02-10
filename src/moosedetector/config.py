"""Configuration dataclasses for MooseDetector"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DetectionConfig:
    """YOLO detection and tracking configuration"""
    model_path: Path = Path("/home/moose/projects/MooseDetector/models/yolo26_best_v1.pt")
    confidence_threshold: float = 0.5
    tracker_type: str = "botsort.yaml"  # BoT-SORT for better tracking across frames

    def __post_init__(self):
        """Validate configuration"""
        if not isinstance(self.model_path, Path):
            self.model_path = Path(self.model_path)
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"Confidence threshold must be between 0 and 1, got {self.confidence_threshold}")


@dataclass
class AlertConfig:
    """GPIO alert configuration"""
    gpio_pin: int = 17  # BCM numbering
    alert_classes: List[str] = field(default_factory=lambda: ["Animal", "person"])
    object_persistence_frames: int = 30  # Keep alert active for ~3.3s after object disappears (at 9 FPS)
    enabled: bool = True

    def __post_init__(self):
        """Validate configuration"""
        if self.gpio_pin < 0 or self.gpio_pin > 27:
            raise ValueError(f"GPIO pin must be between 0 and 27, got {self.gpio_pin}")
        if self.object_persistence_frames < 0:
            raise ValueError(f"Persistence frames must be non-negative, got {self.object_persistence_frames}")


@dataclass
class MetricsConfig:
    """Performance metrics and logging configuration"""
    log_file: Path = Path("/home/moose/projects/MooseDetector/logs/metrics.csv")
    terminal_logging: bool = True
    overlay_display: bool = True
    fps_smoothing_window: int = 10  # Number of frames to average for FPS calculation
    terminal_log_interval: int = 30  # Print to terminal every N frames

    def __post_init__(self):
        """Validate and create log directory"""
        if not isinstance(self.log_file, Path):
            self.log_file = Path(self.log_file)

        # Create log directory if it doesn't exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        if self.fps_smoothing_window < 1:
            raise ValueError(f"FPS smoothing window must be positive, got {self.fps_smoothing_window}")
        if self.terminal_log_interval < 1:
            raise ValueError(f"Terminal log interval must be positive, got {self.terminal_log_interval}")


@dataclass
class MooseDetectorConfig:
    """Main configuration container for MooseDetector"""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    display_window_name: str = "MooseDetector - Thermal Detection"

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MooseDetectorConfig':
        """Create config from dictionary (for future file-based config loading)"""
        return cls(
            detection=DetectionConfig(**config_dict.get('detection', {})),
            alert=AlertConfig(**config_dict.get('alert', {})),
            metrics=MetricsConfig(**config_dict.get('metrics', {})),
            display_window_name=config_dict.get('display_window_name', cls.display_window_name)
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary (for saving to file)"""
        return {
            'detection': {
                'model_path': str(self.detection.model_path),
                'confidence_threshold': self.detection.confidence_threshold,
                'tracker_type': self.detection.tracker_type,
            },
            'alert': {
                'gpio_pin': self.alert.gpio_pin,
                'alert_classes': self.alert.alert_classes,
                'object_persistence_frames': self.alert.object_persistence_frames,
                'enabled': self.alert.enabled,
            },
            'metrics': {
                'log_file': str(self.metrics.log_file),
                'terminal_logging': self.metrics.terminal_logging,
                'overlay_display': self.metrics.overlay_display,
                'fps_smoothing_window': self.metrics.fps_smoothing_window,
                'terminal_log_interval': self.metrics.metrics.terminal_log_interval,
            },
            'display_window_name': self.display_window_name,
        }


# Default configuration instance for convenience
DEFAULT_CONFIG = MooseDetectorConfig()
