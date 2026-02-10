"""Simple performance metrics using YOLO's built-in timing"""

import time
import csv
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Optional, Dict
from datetime import datetime
import cv2
import numpy as np

from moosedetector.config import MooseDetectorConfig


@dataclass
class FrameMetrics:
    """Metrics for a single processed frame"""
    timestamp: float
    frame_number: int
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    fps: float
    objects_detected: int
    alert_active: bool


class PerformanceMetrics:
    """Tracks FPS and frame metrics"""

    def __init__(self, config):
        # Use default config if none provided
        if config is None:
            config = MooseDetectorConfig()
        self.config = config
        self._frame_number = 0
        self._fps_smoothing_window = config.metrics.fps_smoothing_window
        self._frame_times = deque(maxlen=self._fps_smoothing_window)
        self._last_frame_time = None

    def record_frame(self, results, objects_detected: int, alert_active: bool) -> FrameMetrics:
        """Extract metrics from YOLO results and calculate FPS

        Args:
            results: YOLO results object (has .speed attribute)
            objects_detected: Number of objects detected
            alert_active: Whether alert is currently active

        Returns:
            FrameMetrics dataclass
        """
        self._frame_number += 1
        current_time = time.time()

        # Calculate FPS from frame intervals
        if self._last_frame_time is not None:
            frame_interval = current_time - self._last_frame_time
            self._frame_times.append(frame_interval)

        self._last_frame_time = current_time

        # Calculate average FPS
        if len(self._frame_times) > 0:
            avg_interval = sum(self._frame_times) / len(self._frame_times)
            fps = 1.0 / avg_interval if avg_interval > 0 else 0.0
        else:
            fps = 0.0

        # Extract timing from YOLO results
        # results[0].speed is a dict with 'preprocess', 'inference', 'postprocess' keys (in ms)
        speed = results[0].speed

        return FrameMetrics(
            timestamp=current_time,
            frame_number=self._frame_number,
            preprocess_ms=speed['preprocess'],
            inference_ms=speed['inference'],
            postprocess_ms=speed['postprocess'],
            fps=fps,
            objects_detected=objects_detected,
            alert_active=alert_active
        )


class MetricsLogger:
    """Logs metrics to CSV file and terminal"""

    def __init__(self, config):
        # Use default config if none provided
        if config is None:
            config = MooseDetectorConfig()
        self.config = config

        self.log_file = config.metrics.log_file
        self.terminal_logging = config.metrics.terminal_logging
        self.terminal_log_interval = config.metrics.terminal_log_interval

        self._csv_file = None
        self._csv_writer = None
        self._frames_since_terminal_log = 0

        self._open_csv()

    def _open_csv(self):
        """Open CSV file with timestamp in filename and write header"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create timestamped filename
        # If log_file is "logs/metrics.csv", create "logs/metrics_20250210_143022.csv"
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = self.log_file.stem  # "metrics"
        suffix = self.log_file.suffix  # ".csv"
        timestamped_name = f"{stem}_{timestamp_str}{suffix}"
        timestamped_path = self.log_file.parent / timestamped_name

        # Store the actual path being used
        self.actual_log_file = timestamped_path

        self._csv_file = open(timestamped_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)

        # Write header
        self._csv_writer.writerow([
            'timestamp', 'frame_number', 'preprocess_ms', 'inference_ms',
            'postprocess_ms', 'fps', 'objects_detected', 'alert_active'
        ])
        self._csv_file.flush()

        print(f"Logging metrics to: {timestamped_path}")

    def log(self, metrics: FrameMetrics):
        """Log metrics to CSV and optionally to terminal"""
        # Write to CSV
        if self._csv_writer is not None:
            self._csv_writer.writerow([
                f"{metrics.timestamp:.3f}",
                metrics.frame_number,
                f"{metrics.preprocess_ms:.1f}",
                f"{metrics.inference_ms:.1f}",
                f"{metrics.postprocess_ms:.1f}",
                f"{metrics.fps:.1f}",
                metrics.objects_detected,
                metrics.alert_active
            ])
            self._csv_file.flush()

        # Terminal logging (every N frames)
        self._frames_since_terminal_log += 1
        if self.terminal_logging and self._frames_since_terminal_log >= self.terminal_log_interval:
            self._frames_since_terminal_log = 0
            self._print_terminal_log(metrics)

    def _print_terminal_log(self, metrics: FrameMetrics):
        """Print formatted metrics to terminal"""
        alert_status = "🚨 ALERT" if metrics.alert_active else "✓ Clear"
        total_ms = metrics.preprocess_ms + metrics.inference_ms + metrics.postprocess_ms
        print(f"[Frame {metrics.frame_number:>5}] "
              f"FPS: {metrics.fps:>5.1f} | "
              f"Inference: {metrics.inference_ms:>5.1f}ms | "
              f"Total: {total_ms:>5.1f}ms | "
              f"Objects: {metrics.objects_detected} | "
              f"{alert_status}")

    def close(self):
        """Close CSV file"""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None


class MetricsOverlay:
    """Draws simple metrics overlay on frames"""

    @staticmethod
    def draw(img: np.ndarray, metrics: FrameMetrics,
             buffer_stats: Optional[Dict] = None) -> np.ndarray:
        """Draw timestamp and key metrics on image

        Args:
            img: Input image (RGB)
            metrics: FrameMetrics to display
            buffer_stats: Optional frame buffer statistics

        Returns:
            Image with overlay
        """
        img_with_overlay = img.copy()

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (0, 255, 0)  # Green
        bg_color = (0, 0, 0)  # Black background

        # Helper to draw text with background
        def draw_text_with_bg(text: str, position: tuple, text_color=color):
            x, y = position
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            # Draw background rectangle
            cv2.rectangle(img_with_overlay,
                         (x - 2, y - text_height - 2),
                         (x + text_width + 2, y + baseline + 2),
                         bg_color, -1)
            # Draw text
            cv2.putText(img_with_overlay, text, (x, y),
                       font, font_scale, text_color, thickness, cv2.LINE_AA)
            return y + text_height + baseline + 10

        # Draw metrics (top-left corner)
        y_pos = 20
        draw_text_with_bg(f"FPS: {metrics.fps:.1f}, Objects: {metrics.objects_detected}", (10, y_pos))

        # Alert status (top-right corner if active)
        if metrics.alert_active:
            h, w = img_with_overlay.shape[:2]
            alert_text = "ALERT"
            (text_width, text_height), baseline = cv2.getTextSize(alert_text, font, 0.7, 2)
            cv2.putText(img_with_overlay, alert_text,
                       (w - text_width - 20, 30),
                       font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        return img_with_overlay
