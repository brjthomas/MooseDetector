# Handles data flow. Frame -> pre processing -> inference -> post processing -> output

import numpy as np
from ultralytics import YOLO
import cv2
import threading
from moosedetector.config import MooseDetectorConfig
from moosedetector.metrics import MetricsOverlay

class FramePipeline:
    
    def __init__(self, config: MooseDetectorConfig = None):
        # Use default config if none provided
        if config is None:
            config = MooseDetectorConfig()
        self.config = config

        # Frame buffer for thread-safe frame storage
        self._frame = None
        self._img_rgb = None
        self._lock = threading.Lock()
        self._frame_available = threading.Event()
        self._frames_dropped = 0
        self._frames_received = 0
        self._frame_count = 0  # For processing

        # Load YOLO model
        self.model = YOLO(str(config.detection.model_path))

    def update(self, frame):
        """Fast, non-blocking update from camera thread"""
        with self._lock:
            if self._frame is not None and not self._frame_available.is_set():
                self._frames_dropped += 1  # Previous frame wasn't consumed

            self._frame = frame
            self._frames_received += 1
            self._frame_available.set()

    def get_latest(self, timeout=0.5):
        """Blocking get from processing thread

        Args:
            timeout: Maximum seconds to wait for frame

        Returns:
            Latest frame or None if timeout
        """
        if not self._frame_available.wait(timeout=timeout):
            return None

        with self._lock:
            frame = self._frame
            self._frame_available.clear()

        return frame

    def get_stats(self):
        """Get frame buffer statistics for monitoring"""
        with self._lock:
            return {
                "frames_received": self._frames_received,
                "frames_dropped": self._frames_dropped,
                "drop_rate": self._frames_dropped / max(self._frames_received, 1)
            }
            
    def bgra2rgb(self, bgra):
        row, col, ch = bgra.shape

        assert ch == 4, "ARGB image has 4 channels."

        rgb = np.zeros((row, col, 3), dtype="uint8")
        # convert to rgb expected to generate the jpeg image
        rgb[:, :, 0] = bgra[:, :, 2]
        rgb[:, :, 1] = bgra[:, :, 1]
        rgb[:, :, 2] = bgra[:, :, 0]

        return rgb


    def process(self, frame):
        """Process a single frame with YOLO tracking

        Args:
            frame: Frame object from thermal camera with .data attribute

        Returns:
            Tuple of (results, detections_list)
            - results: YOLO results object
            - detections_list: List of detection dicts with track_id, class_name, confidence, bbox
        """
        self._frame_count += 1

        # Convert ARGB -> RGB
        self._img_rgb = self.bgra2rgb(frame.data)

        # Run YOLO tracking (not just detection)
        results = self.model.track(
            self._img_rgb,
            persist=True,  # Maintain tracking across frames
            tracker=self.config.detection.tracker_type,  # "botsort.yaml"
            conf=self.config.detection.confidence_threshold,  # 0.5
            verbose=False
        )

        # Extract structured detections with tracking info
        detections = self._extract_detections(results)

        return results, detections

    def _extract_detections(self, results):
        """Extract structured detection information with tracking IDs

        Args:
            results: YOLO tracking results

        Returns:
            List of detection dicts with keys:
                - track_id: int, unique object ID (persistent across frames)
                - class_name: str, detected class
                - confidence: float, detection confidence
                - bbox: array, bounding box [x1, y1, x2, y2]
        """
        detections = []
        result = results[0]

        # Check if we have detections
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # Extract tracking IDs if available (will be None on first frame)
        track_ids = result.boxes.id
        if track_ids is not None:
            track_ids = track_ids.cpu().numpy().astype(int)
        else:
            # No tracking IDs yet, use index as placeholder
            track_ids = list(range(len(result.boxes)))

        # Build detection list
        for i, box in enumerate(result.boxes.xyxy):
            detections.append({
                'track_id': int(track_ids[i]),
                'class_name': self.model.names[int(result.boxes.cls[i])],
                'confidence': float(result.boxes.conf[i]),
                'bbox': box.cpu().numpy()
            })

        return detections

    def visualize(self, results, frame_metrics=None):
        """Visualize detection results with tracking IDs and optional metrics overlay

        Args:
            results: YOLO results object
            frame_metrics: Optional FrameMetrics to display as overlay
        """
        img_copy = self._img_rgb.copy()
        result = results[0]

        # Get tracking IDs if available
        track_ids = result.boxes.id
        if track_ids is not None:
            track_ids = track_ids.cpu().numpy().astype(int)
        else:
            track_ids = None

        # Draw bounding boxes with track IDs and class labels
        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)

            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Create label with class name and track ID
            if result.boxes.cls is not None:
                class_name = self.model.names[int(result.boxes.cls[i])]
                confidence = float(result.boxes.conf[i])

                if track_ids is not None:
                    # Show track ID if available
                    label = f"ID:{track_ids[i]} {class_name} {confidence:.2f}"
                else:
                    label = f"{class_name} {confidence:.2f}"

                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(img_copy, (x1, y1 - label_height - 5),
                            (x1 + label_width, y1), (0, 255, 0), -1)

                # Draw label text
                cv2.putText(img_copy, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Add metrics overlay if provided
        if frame_metrics is not None and self.config.metrics.overlay_display:
            buffer_stats = self.get_stats()
            img_copy = MetricsOverlay.draw(img_copy, frame_metrics, buffer_stats)

        # Show the image
        cv2.imshow(self.config.display_window_name, img_copy)
        cv2.waitKey(1)  # 1ms delay for live feed

    def cleanup(self):
        """Clean up resources (must be called from same thread as cv2.imshow)"""
        cv2.destroyWindow(self.config.display_window_name)
        print("Destroyed cv2 window")