# Handles data flow. Frame -> pre processing -> inference -> post processing -> output

import numpy as np 
from ultralytics import YOLO
import cv2
import threading

class FramePipeline:
    def __init__(self):
        # Frame buffer for thread-safe frame storage
        self._frame = None
        self._img_rgb = None
        self._lock = threading.Lock() 
        self._frame_available = threading.Event() 
        self._frames_dropped = 0
        self._frames_received = 0
        self._frame_count = 0  # For processing

        # Load YOLO model
        self.model = YOLO("/home/moose/projects/MooseDetector/models/yolo26_best_v1.pt")

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
        """Process a single frame with YOLO inference

        Args:
            frame: Frame object from thermal camera with .data attribute
        """
        self._frame_count += 1

        # Convert ARGB -> RGB
        self._img_rgb = self.bgra2rgb(frame.data)

        # Run YOLO inference
        results = self.model(self._img_rgb, verbose=False)

        return results

        

    def visualize(self, results):
        # results[0].boxes contains the bounding boxes
        # results[0].boxes.xyxy is a NumPy array: [x1, y1, x2, y2]
        img_copy = self._img_rgb.copy()
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Optionally, add labels
        if results[0].boxes.cls is not None:
            for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                x1, y1, _, _ = map(int, box)
                label = f"{self.model.names[int(cls)]}"
                cv2.putText(img_copy, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Show the image
        cv2.imshow("YOLO Detection", img_copy)
        cv2.waitKey(1)  # 1ms delay for live feed

    def cleanup(self):
        """Clean up resources (must be called from same thread as cv2.imshow)"""
        cv2.destroyWindow("YOLO Detection")
        print("Destroyed cv2 window")