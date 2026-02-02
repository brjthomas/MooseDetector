# Handles data flow. Frame -> pre processing -> inference -> post processing -> output

import numpy as np 
from ultralytics import YOLO

class FramePipeline:
    def __init__(self):
        self._frame_count = 0

    def process(self, frame):
        self._frame_count += 1
        img = np.array(frame.data)
        img_rgb = img[:, :, 1:4]  # Keep R,G,B channels
        model = YOLO("/home/moose/projects/MooseDetector/models/yolo26_best_v1.pt")
        results = model(img_rgb)
        print(results)
        print(f"Processed frame {self._frame_count} with shape {img.shape}")