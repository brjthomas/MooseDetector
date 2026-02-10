# Handles data flow. Frame -> pre processing -> inference -> post processing -> output

import numpy as np 
from ultralytics import YOLO
import cv2

class FramePipeline:
    def __init__(self):
        self._frame_count = 0
        self.model = YOLO("/home/moose/projects/MooseDetector/models/yolo26_best_v1.pt")

    def process(self, frame):
        self._frame_count += 1

        img = np.array(frame.data)
        # Convert ARGB -> RGB
        img_rgb = img[:, :, 1:4]

        # Run YOLO inference 
        results = self.model(img_rgb)
        
        # Visualize results
        self.visualize(img_rgb, results)

    def visualize(self, img, results):
        # results[0].boxes contains the bounding boxes
        # results[0].boxes.xyxy is a NumPy array: [x1, y1, x2, y2]
        img_copy = img.copy()
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