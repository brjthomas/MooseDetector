import numpy as np 

class FramePipeline:
    def __init__(self):
        self._frame_count = 0

    def process(self, frame):
        self._frame_count += 1
        img = np.array(frame.data)
        print(f"Processed frame {self._frame_count} with shape {img.shape}")