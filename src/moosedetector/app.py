from moosedetector.thermalcamera import ThermalCamera
from moosedetector.pipeline import FramePipeline
from time import sleep

def main():
    """Main entry point for the Moose Detector application."""

    print("Starting Moose Detector...")

    # Create the frame processing pipeline.
    pipeline = FramePipeline()

    def handle_frame(frame):
        # Intermediate callback function that handles incoming frames.
        pipeline.process(frame)
 
    # Create and start the thermal camera with the frame callback.
    camera = ThermalCamera(handle_frame)
    camera.start()

    try:
        while True:
            sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")
        camera.stop()


if __name__ == "__main__":
    main()