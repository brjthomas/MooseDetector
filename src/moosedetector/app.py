from moosedetector.thermalcamera import ThermalCamera
from moosedetector.pipeline import FramePipeline
import threading
import time


def main():
    """Main entry point for the Moose Detector application."""

    print("Starting Moose Detector...")

    # Create the frame processing pipeline
    pipeline = FramePipeline()

    # Event to signal shutdown
    stop_event = threading.Event()

    def processing_loop():
        """Processing thread - continuously processes latest frames"""
        print("Processing thread started")

        while not stop_event.is_set():
            # Get the latest frame (blocks until available or timeout)
            frame = pipeline.get_latest(timeout=0.5)

            if frame is not None:
                results = pipeline.process(frame) # Process the frame (YOLO inference)
                pipeline.visualize(results) # Visualize results
            else:
                # Timeout - no frame received
                # This is normal during startup or if camera disconnects
                pass

        pipeline.cleanup()
        print("Processing thread stopped")


    # Create and start the thermal camera with the frame callback
    # Camera callback - runs in SDK thread (must be fast!)
    # Just update the frame buffer, don't process here
    camera = ThermalCamera(pipeline.update)
    camera.start()

    # Start the processing thread
    processing_thread = threading.Thread(
        target=processing_loop,
        name="FrameProcessor",
        daemon=False  # Don't use daemon - we want clean shutdown
    )
    
    processing_thread.start()

    try:
        # Main thread just monitors and prints stats
        while True:
            time.sleep(5.0)

            # Print frame buffer statistics every 5 seconds
            stats = pipeline.get_stats()
            print(f"Stats: Received={stats['frames_received']}, "
                  f"Dropped={stats['frames_dropped']}, "
                  f"Drop Rate={stats['drop_rate']*100:.1f}%")

    except KeyboardInterrupt:
        
        print("\nShutting down...")

        # Stop the camera
        camera.stop()

        # Signal processing thread to stop
        stop_event.set()

        # Wait for processing thread to finish (with timeout)
        processing_thread.join(timeout=2.0)

        if processing_thread.is_alive():
            print("Warning: Processing thread did not stop cleanly")


        # Print final statistics
        stats = pipeline.get_stats()
        print(f"\nFinal Stats:")
        print(f"  Total frames received: {stats['frames_received']}")
        print(f"  Total frames dropped: {stats['frames_dropped']}")
        print(f"  Drop rate: {stats['drop_rate']*100:.2f}%")


if __name__ == "__main__":
    main()
