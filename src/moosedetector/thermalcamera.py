# The hardware boundary and should be the only file that knows seek SDK details.

from seekcamera import (
    SeekCameraManager,
    SeekCameraIOType,
    SeekCameraManagerEvent,
    SeekCameraFrameFormat
)

# ThermalCamera encapsulates the Seek Thermal camera functionality. 
class ThermalCamera:
    def __init__(self, frame_callback):
        self._frame_callback = frame_callback
        self._manager = None
        self._camera = None
    
    def _on_frame_impl(self, camera_frame):
        # Internal SDK Async callback fired whenever a new frame is available.

        #Make sure we specified a frame callback
        if self._frame_callback is None:
            return

        # Access frame data safely
        frame = camera_frame.color_argb8888

        # Call the frame callback in the FramePipeline class in pipeline.py 
        self._frame_callback(frame)

    def _on_event(self, camera, event_type, event_status, _user_data):
        # Internal SDK Async callback fired whenever a camera event occurs.
        if event_type == SeekCameraManagerEvent.CONNECT:
            print("Seek SDK Camera Connect Event")
            self._camera = camera

            # Cannot invoke the frame callback _on_frame_impl directly here
            # because the SDK requires the callback to be a free function.
            # So we define a wrapper function to call the instance method.
            def _on_frame(camera, camera_frame, _user_data):
                self._on_frame_impl(camera_frame)

            camera.register_frame_available_callback(_on_frame)
            # Start the capture session to receive frames with the specified format.
            camera.capture_session_start(SeekCameraFrameFormat.COLOR_ARGB8888)

        elif event_type == SeekCameraManagerEvent.DISCONNECT:
            if self._camera:
                self._camera.capture_session_stop()
                self._camera = None

        elif event_type == SeekCameraManagerEvent.ERROR:
            raise RuntimeError(event_status)

    def start(self):
        # Initialize the Seek Camera Manager and register the high level event callback.
        self._manager = SeekCameraManager(SeekCameraIOType.USB)
        self._manager.register_event_callback(self._on_event, None)

    def stop(self):
        if self._camera:
            self._camera.capture_session_stop()

        # Important to destroy the manager to clean up resources. Make get seek communication error otherwise.
        if self._manager:
            self._manager.destroy()