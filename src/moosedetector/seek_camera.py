from seekcamera import (
    SeekCameraManager,
    SeekCameraIOType,
    SeekCameraManagerEvent,
    SeekCameraFrameFormat,
)

class ThermalCamera:
    def __init__(self, frame_callback):
        self._frame_callback = frame_callback
        self._manager = None
        self._camera = None
    
    def on_frame(camera, camera_frame, _user_data):
        # Internal SDK Async callback fired whenever a new frame is available.
        if self._frame_callback:
            self._frame_callback(camera_frame.color_argb8888)

    def _on_event(self, camera, event_type, event_status, _user_data):
        # Internal SDK Async callback fired whenever a camera event occurs.
        if event_type == SeekCameraManagerEvent.CONNECT:
            self._camera = camera
            camera.register_frame_available_callback(self._on_frame)
            camera.capture_session_start(SeekCameraFrameFormat.COLOR_ARGB8888)

        elif event_type == SeekCameraManagerEvent.DISCONNECT:
            if self._camera:
                self._camera.capture_session_stop()
                self._camera = None

        elif event_type == SeekCameraManagerEvent.ERROR:
            raise RuntimeError(event_status)

    def start(self):
        self._manager = SeekCameraManager(SeekCameraIOType.USB)
        self._manager.register_event_callback(self._on_event, None)

    def stop(self):
        if self._camera:
            self._camera.capture_session_stop()

        if self._manager:
            self._manager.close()