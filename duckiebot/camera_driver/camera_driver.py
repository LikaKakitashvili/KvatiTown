import cv2
import numpy as np
import os
import yaml
from typing import Tuple, Optional
from .camera_driver_abs import CameraDriverAbs


class CameraDriver(CameraDriverAbs):
    def _build_gstreamer_pipeline(self) -> str:
        # nvarguscamerasrc only outputs native sensor modes (1280x720, 1640x1232, etc.)
        # Requesting a non-native size (e.g. 640x480) in the source caps causes caps
        # negotiation failure. Let the source pick a native mode by framerate, then
        # scale to the desired output size via nvvidconv.
        pipeline = (
          f"nvarguscamerasrc sensor-id={int(self.sensor_mode)} ! "
          f"video/x-raw(memory:NVMM), format=NV12, framerate={self.framerate}/1 ! "
          f"nvvidconv ! "
          f"video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
          f"videoconvert ! "
          f"appsink drop=true sync=false"
        )


        return pipeline



    def _initialize_camera(self):
        import time

        pipeline = self._build_gstreamer_pipeline()
        print(f"[JetsonCamera] Pipeline: {pipeline}")

        cap = None
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                raise RuntimeError("Failed to open camera (isOpened returned False)")

            # isOpened() can return True even when GStreamer fails internally.
            warmup_attempts = 15
            for attempt in range(warmup_attempts):
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._device = cap
                    print(f"[JetsonCamera] First frame after {attempt + 1} attempt(s)")
                    break
                time.sleep(0.4)
            else:
                raise RuntimeError(
                    "Camera opened but returned no frames — "
                    "try: sudo systemctl restart nvargus-daemon"
                )
        except Exception:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self._device = None
            raise

        actual_w = int(self._device.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._device.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _capture_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._device is None:
            return False, None

        ret, frame = self._device.read()
        return (True, frame) if ret else (False, None)

    def _release_camera(self):
        if self._device is not None:
            self._device.release()
            self._device = None
