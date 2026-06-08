import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# OpenCV patternSize = (circles per row, number of rows)
PATTERN_SIZE = (7, 3)


@dataclass
class CircleGridFollowConfig:
    base_speed: float = 0.25
    steer_gain: float = 0.45

    # Apparent grid width at the desired follow distance (tune on the bot)
    target_width_px: float = 110.0
    width_gain: float = 0.003
    width_deadband_px: float = 10.0
    max_speed: float = 0.40


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _blob_detector() -> cv2.SimpleBlobDetector:
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 15000
    params.filterByCircularity = True
    params.minCircularity = 0.30
    params.filterByColor = True
    params.blobColor = 0  # dark dots on white plate
    return cv2.SimpleBlobDetector_create(params)


def _detect_circle_grid(
    frame_bgr: np.ndarray,
) -> Tuple[bool, Optional[Tuple[float, float]], Optional[float]]:
    """
    Detect the 7x3 dot grid on the lead robot.

    Returns (found, center_xy, width_px).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    flags = cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
    found, centers = cv2.findCirclesGrid(
        gray,
        PATTERN_SIZE,
        flags=flags,
        blobDetector=_blob_detector(),
    )
    if not found or centers is None or len(centers) == 0:
        return False, None, None

    cx = float(np.mean(centers[:, 0, 0]))
    cy = float(np.mean(centers[:, 0, 1]))
    width_px = float(np.max(centers[:, 0, 0]) - np.min(centers[:, 0, 0]))
    return True, (cx, cy), width_px


def _braitenberg_commands(
    cx: float,
    width_px: float,
    frame_width: int,
    cfg: CircleGridFollowConfig,
) -> Tuple[float, float]:
    """Steer toward the grid centre and adjust speed to hold a gap."""
    # Pattern right of centre -> speed up left wheel -> turn right into it.
    x_norm = (cx - frame_width * 0.5) / (frame_width * 0.5)
    turn = _clamp(cfg.steer_gain * x_norm, -0.5, 0.5)

    width_err = cfg.target_width_px - width_px
    if abs(width_err) <= cfg.width_deadband_px:
        v = cfg.base_speed
    else:
        v = cfg.base_speed + cfg.width_gain * width_err
    v = _clamp(v, 0.0, cfg.max_speed)

    left = _clamp(v + turn, -1.0, 1.0)
    right = _clamp(v - turn, -1.0, 1.0)
    return left, right


def main(camera, wheels, leds, stop_event):
    cfg = CircleGridFollowConfig()

    wheels.set_wheels_speed(0.0, 0.0)

    if leds:
        leds.set_rgb(0, [0.0, 0.0, 1.0])
        leds.set_rgb(2, [0.0, 0.0, 1.0])

    try:
        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            found, center, width_px = _detect_circle_grid(frame)

            if not found or center is None or width_px is None or width_px <= 1.0:
                wheels.set_wheels_speed(0.0, 0.0)
                time.sleep(0.02)
                continue

            cx, _cy = center
            left, right = _braitenberg_commands(cx, width_px, frame.shape[1], cfg)
            wheels.set_wheels_speed(left, right)
            time.sleep(0.02)
    finally:
        wheels.set_wheels_speed(0.0, 0.0)
        if leds:
            leds.all_off()
