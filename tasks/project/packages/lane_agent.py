import os
import time
import yaml
import numpy as np
import cv2
from collections import deque
from typing import Tuple

from tasks.project.packages import visual_servoing_activity as student
from tasks.project.packages.cuvrve_behavior import detect_curve

_CONFIG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "config", "lane_servoing_config.yaml"
))

_LINE_OFFSET = 160
_ROI_START   = 0.45
_ROI_SPAN    = 0.45
_NUM_SLICES  = 4
_SLICE_TOL   = 5


def detect_lines_in_slices(mask_yellow, mask_white, h):
    # type: (np.ndarray, np.ndarray, int) -> Tuple[list, list]
    slice_height = int(h * _ROI_SPAN / _NUM_SLICES)
    start_y      = int(h * _ROI_START)
    yellow_xs, white_xs = [], []

    for i in range(_NUM_SLICES):
        y = start_y + i * slice_height + slice_height // 2

        strip_y = mask_yellow[max(0, y - _SLICE_TOL): min(h, y + _SLICE_TOL), :]
        idx = np.where(strip_y > 0)[1]
        if len(idx) > 0:
            yellow_xs.append(int(np.mean(idx)))

        strip_w = mask_white[max(0, y - _SLICE_TOL): min(h, y + _SLICE_TOL), :]
        idx = np.where(strip_w > 0)[1]
        if len(idx) > 0:
            white_xs.append(int(np.mean(idx)))

    return yellow_xs, white_xs


class LaneServoingAgent:

    def __init__(self, config_path=None):
        path = config_path or _CONFIG_FILE
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        self.p_gain              = cfg.get("p_gain",              0.20)
        self.d_gain              = cfg.get("d_gain",              0.11)
        self.max_steer           = cfg.get("max_steer",           0.22)
        self.base_speed          = cfg.get("base_speed",          0.25)
        self.curve_speed         = cfg.get("curve_speed",         0.15)
        self.curve_threshold     = cfg.get("curve_threshold",     350)
        self.steering_threshold  = cfg.get("steering_threshold",  0.2)
        self.curve_boost         = cfg.get("curve_boost",         1.0)
        self.detection_threshold = cfg.get("detection_threshold", 30)

        self.frame_count         = 0
        self._prev_error         = 0.0
        self._prev_diff          = 0.0
        self._filtered_error     = 0.0
        self._filtered_steering  = 0.0
        self._lane_half_width    = float(_LINE_OFFSET)
        self._left_history       = deque(maxlen=3)
        self._right_history      = deque(maxlen=3)
        self.last_debug_info     = self._empty_debug_info(480, 640)

        # Temporary lateral bias (normalized): negative = steer left, away from white line.
        self._lateral_bias       = 0.0
        self._bias_until         = 0.0

    def apply_post_stop_bias(self, bias=-0.12, duration_s=4.0):
        # type: (float, float) -> None
        """After a stop sign, nudge the bot left so it resumes with more clearance from white."""
        self._lateral_bias = float(np.clip(bias, -0.35, 0.35))
        self._bias_until   = time.monotonic() + max(0.0, float(duration_s))

    def _active_lateral_bias(self):
        # type: () -> float
        if time.monotonic() >= self._bias_until:
            self._lateral_bias = 0.0
            return 0.0
        return self._lateral_bias

    def _calculate_error(self, yellow_xs, white_xs, left_det, right_det, w):
        if left_det and right_det and yellow_xs and white_xs:
            y_mean = float(np.mean(yellow_xs))
            w_mean = float(np.mean(white_xs))
            if w_mean > y_mean + 20:
                measured = (w_mean - y_mean) / 2.0
                if 40 < measured < 350:
                    self._lane_half_width = 0.9 * self._lane_half_width + 0.1 * measured
                error = w / 2.0 - (y_mean + w_mean) / 2.0
            else:
                error = w / 2.0 - (y_mean + self._lane_half_width)

        elif left_det and yellow_xs:
            error = w / 2.0 - (float(np.mean(yellow_xs)) + self._lane_half_width)

        elif right_det and white_xs:
            error = w / 2.0 - (float(np.mean(white_xs)) - self._lane_half_width)

        else:
            error = self._prev_error * (w / 2.0)

        return float(np.clip(error / (w / 2.0), -1.0, 1.0))

    def _calculate_steering(self, error):
        # type: (float) -> float
        raw_diff        = error - self._prev_error
        error_diff      = 0.7 * self._prev_diff + 0.3 * raw_diff
        self._prev_diff = error_diff
        self._prev_error = error

        raw_steering = self.p_gain * error + self.d_gain * error_diff
        raw_steering = float(np.clip(raw_steering, -self.max_steer, self.max_steer))

        # Rate-limit steering to prevent sudden flicker turns
        max_delta = 0.04
        delta = float(np.clip(raw_steering - self._filtered_steering, -max_delta, max_delta))
        self._filtered_steering += delta
        self._filtered_steering = float(np.clip(self._filtered_steering, -self.max_steer, self.max_steer))
        return self._filtered_steering

    def _motor_commands(self, steering, lane_detected, is_curve, both_visible):
        # type: (float, bool, bool, bool) -> Tuple[float, float]
        speed = self.curve_speed if is_curve else self.base_speed

        if not both_visible:
            speed *= 0.8

        # Slow down when steering hard
        speed *= max(0.6, 1.0 - abs(steering) * 1.5)
        speed  = max(speed, 0.08)   # always keep moving — never return 0

        left  = speed - steering
        right = speed + steering

        # Clamp: let one wheel go to 0 on sharp turns but never negative
        left  = float(np.clip(left,  0.0, 0.5))
        right = float(np.clip(right, 0.0, 0.5))

        # Keep at least one wheel spinning
        if left < 0.05 and right < 0.05:
            left  = 0.08
            right = 0.08

        return left, right

    def _smooth(self, left, right, both_visible):
        # type: (float, float, bool) -> Tuple[float, float]
        buf = 3 if both_visible else 2
        if self._left_history.maxlen != buf:
            self._left_history  = deque(maxlen=buf)
            self._right_history = deque(maxlen=buf)
        self._left_history.append(left)
        self._right_history.append(right)
        return (
            sum(self._left_history)  / len(self._left_history),
            sum(self._right_history) / len(self._right_history),
        )

    def compute_commands(self, image, bgr_input=False):
        # type: (np.ndarray, bool) -> Tuple[float, float]
        self.frame_count += 1
        bgr = image if bgr_input else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            mask_left, mask_right = student.detect_lane_markings(bgr)
        except Exception as e:
            print(f"[LaneAgent] detect_lane_markings error: {e}")
            # Detection failed — keep moving straight so robot doesn't freeze
            fwd = max(0.08, self.base_speed * 0.5)
            return fwd, fwd

        mask_y = (mask_left  * 255).astype(np.uint8)
        mask_w = (mask_right * 255).astype(np.uint8)

        yellow_pixels = int(np.count_nonzero(mask_y))
        white_pixels  = int(np.count_nonzero(mask_w))
        total_pixels  = yellow_pixels + white_pixels

        h, w = mask_y.shape

        yellow_xs, white_xs = detect_lines_in_slices(mask_y, mask_w, h)

        left_det     = len(yellow_xs) > 0
        right_det    = len(white_xs)  > 0
        lane_detected = total_pixels >= self.detection_threshold
        both_visible  = left_det and right_det

        is_curve, curve_dir = detect_curve(yellow_xs, white_xs, self.curve_threshold)

        raw_error            = self._calculate_error(yellow_xs, white_xs, left_det, right_det, w)
        self._filtered_error = 0.82 * self._filtered_error + 0.18 * raw_error

        biased_error = self._filtered_error + self._active_lateral_bias()
        biased_error = float(np.clip(biased_error, -1.0, 1.0))
        steering     = self._calculate_steering(biased_error)

        left, right = self._motor_commands(steering, lane_detected, is_curve, both_visible)
        left, right = self._smooth(left, right, both_visible)

        slice_height = int(h * _ROI_SPAN / _NUM_SLICES)
        start_y      = int(h * _ROI_START)
        combined     = np.clip(mask_left + mask_right, 0, 1)
        self.last_debug_info = {
            "roi":               bgr,
            "lane_mask":         (combined * 255).astype(np.uint8),
            "white_mask":        mask_w,
            "yellow_mask":       mask_y,
            "total_lane_pixels": total_pixels,
            "lateral_error":     float(np.clip(self._prev_error, -1.0, 1.0)),
            "lane_detected":     lane_detected,
            "frame_count":       self.frame_count,
            "yellow_xs":         yellow_xs,
            "white_xs":          white_xs,
            "slice_ys":          [start_y + i * slice_height + slice_height // 2 for i in range(_NUM_SLICES)],
            "is_curve":          is_curve,
            "curve_dir":         curve_dir,
        }

        if self.frame_count % 30 == 0:
            print(
                f"[LaneAgent] y={yellow_xs} w={white_xs} px={total_pixels} "
                f"err={self._filtered_error:.3f} steer={steering:.3f} "
                f"L={left:.3f} R={right:.3f}"
            )

        return left, right

    def step(self, image, wheels_driver):
        left, right = self.compute_commands(image)
        wheels_driver.set_wheels_speed(left, right)
        return left, right

    def reset(self):
        self.frame_count        = 0
        self._prev_error        = 0.0
        self._prev_diff         = 0.0
        self._filtered_error    = 0.0
        self._filtered_steering = 0.0
        self._lane_half_width   = float(_LINE_OFFSET)
        self._left_history      = deque(maxlen=3)
        self._right_history     = deque(maxlen=3)
        self._lateral_bias      = 0.0
        self._bias_until        = 0.0

    def get_debug_info(self, image):
        return self.last_debug_info

    def _empty_debug_info(self, h, w):
        return {
            "roi":               np.zeros((h, w, 3), dtype=np.uint8),
            "lane_mask":         np.zeros((h, w),    dtype=np.uint8),
            "white_mask":        np.zeros((h, w),    dtype=np.uint8),
            "yellow_mask":       np.zeros((h, w),    dtype=np.uint8),
            "total_lane_pixels": 0,
            "lateral_error":     0.0,
            "lane_detected":     False,
            "frame_count":       0,
        }
