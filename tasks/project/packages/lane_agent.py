import os
import yaml
import numpy as np
import cv2
from collections import deque
from typing import List, Optional, Tuple

from tasks.project.packages import visual_servoing_activity as student
from tasks.project.packages.cuvrve_behavior import detect_curve

_CONFIG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_config.yaml'
))

_LINE_OFFSET = 160
_ROI_START = 0.47
_ROI_SLICE_BAND = 0.35
_NUM_SLICES = 3
_SLICE_TOL = 5


def _line_x_in_strip(strip: np.ndarray) -> Optional[int]:
    if strip.size == 0:
        return None
    xs = np.where(strip > 0)[1]
    if len(xs) == 0:
        return None
    return int(np.median(xs))


def _interpolate_slice_xs(values: List[Optional[int]]) -> List[int]:
    """Fill gaps between dashes so near/far slices still form a curve estimate."""
    if not any(v is not None for v in values):
        return []
    filled = list(values)
    known = [i for i, v in enumerate(filled) if v is not None]
    if len(known) == 1:
        return [int(filled[known[0]])] * len(filled)
    for i, v in enumerate(filled):
        if v is not None:
            continue
        prev_i = max((k for k in known if k < i), default=None)
        next_i = min((k for k in known if k > i), default=None)
        if prev_i is not None and next_i is not None:
            t = (i - prev_i) / max(1, next_i - prev_i)
            filled[i] = int(filled[prev_i] + t * (filled[next_i] - filled[prev_i]))
        elif prev_i is not None:
            filled[i] = filled[prev_i]
        elif next_i is not None:
            filled[i] = filled[next_i]
    return [int(v) for v in filled]


def detect_lines_in_slices(
    mask_yellow: np.ndarray,
    mask_white: np.ndarray,
    h: int,
    roi_start: float = _ROI_START,
    roi_slice_band: float = _ROI_SLICE_BAND,
    num_slices: int = _NUM_SLICES,
    slice_tol: int = _SLICE_TOL,
) -> Tuple[list, list]:
    """Sample lane x-positions on horizontal slices; interpolate across dash gaps."""
    slice_height = int(h * roi_slice_band / num_slices)
    start_y = int(h * roi_start)
    tol = max(3, int(slice_tol))
    yellow_raw: List[Optional[int]] = []
    white_raw: List[Optional[int]] = []

    for i in range(num_slices):
        y = start_y + i * slice_height + slice_height // 2
        y0 = max(0, y - tol)
        y1 = min(h, y + tol + 1)

        yellow_raw.append(_line_x_in_strip(mask_yellow[y0:y1, :]))
        white_raw.append(_line_x_in_strip(mask_white[y0:y1, :]))

    return _interpolate_slice_xs(yellow_raw), _interpolate_slice_xs(white_raw)


class LaneServoingAgent:

    def __init__(self, config_path: str = None):
        path = config_path or _CONFIG_FILE
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        self.p_gain = cfg.get('p_gain', 0.1)
        self.d_gain = cfg.get('d_gain', 0.35)
        self.max_steer = cfg.get('max_steer', 0.4)
        self.base_speed = cfg.get('base_speed', 0.2)
        self.curve_speed = cfg.get('curve_speed', 0.2)
        self.curve_threshold = cfg.get('curve_threshold', 350)
        self.steering_threshold = cfg.get('steering_threshold', 0.2)
        self.curve_boost = cfg.get('curve_boost', 1.3)
        self.detection_threshold = cfg.get('detection_threshold', 500)

        self.roi_start = float(cfg.get('roi_start', _ROI_START))
        self.roi_slice_band = float(cfg.get('roi_slice_band', _ROI_SLICE_BAND))
        self.num_slices = max(1, int(cfg.get('num_slices', _NUM_SLICES)))
        self.slice_tol = max(3, int(cfg.get('slice_tol', _SLICE_TOL)))

        self.frame_count = 0
        self._prev_error = 0.0
        self._filtered_error = 0.0
        self._lane_half_width = float(_LINE_OFFSET)
        self._left_history = deque(maxlen=3)
        self._right_history = deque(maxlen=3)
        self.last_debug_info = self._empty_debug_info(480, 640)

    def _roi_bounds(self, h: int, w: int) -> Tuple[int, int, int, int]:
        y0 = int(h * self.roi_start)
        y1 = min(h, y0 + int(h * self.roi_slice_band))
        return y0, y1, 0, w

    def _slice_ys(self, h: int) -> list:
        slice_height = int(h * self.roi_slice_band / self.num_slices)
        start_y = int(h * self.roi_start)
        return [start_y + i * slice_height + slice_height // 2 for i in range(self.num_slices)]

    def _calculate_error(self, yellow_xs, white_xs, left_det, right_det, w):
        if left_det and right_det and yellow_xs and white_xs:
            y_mean = float(np.mean(yellow_xs))
            w_mean = float(np.mean(white_xs))
            measured = (w_mean - y_mean) / 2.0
            if measured > 20:
                self._lane_half_width = 0.9 * self._lane_half_width + 0.1 * measured
            error = w / 2.0 - (y_mean + w_mean) / 2.0
        elif left_det and yellow_xs:
            error = w / 2.0 - (float(np.mean(yellow_xs)) + self._lane_half_width)
        elif right_det and white_xs:
            error = w / 2.0 - (float(np.mean(white_xs)) - self._lane_half_width)
        else:
            error = self._prev_error

        return float(np.clip(error / (w / 2.0), -1.0, 1.0))

    def _calculate_steering(self, error: float) -> float:
        error_diff = error - self._prev_error
        self._prev_error = error
        steering = self.p_gain * error + self.d_gain * error_diff
        return float(np.clip(steering, -self.max_steer, self.max_steer))

    def _motor_commands(self, steering: float, recovery: bool, is_curve: bool, both_visible: bool):
        if recovery:
            return 0.0, 0.0

        speed = self.curve_speed if is_curve else self.base_speed
        if not both_visible:
            speed *= 0.8

        left = speed - steering
        right = speed + steering

        if is_curve and abs(steering) > self.steering_threshold:
            if steering > 0:
                right *= 5
            else:
                left *= self.curve_boost

        return float(np.clip(left, 0.0, 1.0)), float(np.clip(right, 0.0, 1.0))

    def _smooth(self, left, right, both_visible):
        buf = 2 if both_visible else 1
        if self._left_history.maxlen != buf:
            self._left_history = deque(maxlen=buf)
            self._right_history = deque(maxlen=buf)
        self._left_history.append(left)
        self._right_history.append(right)
        return (
            sum(self._left_history) / len(self._left_history),
            sum(self._right_history) / len(self._right_history),
        )

    def compute_commands(self, image: np.ndarray, bgr_input: bool = True) -> Tuple[float, float]:
        """BGR in by default (CameraDriver); same control flow as visual_lane_servoing agent."""
        self.frame_count += 1
        bgr = image if bgr_input else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            mask_left, mask_right = student.detect_lane_markings(bgr)
            yellow_color_mask, white_color_mask = student.get_color_masks(bgr)
        except Exception as e:
            print(f"[Agent] detect_lane_markings error: {e}")
            return 0.0, 0.0

        mask_y_edge = (mask_left * 255).astype(np.uint8)
        mask_w_edge = (mask_right * 255).astype(np.uint8)
        # Color masks are more stable in curves; edge masks add detail when available.
        mask_y = np.maximum(yellow_color_mask, mask_y_edge)
        mask_w = np.maximum(white_color_mask, mask_w_edge)

        yellow_pixels = int(np.count_nonzero(mask_y))
        white_pixels = int(np.count_nonzero(mask_w))
        total_pixels = yellow_pixels + white_pixels

        combined = np.clip(mask_left + mask_right, 0, 1)
        h, w = mask_y.shape
        y0, y1, x0, x1 = self._roi_bounds(h, w)

        self.last_debug_info = {
            'roi': bgr,
            'lane_mask': (combined * 255).astype(np.uint8),
            'white_mask': white_color_mask,
            'yellow_mask': yellow_color_mask,
            'total_lane_pixels': total_pixels,
            'lateral_error': float(np.clip(self._prev_error, -1.0, 1.0)),
            'lane_detected': total_pixels >= self.detection_threshold,
            'frame_count': self.frame_count,
            'roi_bounds': (y0, y1, x0, x1),
        }

        left_det = yellow_pixels > 0
        right_det = white_pixels > 0
        recovery = total_pixels < self.detection_threshold

        yellow_xs, white_xs = detect_lines_in_slices(
            mask_y,
            mask_w,
            h,
            roi_start=self.roi_start,
            roi_slice_band=self.roi_slice_band,
            num_slices=self.num_slices,
            slice_tol=self.slice_tol,
        )
        both_visible = left_det and right_det and not recovery
        is_curve, curve_dir = detect_curve(yellow_xs, white_xs, self.curve_threshold)

        raw_error = self._calculate_error(yellow_xs, white_xs, left_det, right_det, w)
        self._filtered_error = 0.7 * self._filtered_error + 0.3 * raw_error
        steering = self._calculate_steering(self._filtered_error)
        left, right = self._motor_commands(steering, recovery, is_curve, both_visible)
        left, right = self._smooth(left, right, both_visible)

        self.last_debug_info.update({
            'yellow_xs': yellow_xs,
            'white_xs': white_xs,
            'slice_ys': self._slice_ys(h),
            'is_curve': is_curve,
            'curve_dir': curve_dir,
        })

        return left, right

    def step(self, image: np.ndarray, wheels_driver) -> Tuple[float, float]:
        left, right = self.compute_commands(image)
        wheels_driver.set_wheels_speed(left, right)
        return left, right

    def get_debug_info(self, image: np.ndarray) -> dict:
        return self.last_debug_info

    def _empty_debug_info(self, h, w):
        return {
            'roi': np.zeros((h, w, 3), dtype=np.uint8),
            'lane_mask': np.zeros((h, w), dtype=np.uint8),
            'white_mask': np.zeros((h, w), dtype=np.uint8),
            'yellow_mask': np.zeros((h, w), dtype=np.uint8),
            'total_lane_pixels': 0,
            'lateral_error': 0.0,
            'lane_detected': False,
            'frame_count': 0,
        }
