import os
import time
from collections import deque
from typing import List, Set, Tuple

import cv2
import numpy as np
import yaml

_CONFIG_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "lane_servoing_config.yaml")
)
_HSV_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "lane_servoing_hsv_config.yaml")
)

_LINE_OFFSET = 160
_ROI_START = 0.47
_NUM_SLICES = 3
_SLICE_TOL = 5

# Requested project tuning
BASE_SPEED = 0.31
CURVE_SPEED = 0.25
P_GAIN = 0.16
D_GAIN = 0.79
DETECTION_THRESHOLD = 100
DT = 0.02
STOP_TAG_IDS: Set[int] = {20, 21, 22, 23, 24, 25, 26, 27, 28}
SLOW_TAG_IDS: Set[int] = {39}
STOP_RAMP_SECONDS = 1.2
STOP_HOLD_SECONDS = 2.0
SLOW_SECONDS = 4.5
SLOW_SPEED_MULTIPLIER = 0.5
SIGN_COOLDOWN_SECONDS = 1.0

try:
    with open(_HSV_FILE) as _f:
        _h = yaml.safe_load(_f) or {}
except Exception:
    _h = {}

_yellow_lower = np.array([_h.get("yellow_lower_h", 0), _h.get("yellow_lower_s", 0), _h.get("yellow_lower_v", 0)])
_yellow_upper = np.array([_h.get("yellow_upper_h", 0), _h.get("yellow_upper_s", 0), _h.get("yellow_upper_v", 0)])
_white_lower = np.array([_h.get("white_lower_h", 0), _h.get("white_lower_s", 0), _h.get("white_lower_v", 0)])
_white_upper = np.array([_h.get("white_upper_h", 0), _h.get("white_upper_s", 0), _h.get("white_upper_v", 0)])


def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper
    _yellow_lower = np.array(yellow_lower)
    _yellow_upper = np.array(yellow_upper)
    _white_lower = np.array(white_lower)
    _white_upper = np.array(white_upper)


def get_hsv_bounds():
    return {
        "yellow_lower_h": int(_yellow_lower[0]),
        "yellow_upper_h": int(_yellow_upper[0]),
        "yellow_lower_s": int(_yellow_lower[1]),
        "yellow_upper_s": int(_yellow_upper[1]),
        "yellow_lower_v": int(_yellow_lower[2]),
        "yellow_upper_v": int(_yellow_upper[2]),
        "white_lower_h": int(_white_lower[0]),
        "white_upper_h": int(_white_upper[0]),
        "white_lower_s": int(_white_lower[1]),
        "white_upper_s": int(_white_upper[1]),
        "white_lower_v": int(_white_lower[2]),
        "white_upper_v": int(_white_upper[2]),
    }


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_gaussian_filter = cv2.GaussianBlur(img_gray, (0, 0), 1.0)
    sobelx = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 0, 1)
    gmag = np.sqrt(sobelx**2 + sobely**2)
    mask_mag = gmag > 40

    width = image.shape[1]
    mask_left = np.ones(img_gray.shape, dtype=np.uint8)
    mask_left[:, int(np.floor(width / 2)) : width + 1] = 0
    mask_right = np.ones(img_gray.shape, dtype=np.uint8)
    mask_right[:, 0 : int(np.floor(width / 2))] = 0

    mask_sobelx_pos = sobelx > 0
    mask_sobelx_neg = sobelx < 0
    mask_sobely_neg = sobely < 0

    mask_yellow = cv2.inRange(img_hsv, _yellow_lower, _yellow_upper) > 0
    mask_white = cv2.inRange(img_hsv, _white_lower, _white_upper) > 0

    mask_left_edge = mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white
    return mask_left_edge.astype(np.float32), mask_right_edge.astype(np.float32)


def detect_curve(yellow_xs: List[int], white_xs: List[int], curve_threshold: int = 350) -> Tuple[bool, int]:
    shift = 0
    if len(yellow_xs) >= 2:
        shift = yellow_xs[-1] - yellow_xs[0]
    elif len(white_xs) >= 2:
        shift = white_xs[-1] - white_xs[0]
    if abs(shift) > curve_threshold:
        return True, int(np.sign(shift))
    return False, 0


def detect_lines_in_slices(mask_yellow: np.ndarray, mask_white: np.ndarray, h: int) -> Tuple[list, list]:
    slice_height = int(h * 0.35 / _NUM_SLICES)
    start_y = int(h * _ROI_START)
    yellow_xs, white_xs = [], []
    for i in range(_NUM_SLICES):
        y = start_y + i * slice_height + slice_height // 2
        strip_y = mask_yellow[y - _SLICE_TOL : y + _SLICE_TOL, :]
        idx = np.where(strip_y > 0)[1]
        if len(idx) > 0:
            yellow_xs.append(int(np.mean(idx)))
        strip_w = mask_white[y - _SLICE_TOL : y + _SLICE_TOL, :]
        idx = np.where(strip_w > 0)[1]
        if len(idx) > 0:
            white_xs.append(int(np.mean(idx)))
    return yellow_xs, white_xs


class LaneServoingAgent:
    def __init__(self, config_path: str = None):
        path = config_path or _CONFIG_FILE
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        self.p_gain = cfg.get("p_gain", 0.1)
        self.d_gain = cfg.get("d_gain", 0.35)
        self.max_steer = cfg.get("max_steer", 0.4)
        self.base_speed = cfg.get("base_speed", 0.2)
        self.curve_speed = cfg.get("curve_speed", 0.2)
        self.curve_threshold = cfg.get("curve_threshold", 350)
        self.steering_threshold = cfg.get("steering_threshold", 0.2)
        self.curve_boost = cfg.get("curve_boost", 1.3)
        self.detection_threshold = cfg.get("detection_threshold", 500)

        self.frame_count = 0
        self._prev_error = 0.0
        self._filtered_error = 0.0
        self._lane_half_width = float(_LINE_OFFSET)
        self._left_history = deque(maxlen=3)
        self._right_history = deque(maxlen=3)
        self.last_debug_info = self._empty_debug_info(480, 640)
        self._aruco_detector = None
        self._last_sign_time = -10.0
        self._stop_visible_prev = False
        self._slow_visible_prev = False
        self._slow_until = 0.0
        self._stop_ramp_start = None
        self._stop_hold_until = 0.0
        self._last_sign_state = "none"
        if hasattr(cv2, "aruco"):
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            aruco_params = cv2.aruco.DetectorParameters()
            self._aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

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
        return (sum(self._left_history) / len(self._left_history), sum(self._right_history) / len(self._right_history))

    def _update_sign_state(self, frame_bgr: np.ndarray, now: float):
        if self._aruco_detector is None:
            self._last_sign_state = "none"
            return
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, ids, _ = self._aruco_detector.detectMarkers(gray)
        if ids is None:
            self._stop_visible_prev = False
            self._slow_visible_prev = False
            self._last_sign_state = "none"
            return

        tag_ids = {int(v) for v in ids.flatten()}
        stop_now = any(t in STOP_TAG_IDS for t in tag_ids)
        slow_now = any(t in SLOW_TAG_IDS for t in tag_ids)
        stop_edge = stop_now and not self._stop_visible_prev
        slow_edge = slow_now and not self._slow_visible_prev
        self._stop_visible_prev = stop_now
        self._slow_visible_prev = slow_now

        if now - self._last_sign_time < SIGN_COOLDOWN_SECONDS:
            self._last_sign_state = "cooldown"
            return

        if stop_edge and now >= self._stop_hold_until and self._stop_ramp_start is None:
            self._last_sign_time = now
            self._stop_ramp_start = now
            self._last_sign_state = "stop_detected"
            return

        if slow_edge:
            self._last_sign_time = now
            self._slow_until = max(self._slow_until, now + SLOW_SECONDS)
            self._last_sign_state = "slow_detected"
            return

        self._last_sign_state = "visible"

    def _speed_scale_from_signs(self, now: float) -> float:
        scale = 1.0
        if now < self._slow_until:
            scale *= SLOW_SPEED_MULTIPLIER

        if self._stop_ramp_start is not None:
            alpha = min(1.0, (now - self._stop_ramp_start) / STOP_RAMP_SECONDS)
            scale *= (1.0 - alpha)
            if alpha >= 1.0:
                self._stop_ramp_start = None
                self._stop_hold_until = now + STOP_HOLD_SECONDS

        if now < self._stop_hold_until:
            return 0.0
        return scale

    def compute_commands(self, image: np.ndarray) -> Tuple[float, float]:
        self.frame_count += 1
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        now = time.time()
        self._update_sign_state(bgr, now)
        mask_left, mask_right = detect_lane_markings(bgr)
        mask_y = (mask_left * 255).astype(np.uint8)
        mask_w = (mask_right * 255).astype(np.uint8)
        yellow_pixels = int(np.count_nonzero(mask_y))
        white_pixels = int(np.count_nonzero(mask_w))
        total_pixels = yellow_pixels + white_pixels

        combined = np.clip(mask_left + mask_right, 0, 1)
        self.last_debug_info = {
            "roi": image,
            "lane_mask": (combined * 255).astype(np.uint8),
            "white_mask": mask_w,
            "yellow_mask": mask_y,
            "total_lane_pixels": total_pixels,
            "lateral_error": float(np.clip(self._prev_error, -1.0, 1.0)),
            "lane_detected": total_pixels >= self.detection_threshold,
            "frame_count": self.frame_count,
        }

        h, w = mask_y.shape
        left_det = yellow_pixels > 0
        right_det = white_pixels > 0
        recovery = total_pixels < self.detection_threshold
        yellow_xs, white_xs = detect_lines_in_slices(mask_y, mask_w, h)
        both_visible = left_det and right_det and not recovery
        is_curve, curve_dir = detect_curve(yellow_xs, white_xs, self.curve_threshold)

        raw_error = self._calculate_error(yellow_xs, white_xs, left_det, right_det, w)
        self._filtered_error = 0.7 * self._filtered_error + 0.3 * raw_error
        steering = self._calculate_steering(self._filtered_error)
        left, right = self._motor_commands(steering, recovery, is_curve, both_visible)

        speed_scale = self._speed_scale_from_signs(now)
        if speed_scale <= 0.0:
            left, right = 0.0, 0.0
        else:
            left *= speed_scale
            right *= speed_scale

        left, right = self._smooth(left, right, both_visible)

        slice_height = int(h * 0.35 / _NUM_SLICES)
        start_y = int(h * _ROI_START)
        self.last_debug_info.update(
            {
                "yellow_xs": yellow_xs,
                "white_xs": white_xs,
                "slice_ys": [start_y + i * slice_height + slice_height // 2 for i in range(_NUM_SLICES)],
                "is_curve": is_curve,
                "curve_dir": curve_dir,
                "sign_state": self._last_sign_state,
            }
        )
        return left, right

    def _empty_debug_info(self, h, w):
        return {
            "roi": np.zeros((h, w, 3), dtype=np.uint8),
            "lane_mask": np.zeros((h, w), dtype=np.uint8),
            "white_mask": np.zeros((h, w), dtype=np.uint8),
            "yellow_mask": np.zeros((h, w), dtype=np.uint8),
            "total_lane_pixels": 0,
            "lateral_error": 0.0,
            "lane_detected": False,
            "frame_count": 0,
        }

    @property
    def sign_state(self) -> str:
        return self._last_sign_state


def _set_running_leds(leds):
    if leds is None:
        return
    try:
        leds.set_rgb(0, [0.0, 1.0, 0.0])
        leds.set_rgb(2, [0.0, 1.0, 0.0])
        leds.set_rgb(3, [0.0, 1.0, 0.0])
        leds.set_rgb(4, [0.0, 1.0, 0.0])
    except Exception:
        pass


def build_project_lane_agent() -> LaneServoingAgent:
    agent = LaneServoingAgent()
    agent.base_speed = BASE_SPEED
    agent.curve_speed = CURVE_SPEED
    agent.p_gain = P_GAIN
    agent.d_gain = D_GAIN
    agent.detection_threshold = DETECTION_THRESHOLD
    return agent


def main(camera, wheels, leds, stop_event):
    agent = build_project_lane_agent()
    _set_running_leds(leds)
    while not stop_event.is_set():
        ok, frame_rgb = camera.read_rgb()
        if not ok or frame_rgb is None:
            wheels.set_wheels_speed(BASE_SPEED, BASE_SPEED)
            time.sleep(DT)
            continue
        left, right = agent.compute_commands(frame_rgb)
        if left <= 0.0 and right <= 0.0:
            left, right = BASE_SPEED, BASE_SPEED
        wheels.set_wheels_speed(left, right)
        time.sleep(DT)

    wheels.set_wheels_speed(0.0, 0.0)
    if leds is not None:
        try:
            leds.all_off()
        except Exception:
            pass