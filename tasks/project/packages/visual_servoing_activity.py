from typing import Tuple
import os
import numpy as np
import cv2
import yaml

_HSV_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_hsv_config.yaml'
))

try:
    with open(_HSV_FILE) as _f:
        _h = yaml.safe_load(_f) or {}
except Exception:
    _h = {}

# Wider defaults than the Sobel version — works on real hardware under varied lighting.
_yellow_lower = np.array([_h.get('yellow_lower_h', 15), _h.get('yellow_lower_s', 60),  _h.get('yellow_lower_v', 60)])
_yellow_upper = np.array([_h.get('yellow_upper_h', 50), _h.get('yellow_upper_s', 255), _h.get('yellow_upper_v', 255)])
_white_lower  = np.array([_h.get('white_lower_h',  0),  _h.get('white_lower_s',  0),   _h.get('white_lower_v',  140)])
_white_upper  = np.array([_h.get('white_upper_h',  179),_h.get('white_upper_s',  70),  _h.get('white_upper_v',  255)])


def detect_lane_markings(image):
    # type: (np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    """
    Detect yellow (left) and white (right) lane markings using HSV color masking.

    Returns two uint8 arrays with values 0 or 1.
    lane_agent.py multiplies the result by 255, so returning 0/1 gives correct
    0/255 masks for both detection and visualization.
    """
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    height, width = img_hsv.shape[:2]
    half_width = width // 2

    yellow_mask = cv2.inRange(img_hsv, _yellow_lower, _yellow_upper)
    white_mask  = cv2.inRange(img_hsv, _white_lower,  _white_upper)

    # Small morphological opening removes isolated noise pixels.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    white_mask  = cv2.morphologyEx(white_mask,  cv2.MORPH_OPEN, kernel)

    # Yellow line is on the left half, white line on the right half.
    left_mask         = yellow_mask.copy()
    left_mask[:,  half_width:] = 0

    right_mask        = white_mask.copy()
    right_mask[:, :half_width] = 0

    # Return 0/1 — lane_agent multiplies by 255 to get 0/255 display values.
    return (left_mask > 0).astype(np.uint8), (right_mask > 0).astype(np.uint8)


def reload_hsv_config():
    """Reload HSV bounds from disk after UI slider save."""
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper
    try:
        with open(_HSV_FILE) as f:
            h = yaml.safe_load(f) or {}
        _yellow_lower = np.array([h.get('yellow_lower_h', 15), h.get('yellow_lower_s', 60),  h.get('yellow_lower_v', 60)])
        _yellow_upper = np.array([h.get('yellow_upper_h', 50), h.get('yellow_upper_s', 255), h.get('yellow_upper_v', 255)])
        _white_lower  = np.array([h.get('white_lower_h',  0),  h.get('white_lower_s',  0),   h.get('white_lower_v',  140)])
        _white_upper  = np.array([h.get('white_upper_h',  179),h.get('white_upper_s',  70),  h.get('white_upper_v',  255)])
    except Exception:
        pass


def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper
    _yellow_lower = np.array(yellow_lower)
    _yellow_upper = np.array(yellow_upper)
    _white_lower  = np.array(white_lower)
    _white_upper  = np.array(white_upper)


def get_hsv_bounds():
    return {
        'yellow_lower_h': int(_yellow_lower[0]), 'yellow_upper_h': int(_yellow_upper[0]),
        'yellow_lower_s': int(_yellow_lower[1]), 'yellow_upper_s': int(_yellow_upper[1]),
        'yellow_lower_v': int(_yellow_lower[2]), 'yellow_upper_v': int(_yellow_upper[2]),
        'white_lower_h':  int(_white_lower[0]),  'white_upper_h':  int(_white_upper[0]),
        'white_lower_s':  int(_white_lower[1]),  'white_upper_s':  int(_white_upper[1]),
        'white_lower_v':  int(_white_lower[2]),  'white_upper_v':  int(_white_upper[2]),
    }
