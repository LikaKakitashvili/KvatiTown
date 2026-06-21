from typing import Tuple
import os
import cv2
import numpy as np
import yaml

HSV_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "config", "lane_servoing_hsv_config.yaml",
))

try:
    with open(HSV_FILE, "r") as f:
        _h = yaml.safe_load(f) or {}
except Exception:
    _h = {}

_yellow_lower = np.array([_h.get("yellow_lower_h", 18), _h.get("yellow_lower_s", 70), _h.get("yellow_lower_v", 70)], dtype=np.uint8)
_yellow_upper = np.array([_h.get("yellow_upper_h", 42), _h.get("yellow_upper_s", 255), _h.get("yellow_upper_v", 255)], dtype=np.uint8)
_white_lower  = np.array([_h.get("white_lower_h",   0), _h.get("white_lower_s",   0), _h.get("white_lower_v", 180)], dtype=np.uint8)
_white_upper  = np.array([_h.get("white_upper_h", 179), _h.get("white_upper_s",  60), _h.get("white_upper_v", 255)], dtype=np.uint8)

# ROI: only process the bottom portion of the frame
_ROI_START = 0.45
# White line: small left clip to avoid yellow-line false positives on turns
_WHITE_LEFT_CLIP = 0.20


def detect_lane_markings(image):
    # type: (np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    """
    image is BGR.
    Returns (yellow_mask, white_mask) as float32 arrays (0.0 or 1.0).
    Pure HSV + morphological cleaning — no complex component analysis.
    """
    h, w = image.shape[:2]

    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[int(h * _ROI_START):, :] = 255

    raw_yellow = cv2.bitwise_and(cv2.inRange(imghsv, _yellow_lower, _yellow_upper), roi_mask)
    raw_white  = cv2.bitwise_and(cv2.inRange(imghsv, _white_lower,  _white_upper),  roi_mask)

    # Remove left-half white detections (white line is always on the right)
    raw_white[:, :int(w * _WHITE_LEFT_CLIP)] = 0

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    clean_yellow = cv2.morphologyEx(raw_yellow, cv2.MORPH_OPEN,  k_open)
    clean_yellow = cv2.morphologyEx(clean_yellow, cv2.MORPH_CLOSE, k_close)

    clean_white  = cv2.morphologyEx(raw_white,  cv2.MORPH_OPEN,  k_open)
    clean_white  = cv2.morphologyEx(clean_white,  cv2.MORPH_CLOSE, k_close)

    return (clean_yellow > 0).astype(np.float32), (clean_white > 0).astype(np.float32)


def reload_hsv_config():
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper
    try:
        with open(HSV_FILE) as f:
            h = yaml.safe_load(f) or {}
        _yellow_lower = np.array([h.get("yellow_lower_h", 18), h.get("yellow_lower_s", 70), h.get("yellow_lower_v", 70)], dtype=np.uint8)
        _yellow_upper = np.array([h.get("yellow_upper_h", 42), h.get("yellow_upper_s", 255), h.get("yellow_upper_v", 255)], dtype=np.uint8)
        _white_lower  = np.array([h.get("white_lower_h",   0), h.get("white_lower_s",   0), h.get("white_lower_v", 180)], dtype=np.uint8)
        _white_upper  = np.array([h.get("white_upper_h", 179), h.get("white_upper_s",  60), h.get("white_upper_v", 255)], dtype=np.uint8)
    except Exception:
        pass


def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper
    _yellow_lower = np.array(yellow_lower, dtype=np.uint8)
    _yellow_upper = np.array(yellow_upper, dtype=np.uint8)
    _white_lower  = np.array(white_lower,  dtype=np.uint8)
    _white_upper  = np.array(white_upper,  dtype=np.uint8)


def get_hsv_bounds():
    return {
        "yellow_lower_h": int(_yellow_lower[0]), "yellow_upper_h": int(_yellow_upper[0]),
        "yellow_lower_s": int(_yellow_lower[1]), "yellow_upper_s": int(_yellow_upper[1]),
        "yellow_lower_v": int(_yellow_lower[2]), "yellow_upper_v": int(_yellow_upper[2]),
        "white_lower_h":  int(_white_lower[0]),  "white_upper_h":  int(_white_upper[0]),
        "white_lower_s":  int(_white_lower[1]),  "white_upper_s":  int(_white_upper[1]),
        "white_lower_v":  int(_white_lower[2]),  "white_upper_v":  int(_white_upper[2]),
    }
