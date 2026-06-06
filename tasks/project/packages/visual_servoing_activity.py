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
except FileNotFoundError:
    _h = {}

_yellow_lower = np.array([_h.get('yellow_lower_h', 15), _h.get('yellow_lower_s', 80),  _h.get('yellow_lower_v', 77)])
_yellow_upper = np.array([_h.get('yellow_upper_h', 80), _h.get('yellow_upper_s', 255), _h.get('yellow_upper_v', 255)])
_white_lower  = np.array([_h.get('white_lower_h',  0),  _h.get('white_lower_s',  0),   _h.get('white_lower_v',  180)])
_white_upper  = np.array([_h.get('white_upper_h',  179),_h.get('white_upper_s',  50),  _h.get('white_upper_v',  255)])


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects the left (yellow) and right (white) lane markings in the image.

    Args:
        image: BGR image from the camera.

    Returns:
        Tuple of two uint8 arrays (0 or 255) for left and right lane masks.
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    img_blur = cv2.GaussianBlur(img_gray, (0, 0), 3)

    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1)

    Gmag     = np.sqrt(sobelx**2 + sobely**2)
    mask_mag = (Gmag > 50)

    height, width = img_gray.shape
    half_width    = int(np.floor(width / 2))

    mask_left  = np.ones_like(img_gray, dtype=bool)
    mask_left[:, half_width:] = False

    mask_right = np.ones_like(img_gray, dtype=bool)
    mask_right[:, :half_width] = False

    mask_sobelx_neg = (sobelx < 0)
    mask_sobelx_pos = (sobelx > 0)
    mask_sobely_neg = (sobely < 0)

    mask_yellow_color = cv2.inRange(img_hsv, _yellow_lower, _yellow_upper) > 0
    mask_white_color  = cv2.inRange(img_hsv, _white_lower,  _white_upper)  > 0

    mask_left_edge  = mask_left  & mask_mag & mask_sobelx_neg & mask_sobely_neg & mask_yellow_color
    mask_right_edge = mask_right & mask_mag & mask_sobelx_pos & mask_sobely_neg & mask_white_color

    return (mask_left_edge.astype(np.uint8)  * 255,
            mask_right_edge.astype(np.uint8) * 255)


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
