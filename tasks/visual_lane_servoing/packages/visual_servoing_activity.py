from typing import Tuple
import os
import numpy as np
import cv2
import yaml

_HSV_FILE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_hsv_config.yaml')
try:
    with open(_HSV_FILE) as _f:
        _h = yaml.safe_load(_f) or {}
except FileNotFoundError:
    _h = {}

_yellow_lower = np.array([_h.get('yellow_lower_h', 0),  _h.get('yellow_lower_s', 0),  _h.get('yellow_lower_v', 0)])
_yellow_upper = np.array([_h.get('yellow_upper_h', 0),  _h.get('yellow_upper_s', 0), _h.get('yellow_upper_v', 0)])

_white_lower = np.array([_h.get('white_lower_h', 0),   _h.get('white_lower_s', 0), _h.get('white_lower_v', 0)])
_white_upper = np.array([_h.get('white_upper_h', 0), _h.get('white_upper_s', 0), _h.get('white_upper_v', 0)])

def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Generate Grayscale and HSV variants of the image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Smooth the image using a Gaussian kernel to reduce high-frequency noise
    img_gaussian_filter = cv2.GaussianBlur(img_gray, (0, 0), 1.0)

    # Compute horizontal and vertical gradients using Sobel Operator
    sobelx = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 0, 1)

    # Calculate magnitude of gradients and filter out weaker edges
    gmag = np.sqrt(sobelx**2 + sobely**2)
    mask_mag = (gmag > 40)  # Minimum threshold on gradient magnitude

    # Calculate left/right positional masks
    width = image.shape[1]
    mask_left = np.ones(img_gray.shape, dtype=np.uint8)
    mask_left[:, int(np.floor(width / 2)):width + 1] = 0

    mask_right = np.ones(img_gray.shape, dtype=np.uint8)
    mask_right[:, 0:int(np.floor(width / 2))] = 0

    # Directional masks based on positive/negative derivative trends
    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_neg = (sobely < 0)

    # Color isolation masks
    mask_yellow = cv2.inRange(img_hsv, _yellow_lower, _yellow_upper) > 0
    mask_white = cv2.inRange(img_hsv, _white_lower, _white_upper) > 0

    # Compose final detection masks
    mask_left_edge = mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    return mask_left_edge.astype(np.float32), mask_right_edge.astype(np.float32)




def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper
    _yellow_lower    = np.array(yellow_lower)
    _yellow_upper    = np.array(yellow_upper)
    _white_lower = np.array(white_lower)
    _white_upper = np.array(white_upper)

def get_hsv_bounds():
    return {
        'yellow_lower_h': int(_yellow_lower[0]),    'yellow_upper_h': int(_yellow_upper[0]),
        'yellow_lower_s': int(_yellow_lower[1]),    'yellow_upper_s': int(_yellow_upper[1]),
        'yellow_lower_v': int(_yellow_lower[2]),    'yellow_upper_v': int(_yellow_upper[2]),
        'white_lower_h':  int(_white_lower[0]), 'white_upper_h':  int(_white_upper[0]),
        'white_lower_s':  int(_white_lower[1]), 'white_upper_s':  int(_white_upper[1]),
        'white_lower_v':  int(_white_lower[2]), 'white_upper_v':  int(_white_upper[2]),
    }