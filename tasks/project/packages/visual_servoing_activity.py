from typing import Tuple
import os
import numpy as np
import cv2
import yaml

_HSV_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_hsv_config.yaml'
))


def _load_hsv_config() -> dict:
    try:
        with open(_HSV_FILE) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _apply_hsv_config(h: dict) -> None:
    global _yellow_lower, _yellow_upper, _yellow_alt_lower, _yellow_alt_upper
    global _white_lower, _white_upper
    global _yellow_side_width_frac, _white_side_width_frac, _yellow_center_exclude_frac
    global _edge_mag_threshold, _use_white_balance, _top_exclude_frac

    _yellow_lower = np.array([
        h.get('yellow_lower_h', 8), h.get('yellow_lower_s', 30), h.get('yellow_lower_v', 40),
    ])
    _yellow_upper = np.array([
        h.get('yellow_upper_h', 55), h.get('yellow_upper_s', 255), h.get('yellow_upper_v', 255),
    ])
    _yellow_alt_lower = np.array([
        h.get('yellow_alt_lower_h', 82), h.get('yellow_alt_lower_s', 55), h.get('yellow_alt_lower_v', 70),
    ])
    _yellow_alt_upper = np.array([
        h.get('yellow_alt_upper_h', 105), h.get('yellow_alt_upper_s', 255), h.get('yellow_alt_upper_v', 255),
    ])
    _white_lower = np.array([
        h.get('white_lower_h', 0), h.get('white_lower_s', 0), h.get('white_lower_v', 120),
    ])
    _white_upper = np.array([
        h.get('white_upper_h', 179), h.get('white_upper_s', 110), h.get('white_upper_v', 255),
    ])
    _yellow_side_width_frac = float(h.get('yellow_side_width_frac', 0.75))
    _white_side_width_frac = float(h.get('white_side_width_frac', 0.75))
    _yellow_center_exclude_frac = float(h.get('yellow_center_exclude_frac', 0.18))
    _edge_mag_threshold = float(h.get('edge_mag_threshold', 28))
    _use_white_balance = bool(h.get('use_white_balance', True))
    _top_exclude_frac = float(h.get('top_exclude_frac', 0.0))


_apply_hsv_config(_load_hsv_config())


def reload_hsv_config() -> None:
    _apply_hsv_config(_load_hsv_config())


def _top_exclude_row(h: int) -> int:
    frac = max(0.0, min(0.9, float(_top_exclude_frac)))
    return int(np.floor(h * frac))


def _mask_top_rows(mask: np.ndarray) -> np.ndarray:
    cut = _top_exclude_row(mask.shape[0])
    if cut > 0:
        mask[:cut, :] = False
    return mask


def _correct_white_balance(bgr: np.ndarray) -> np.ndarray:
    img = bgr.astype(np.float32)
    means = [img[:, :, c].mean() + 1e-6 for c in range(3)]
    gray = sum(means) / 3.0
    for c in range(3):
        img[:, :, c] *= gray / means[c]
    return np.clip(img, 0, 255).astype(np.uint8)


def _yellow_color_mask(img_hsv: np.ndarray) -> np.ndarray:
    h, w = img_hsv.shape[:2]
    m_primary = cv2.inRange(img_hsv, _yellow_lower, _yellow_upper) > 0
    m_alt = cv2.inRange(img_hsv, _yellow_alt_lower, _yellow_alt_upper) > 0
    m = m_primary | m_alt

    white = cv2.inRange(img_hsv, _white_lower, _white_upper) > 0
    m = m & ~white

    yellow_cut = int(np.floor(w * _yellow_side_width_frac))
    m[:, yellow_cut:] = False

    half = max(0.0, min(0.45, float(_yellow_center_exclude_frac))) * 0.5
    cx0 = int(w * (0.5 - half))
    cx1 = int(w * (0.5 + half))
    m[:, cx0:cx1] = False

    m_u8 = (m.astype(np.uint8) * 255)
    m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return _mask_top_rows(m_u8 > 0)


def _white_color_mask(img_hsv: np.ndarray) -> np.ndarray:
    w = img_hsv.shape[1]
    m = cv2.inRange(img_hsv, _white_lower, _white_upper) > 0
    white_start = int(np.floor(w * (1.0 - _white_side_width_frac)))
    m[:, 0:white_start] = False
    return _mask_top_rows(m)


def get_color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    work = _correct_white_balance(bgr) if _use_white_balance else bgr
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    return (
        (_yellow_color_mask(hsv).astype(np.uint8) * 255),
        (_white_color_mask(hsv).astype(np.uint8) * 255),
    )


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    work = _correct_white_balance(image) if _use_white_balance else image
    img_gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

    img_gaussian_filter = cv2.GaussianBlur(img_gray, (0, 0), 1.0)

    sobelx = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 0, 1)

    gmag = np.sqrt(sobelx**2 + sobely**2)
    mask_mag = gmag > _edge_mag_threshold

    width = work.shape[1]
    yellow_cut = int(np.floor(width * _yellow_side_width_frac))
    mask_left = np.ones(img_gray.shape, dtype=np.uint8)
    mask_left[:, yellow_cut:width + 1] = 0

    white_start = int(np.floor(width * (1.0 - _white_side_width_frac)))
    mask_right = np.ones(img_gray.shape, dtype=np.uint8)
    mask_right[:, 0:white_start] = 0

    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_neg = (sobely < 0)

    mask_yellow = _yellow_color_mask(img_hsv)
    mask_white = _white_color_mask(img_hsv)

    top_cut = _top_exclude_row(img_gray.shape[0])
    if top_cut > 0:
        mask_left[:top_cut, :] = 0
        mask_right[:top_cut, :] = 0
        mask_mag[:top_cut, :] = False

    mask_left_edge = mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    return mask_left_edge.astype(np.float32), mask_right_edge.astype(np.float32)


def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper
    _yellow_lower = np.array(yellow_lower)
    _yellow_upper = np.array(yellow_upper)
    _white_lower = np.array(white_lower)
    _white_upper = np.array(white_upper)


def get_hsv_bounds():
    return {
        'yellow_lower_h': int(_yellow_lower[0]), 'yellow_upper_h': int(_yellow_upper[0]),
        'yellow_lower_s': int(_yellow_lower[1]), 'yellow_upper_s': int(_yellow_upper[1]),
        'yellow_lower_v': int(_yellow_lower[2]), 'yellow_upper_v': int(_yellow_upper[2]),
        'white_lower_h': int(_white_lower[0]), 'white_upper_h': int(_white_upper[0]),
        'white_lower_s': int(_white_lower[1]), 'white_upper_s': int(_white_upper[1]),
        'white_lower_v': int(_white_lower[2]), 'white_upper_v': int(_white_upper[2]),
    }
