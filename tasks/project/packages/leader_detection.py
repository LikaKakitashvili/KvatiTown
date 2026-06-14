"""Detect the leader duckiebot with the object-detection CNN (class: duckie)."""
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

DUCKIE_CLASS_ID = 0

_det_agent = None
_det_init_failed = False
_last_ratio: Optional[float] = None
_last_stop = False
_frame_counter = 0


def _get_detector():
    global _det_agent, _det_init_failed
    if _det_init_failed:
        return None
    if _det_agent is not None:
        return _det_agent
    try:
        from tasks.object_detection.packages.agent import ObjectDetectionAgent
        _det_agent = ObjectDetectionAgent()
        if not _det_agent.model_loaded:
            print(f"[Project][Vision] detector unavailable: {_det_agent.load_error}", flush=True)
            _det_init_failed = True
            return None
        print(f"[Project][Vision] duckie detector ready ({_det_agent.img_size}px)", flush=True)
        return _det_agent
    except Exception as e:
        print(f"[Project][Vision] detector init failed: {e}", flush=True)
        _det_init_failed = True
        return None


def _bbox_height_ratio(bbox: Tuple[int, int, int, int], frame_h: int) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(0, y2 - y1)) / max(1.0, float(frame_h))


def check_leader_vision(
    frame_bgr: Any,
    cfg: Dict[str, Any],
) -> Tuple[Optional[float], bool]:
    """
    Returns (proximity_ratio, should_stop).
    proximity_ratio: apparent leader height / frame height (larger = closer).
    should_stop: True when leader is too close and follower must halt.
    """
    global _last_ratio, _last_stop, _frame_counter

    if not cfg.get("vision_avoid_enabled", False):
        return None, False
    if frame_bgr is None:
        return _last_ratio, _last_stop

    interval = max(1, int(cfg.get("vision_detect_interval", 2)))
    _frame_counter += 1
    if _frame_counter % interval != 0:
        return _last_ratio, _last_stop

    detector = _get_detector()
    if detector is None:
        return None, False

    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = detector.detect(frame_rgb)
    except Exception as e:
        print(f"[Project][Vision] detect failed: {e}", flush=True)
        return _last_ratio, _last_stop

    if not detections:
        _last_ratio, _last_stop = None, False
        return _last_ratio, _last_stop

    conf_min = float(cfg.get("vision_conf_min", 0.5))
    leader_cls = int(cfg.get("vision_leader_class_id", DUCKIE_CLASS_ID))
    h, w = frame_bgr.shape[:2]

    duckies = [
        d for d in detections
        if int(d[2]) == leader_cls and float(d[1]) >= conf_min
    ]
    if not duckies:
        _last_ratio, _last_stop = None, False
        return _last_ratio, _last_stop

    best = max(duckies, key=lambda d: _bbox_height_ratio(d[0], h))
    ratio = _bbox_height_ratio(best[0], h)
    stop_ratio = float(cfg.get("vision_stop_ratio", 0.12))
    _, _, _, y2 = best[0]
    too_low = float(y2) > h * float(cfg.get("vision_stop_y_frac", 0.55))

    should_stop = ratio >= stop_ratio or too_low
    _last_ratio, _last_stop = ratio, should_stop

    if should_stop:
        print(f"[Project][Vision] leader duckie close (ratio={ratio:.3f}) — stop", flush=True)

    return _last_ratio, _last_stop


def speed_cap_from_vision(
    proximity_ratio: float,
    cfg: Dict[str, Any],
    base_speed: float,
) -> float:
    """Reduce speed when the leader fills more of the frame (closer)."""
    stop_ratio = float(cfg.get("vision_stop_ratio", 0.12))
    slow_ratio = float(cfg.get("vision_slow_ratio", 0.06))
    if proximity_ratio >= stop_ratio:
        return 0.0
    if proximity_ratio <= slow_ratio:
        return base_speed
    t = (proximity_ratio - slow_ratio) / max(1e-6, stop_ratio - slow_ratio)
    return max(0.0, base_speed * (1.0 - t))
