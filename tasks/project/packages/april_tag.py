import cv2
import numpy as np
from typing import List, Optional, Tuple

_dt_detector = None
_backend: Optional[str] = None
_backend_logged = False


class TagDetection:
    __slots__ = ("tag_id", "center", "corners")

    def __init__(self, tag_id: int, center: Tuple[float, float], corners: np.ndarray):
        self.tag_id = int(tag_id)
        self.center = center
        self.corners = corners


def get_detector_backend() -> str:
    global _backend
    if _backend is not None:
        return _backend
    if _init_dt_detector() is not None:
        _backend = "dt_apriltags"
    elif _has_aruco():
        _backend = "aruco"
    else:
        _backend = "none"
    return _backend


def _log_backend_once() -> None:
    global _backend_logged
    if not _backend_logged:
        print(f"[Project][AprilTag] backend: {get_detector_backend()}", flush=True)
        _backend_logged = True


def _init_dt_detector():
    global _dt_detector
    if _dt_detector is not None:
        return _dt_detector
    try:
        from dt_apriltags import Detector
    except ImportError:
        return None

    _dt_detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )
    return _dt_detector


def _has_aruco() -> bool:
    try:
        cv2.aruco  # noqa: B018
        return True
    except AttributeError:
        return False


def _detect_dt_apriltags(gray: np.ndarray) -> List[TagDetection]:
    detector = _init_dt_detector()
    if detector is None:
        return []

    tags: List[TagDetection] = []
    for det in detector.detect(gray, estimate_tag_pose=False):
        corners = np.asarray(det.corners, dtype=np.float32).reshape(4, 2)
        cx = float(corners[:, 0].mean())
        cy = float(corners[:, 1].mean())
        tags.append(TagDetection(int(det.tag_id), (cx, cy), corners))
    return tags


def _detect_aruco(gray: np.ndarray) -> List[TagDetection]:
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []

    tags: List[TagDetection] = []
    for c, tid in zip(corners, ids.flatten()):
        pts = c.reshape(4, 2)
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        tags.append(TagDetection(int(tid), (cx, cy), pts))
    return tags


def detect_tags(frame_gray: Optional[np.ndarray]) -> List[TagDetection]:
    _log_backend_once()
    if frame_gray is None:
        return []

    backend = get_detector_backend()
    if backend == "dt_apriltags":
        tags = _detect_dt_apriltags(frame_gray)
    elif backend == "aruco":
        tags = _detect_aruco(frame_gray)
    else:
        tags = []

    if tags:
        ids = [t.tag_id for t in tags]
        print(f"[Project][AprilTag] detected ({backend}): {ids}", flush=True)
    return tags
