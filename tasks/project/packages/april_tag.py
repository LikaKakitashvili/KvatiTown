import cv2
import numpy as np

tag_confirm_frames = 2

# Lower 36 bits of each AprilTag 36h11 code (from tag36h11.c in the apriltag library).
# Only the IDs used by this project are listed.
# To regenerate: on a machine with opencv-contrib, run:
#   d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
#   for i in [1,4,9,10,11]: print(i, hex(int.from_bytes(bytes(d.bytesList[i].flatten()[:5]),'big') >> 4))
_CODES_36H11 = {
    0xd97f18b49: 1,
    0xebcbca822: 4,   # was 0xe479e9c98 (that's ID 3)
    0x265ad0472: 9,   # was 0xf6c69b971 (invalid)
    0x34fe91b86: 10,  # was 0xfa6f8bf36 (invalid)
    0x3ff962cd5: 11,  # was 0xfe187c4fb (invalid)
    0x81da494af: 20,
    0xa2cabc89c: 24,
    0xadc58d9eb: 25,
    0xb16e7dfb0: 26,
    0x9f53856b5: 39,
}

def _order_corners(pts):
    """Sort 4 points into TL, TR, BR, BL order."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],   # TL: smallest x+y
        pts[np.argmin(d)],   # TR: smallest y-x
        pts[np.argmax(s)],   # BR: largest x+y
        pts[np.argmax(d)],   # BL: largest y-x
    ])


def _decode_warped(bw80):
    """
    bw80: 80x80 binary image (0 or 1).
    Sample the 8x8 cell grid (each cell 10x10, sampled at centre).
    AprilTag 36h11 layout: 1-cell black border + 6x6 data.
    Try all 4 rotations of the inner grid against _CODES_36H11.
    Returns tag_id int or None.
    """
    row_centres = np.arange(8) * 10 + 5   # [5,15,25,35,45,55,65,75]
    col_centres = np.arange(8) * 10 + 5
    bits = bw80[np.ix_(row_centres, col_centres)]  # shape (8,8)

    # outer ring must be all black (0)
    border = np.concatenate([
        bits[0, :], bits[7, :],
        bits[1:7, 0], bits[1:7, 7]
    ])
    if np.any(border != 0):
        return None

    inner = bits[1:7, 1:7]   # 6x6 data bits

    for k in range(4):
        rotated = np.rot90(inner, k=k)
        code = int(rotated.ravel().dot(1 << np.arange(35, -1, -1, dtype=np.int64)))
        if code in _CODES_36H11:
            return _CODES_36H11[code]

    return None


_DST_PTS = np.array([[0, 0], [79, 0], [79, 79], [0, 79]], dtype=np.float32)


def _raw_detect(gray):
    """
    Detect AprilTag 36h11 markers using only base OpenCV (no contrib/aruco).
    Returns list of {"tag_id": int, "corners": (4,2) float32 array}.
    """
    # Blur then adaptive threshold to isolate dark tag borders
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 5
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    max_area = h * w * 0.5

    tags = []
    seen = set()

    for cnt in contours:
        # Keep only reasonably-sized convex quadrilaterals
        area = cv2.contourArea(cnt)
        if area < 300 or area > max_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        # Warp to 80x80 canonical view and decode the tag bits
        src = _order_corners(approx.reshape(4, 2))
        M = cv2.getPerspectiveTransform(src, _DST_PTS)
        warped = cv2.warpPerspective(gray, M, (80, 80))

        _, bw = cv2.threshold(warped, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        tag_id = _decode_warped(bw)
        if tag_id is not None and tag_id not in seen:
            seen.add(tag_id)
            tags.append({"tag_id": tag_id, "corners": src})

    return tags


def detect_tags(signBehavior, frame_rgb):
    # type: (object, np.ndarray) -> list
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    # Fast path: use aruco if opencv-contrib is available (simulation)
    try:
        aruco = cv2.aruco
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            return []

        tags = []
        for c, tid in zip(corners, ids.flatten()):
            tags.append({"tag_id": int(tid), "corners": c.reshape(4, 2)})

        if tags:
            print("[SignBehavior] tags visible: {}".format([t['tag_id'] for t in tags]))
        return tags

    except AttributeError:
        pass   # cv2.aruco not compiled in (bot environment)

    # Raw fallback: no contrib required
    tags = _raw_detect(gray)
    if tags:
        print("[SignBehavior] tags visible (raw): {}".format([t['tag_id'] for t in tags]))
    return tags


def confirm_tags(signBehavior, raw_tags):
    # type: (object, list) -> list
    seen_ids = {t["tag_id"] for t in raw_tags}

    for k in [k for k in signBehavior._tag_buffer if k not in seen_ids]:
        del signBehavior._tag_buffer[k]

    for tid in seen_ids:
        signBehavior._tag_buffer[tid] = (
            signBehavior._tag_buffer.get(tid, 0) + 1
        )

    confirmed = [
        tid
        for tid, cnt in signBehavior._tag_buffer.items()
        if cnt >= tag_confirm_frames
    ]

    if confirmed:
        print("[SignBehavior] confirmed tags: {}".format(confirmed))

    return confirmed
