from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple


# =========================================================
# TAG IDENTITY
# =========================================================

class TagID(IntEnum):
    TURN_RIGHT_FWD  = 9   # right_forward
    TURN_LEFT_FWD   = 10  # left_forward
    TURN_LEFT_RIGHT = 11  # leftRight
    STOP            = 26  # stop
    YIELD           = 39  # yield

# Multiple physical stop-sign variants all resolve to TagID.STOP
_TAG_ID_MAP: Dict[int, TagID] = {
    9:  TagID.TURN_RIGHT_FWD,
    10: TagID.TURN_LEFT_FWD,
    11: TagID.TURN_LEFT_RIGHT,
    20: TagID.STOP,
    24: TagID.STOP,
    25: TagID.STOP,
    26: TagID.STOP,
    39: TagID.YIELD,
}

def resolve_tag(raw_id: int) -> Optional[TagID]:
    """Map a raw detected tag ID to a canonical TagID, or None if unknown."""
    return _TAG_ID_MAP.get(raw_id)

# Allowed turn directions per intersection tag
_TAG_TURNS: Dict[int, List[str]] = {
    TagID.TURN_LEFT_RIGHT: ["left", "right"],
    TagID.TURN_LEFT_FWD:   ["left", "forward"],
    TagID.TURN_RIGHT_FWD:  ["right", "forward"],
}


# =========================================================
# FSM STATE
# =========================================================

class State(IntEnum):
    MOVING      = auto()
    SLOWING     = auto()
    YIELD_CREEP = auto()
    STOPPED     = auto()
    CHECKPATH   = auto()
    POST_STOP   = auto()
    INTERSECT   = auto()
    PRE_TURN    = auto()
    TURNING     = auto()
    EXITING     = auto()


# =========================================================
# BEHAVIOR CONFIG
# =========================================================

@dataclass
class SignBehaviorConfig:
    # Red-line detection
    red_strip_frac = 0.55  # bottom 45%
    red_pixel_frac = 0.045  # 4.5% red pixels
    red_hsv_low1    = (0,   120,  80)   # type: Tuple[int, int, int]
    red_hsv_high1   = (10,  255, 255)   # type: Tuple[int, int, int]
    red_hsv_low2    = (170, 120,  80)   # type: Tuple[int, int, int]
    red_hsv_high2   = (180, 255, 255)   # type: Tuple[int, int, int]

    # Stop-line hold
    stop_hold_frames = 0

    # Speed ramp
    slow_ramp_factor        = 0.92
    stopped_speed_threshold = 0.05

    # YIELD specific
    yield_min_speed    = 0.1
    yield_creep_frames = 5

    # CHECKPATH sweep
    check_left_frames   = 4
    check_right_frames  = 4
    check_turn_speed    = 0.04
    check_settle_frames = 5

    # POST_STOP
    post_stop_frames = 25
    post_stop_speed  = 0.2

    # Pre-turn forward creep
    preturn_right_frames = 11
    preturn_left_frames  = 20
    preturn_speed        = 0.20

    # Intersection manoeuvres
    intersect_forward_frames = 28
    intersect_left_frames    = 18
    intersect_right_frames   = 18
    intersect_forward_speed  = (0.20, 0.20)  # type: Tuple[float, float]

    # changed so one speed isn't 0 - no turning in place
    intersect_left_speed     = (0.15, 0.22)  # type: Tuple[float, float]
    intersect_right_speed    = (0.22, 0.15)  # type: Tuple[float, float]

    # Exiting
    exit_speed            = 0.20
    exit_timeout_frames   = 5    # after intersect - move forward to get into lane
