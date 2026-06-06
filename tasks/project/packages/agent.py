import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import yaml

from tasks.project.packages.lane_agent import LaneServoingAgent
from tasks.project.packages.april_tag import detect_tags as _at_detect, confirm_tags as _at_confirm
from tasks.project.packages.sign_behavior_config import resolve_tag, TagID as _TagID


_DEFAULT_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "project_config.yaml")
)


def _config_path() -> str:
    override = os.environ.get("PROJECT_CONFIG", "").strip()
    if override:
        return os.path.normpath(override)
    return _DEFAULT_CONFIG_PATH


# States
STATE_CRUISING = "CRUISING"
STATE_SLOW     = "SLOW"
STATE_STOPPING = "STOPPING"
STATE_STOPPED  = "STOPPED"

# Events
EVENT_NORMAL    = "EVENT_NORMAL"
EVENT_SLOW_SIGN = "EVENT_SLOW_SIGN"
EVENT_STOP_SIGN = "EVENT_STOP_SIGN"
EVENT_TIMEOUT   = "EVENT_TIMEOUT"


_status_lock = threading.Lock()
_leader_status: Dict[str, Any] = {
    "state": "STOPPED",
    "speed": 0.0,
    "ts": float(time.time()),
}


# Per-role tag confirmation buffers (hold consecutive-frame counts per tag ID)
class _TagBuf:
    def __init__(self):
        self._tag_buffer: Dict[int, int] = {}

_tag_buf_leader   = _TagBuf()
_tag_buf_follower = _TagBuf()

_sign_runtime = {
    "candidate_event": EVENT_NORMAL,
    "candidate_count": 0,
    "active_until": 0.0,
}

_lane_agent: Optional[LaneServoingAgent] = None

_viz_lock = threading.Lock()
_viz_frame_bgr: Optional[Any] = None
_viz_debug_info: Optional[Dict[str, Any]] = None


def set_viz_snapshot(frame_bgr: Optional[Any], debug_info: Optional[Dict[str, Any]] = None) -> None:
    global _viz_frame_bgr, _viz_debug_info
    with _viz_lock:
        _viz_frame_bgr = None if frame_bgr is None else frame_bgr.copy()
        _viz_debug_info = None if debug_info is None else dict(debug_info)


def get_viz_snapshot() -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    with _viz_lock:
        if _viz_frame_bgr is None:
            return None, None
        debug = None if _viz_debug_info is None else dict(_viz_debug_info)
        return _viz_frame_bgr.copy(), debug


def load_config() -> Dict[str, Any]:
    path = _config_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    return {
        "role":                str(cfg.get("role", "leader")).strip().lower(),
        "loop_hz":             float(cfg.get("loop_hz", 20)),
        "leader_host":         str(cfg.get("leader_host", "127.0.0.1")).strip(),
        "leader_port":         int(cfg.get("leader_port", 5055)),
        "cruise_speed":        float(cfg.get("cruise_speed", 0.2)),
        "slow_speed":          float(cfg.get("slow_speed", 0.12)),
        "follower_max_speed":  float(cfg.get("follower_max_speed", 0.2)),
        "follower_min_speed":  float(cfg.get("follower_min_speed", 0.0)),
        "distance_target":     float(cfg.get("distance_target", 0.06)),
        "distance_kp":         float(cfg.get("distance_kp", 0.6)),
        "status_publish_hz":   float(cfg.get("status_publish_hz", 10)),
        "status_poll_hz":      float(cfg.get("status_poll_hz", 10)),
        "request_timeout_s":   float(cfg.get("request_timeout_s", 0.2)),
        "leader_timeout_s":    float(cfg.get("leader_timeout_s", 0.4)),
        "stop_hold_s":         float(cfg.get("stop_hold_s", 2.0)),
        "decel_time_s":        float(cfg.get("decel_time_s", 0.8)),
        "decel_steps":         int(cfg.get("decel_steps", 8)),
        "stop_tag_ids":        [int(x) for x in cfg.get("stop_tag_ids", [])],
        "slow_tag_ids":        [int(x) for x in cfg.get("slow_tag_ids", [])],
        "sign_confirm_frames": int(cfg.get("sign_confirm_frames", 2)),
        "sign_cooldown_s":     float(cfg.get("sign_cooldown_s", 3.0)),
        "sign_center_roi":     float(cfg.get("sign_center_roi", 1.0)),
        "leader_tag_ids":      [int(x) for x in cfg.get("leader_tag_ids", [])],
    }


def build_status_payload(state: str, speed: float) -> Dict[str, Any]:
    return {
        "state": str(state).upper(),
        "speed": float(speed),
        "ts":    float(time.time()),
    }


def set_leader_status(payload: Dict[str, Any]) -> None:
    with _status_lock:
        _leader_status.update(payload)


def get_leader_status() -> Dict[str, Any]:
    with _status_lock:
        return dict(_leader_status)


def _safe_stop(wheels) -> None:
    try:
        if wheels is not None:
            wheels.set_wheels_speed(0.0, 0.0)
    except Exception as e:
        print(f"[Project] Wheels stop failed: {e}")


def smooth_stop(
    wheels,
    current_speed: float,
    decel_time_s: float,
    decel_steps: int,
    stop_event,
) -> None:
    if wheels is None:
        return
    speed0   = max(0.0, float(current_speed))
    steps    = max(1, int(decel_steps))
    step_dt  = max(0.0, float(decel_time_s)) / steps
    try:
        for i in range(steps - 1, -1, -1):
            if stop_event.is_set():
                break
            wheels.set_wheels_speed(speed0 * (i / steps), speed0 * (i / steps))
            if step_dt > 0:
                time.sleep(step_dt)
    except Exception as e:
        print(f"[Project] smooth_stop failed: {e}")
    finally:
        _safe_stop(wheels)


def next_state(current_state: str, event: str) -> str:
    if event == EVENT_TIMEOUT:
        return STATE_STOPPING
    if event == EVENT_STOP_SIGN:
        return STATE_STOPPING
    if event == EVENT_SLOW_SIGN:
        return STATE_SLOW if current_state != STATE_STOPPED else STATE_STOPPED
    # EVENT_NORMAL
    if current_state == STATE_SLOW:
        return STATE_CRUISING
    if current_state == STATE_STOPPED:
        return STATE_STOPPED
    return current_state


# ---------------------------------------------------------------------------
# Sign classification  — uses the canonical resolver from sign_behavior_config
# ---------------------------------------------------------------------------

def _classify_tags(confirmed_ids: List[int]) -> str:
    """Resolve confirmed tag IDs to a sign event using the canonical tag map."""
    for tid in confirmed_ids:
        canonical = resolve_tag(tid)
        if canonical == _TagID.STOP:
            return EVENT_STOP_SIGN
        elif canonical == _TagID.YIELD:
            return EVENT_SLOW_SIGN
    return EVENT_NORMAL


def detect_sign_event(confirmed_ids: List[int], cfg: Dict[str, Any]) -> str:
    """
    Debounce confirmed IDs into a sign event.
    Requires sign_confirm_frames consecutive confirmed detections before
    firing, then locks in the event for sign_cooldown_s seconds so the
    same sign does not retrigger immediately.
    """
    confirm_frames = max(1, int(cfg.get("sign_confirm_frames", 2)))
    cooldown_s     = max(0.0, float(cfg.get("sign_cooldown_s", 3.0)))

    raw_event = _classify_tags(confirmed_ids)

    now = time.time()
    if now < float(_sign_runtime["active_until"]):
        return str(_sign_runtime["candidate_event"])

    if raw_event == EVENT_NORMAL:
        _sign_runtime["candidate_event"] = EVENT_NORMAL
        _sign_runtime["candidate_count"] = 0
        return EVENT_NORMAL

    if _sign_runtime["candidate_event"] == raw_event:
        _sign_runtime["candidate_count"] = int(_sign_runtime["candidate_count"]) + 1
    else:
        _sign_runtime["candidate_event"] = raw_event
        _sign_runtime["candidate_count"] = 1

    if int(_sign_runtime["candidate_count"]) >= confirm_frames:
        _sign_runtime["active_until"] = now + cooldown_s
        return raw_event
    return EVENT_NORMAL


# ---------------------------------------------------------------------------
# Leader-distance estimation (follower uses tag apparent size as proxy)
# ---------------------------------------------------------------------------

def _corner_size_ratio(corners, frame_shape: Optional[Tuple[int, int]]) -> Optional[float]:
    if frame_shape is None or corners is None:
        return None
    h, w = frame_shape
    try:
        pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        e1  = np.linalg.norm(pts[0] - pts[1])
        e2  = np.linalg.norm(pts[1] - pts[2])
        return float(0.5 * (e1 + e2) / max(1.0, min(w, h)))
    except Exception:
        return None


def estimate_leader_distance_signal(
    raw_tags: List[dict],
    frame_shape: Optional[Tuple[int, int]],
    cfg: Dict[str, Any],
) -> Tuple[Optional[float], Optional[int]]:
    leader_ids = set(int(x) for x in cfg.get("leader_tag_ids", []))
    if not raw_tags or not leader_ids:
        return None, None

    best_ratio, best_id = None, None
    for tag in raw_tags:
        try:
            tid = int(tag["tag_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if tid not in leader_ids:
            continue
        ratio = _corner_size_ratio(tag.get("corners"), frame_shape)
        if ratio is None:
            continue
        if best_ratio is None or ratio > best_ratio:
            best_ratio, best_id = ratio, tid
    return best_ratio, best_id


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _read_camera(camera):
    """Return (bgr_frame, rgb_frame) or (None, None)."""
    if camera is None:
        return None, None
    try:
        ok, frame = camera.read()
    except Exception:
        return None, None
    if not ok or frame is None:
        return None, None
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        rgb = None
    return frame, rgb


def get_tag_detector_backend() -> str:
    try:
        cv2.aruco  # noqa: B018
        return "aruco"
    except AttributeError:
        return "raw"


# ---------------------------------------------------------------------------
# Lane agent helpers
# ---------------------------------------------------------------------------

def _get_lane_agent() -> LaneServoingAgent:
    global _lane_agent
    if _lane_agent is None:
        _lane_agent = LaneServoingAgent()
    return _lane_agent


def _scale_drive_commands(left: float, right: float, target_speed: float) -> Tuple[float, float]:
    scale = float(target_speed) / max(1e-6, max(float(left), float(right)))
    return min(1.0, float(left) * scale), min(1.0, float(right) * scale)


def _publish_viz(
    frame_bgr: Any,
    debug_info: Optional[Dict[str, Any]],
    event: str,
    tag_ids: List[int],
    pwm_left: float = 0.0,
    pwm_right: float = 0.0,
) -> None:
    if debug_info is None:
        set_viz_snapshot(frame_bgr)
        return
    merged = dict(debug_info)
    merged["sign_type"] = (
        "STOP" if event == EVENT_STOP_SIGN else
        "SLOW" if event == EVENT_SLOW_SIGN else
        "NONE"
    )
    merged["tag_ids"]   = list(tag_ids)
    merged["pwm_left"]  = float(pwm_left)
    merged["pwm_right"] = float(pwm_right)
    set_viz_snapshot(frame_bgr, merged)


# ---------------------------------------------------------------------------
# Leader loop
# ---------------------------------------------------------------------------

def run_leader(camera, wheels, leds, stop_event, cfg: Dict[str, Any]) -> None:
    loop_hz      = max(1.0, float(cfg.get("loop_hz", 20)))
    dt           = 1.0 / loop_hz
    status_hz    = max(1.0, float(cfg.get("status_publish_hz", 10)))
    status_dt    = 1.0 / status_hz
    cruise_speed = float(cfg.get("cruise_speed", 0.2))
    slow_speed   = float(cfg.get("slow_speed", 0.12))
    stop_hold_s  = float(cfg.get("stop_hold_s", 2.0))
    decel_time_s = float(cfg.get("decel_time_s", 0.8))
    decel_steps  = int(cfg.get("decel_steps", 8))
    print(f"[Project][Leader] FSM loop started at {loop_hz:.1f} Hz.")

    last_log        = 0.0
    last_status_pub = 0.0
    state           = STATE_CRUISING
    current_speed   = cruise_speed
    stop_until      = 0.0
    last_event      = EVENT_NORMAL
    last_tag_ids: List[int] = []

    while not stop_event.is_set():
        now = time.time()
        frame_bgr, frame_rgb = _read_camera(camera)
        frame_shape = None if frame_bgr is None else frame_bgr.shape[:2]

        # Tag detection: aruco fast-path in simulation, raw fallback on hardware
        if frame_rgb is not None:
            raw_tags     = _at_detect(_tag_buf_leader, frame_rgb)
            confirmed_ids = _at_confirm(_tag_buf_leader, raw_tags)
        else:
            raw_tags, confirmed_ids = [], []

        last_tag_ids = list(confirmed_ids)

        # If the stop timer just expired, flush the sign runtime BEFORE
        # classifying this frame — otherwise the still-visible sign immediately
        # re-triggers STATE_STOPPING and the bot never resumes.
        # Use a short fixed cooldown (1.5 s) so the bot clears the stop sign
        # visually without blocking the next different sign (yield etc.).
        if state == STATE_STOPPED and now >= stop_until:
            _sign_runtime["active_until"]    = now + 1.5
            _sign_runtime["candidate_event"] = EVENT_NORMAL
            _sign_runtime["candidate_count"] = 0

        event      = detect_sign_event(confirmed_ids, cfg)
        last_event = event

        if now >= stop_until:
            proposed = next_state(state, event)
        else:
            proposed = state

        if proposed != state:
            print(f"[Project][Leader] transition {state} -> {proposed} ({event})", flush=True)
            state = proposed

        if state == STATE_STOPPING:
            smooth_stop(wheels, current_speed, decel_time_s, decel_steps, stop_event)
            current_speed = 0.0
            state         = STATE_STOPPED
            stop_until    = time.time() + stop_hold_s

        elif state == STATE_STOPPED:
            current_speed = 0.0
            _safe_stop(wheels)
            if now >= stop_until:
                # Timer done — resume unconditionally.
                # Reset sign runtime so the same sign (still visible) doesn't
                # immediately re-trigger a new stop.
                cooldown_s = float(cfg.get("sign_cooldown_s", 4.0))
                _sign_runtime["active_until"]    = now + cooldown_s
                _sign_runtime["candidate_event"] = EVENT_NORMAL
                _sign_runtime["candidate_count"] = 0
                state = STATE_CRUISING

        elif state == STATE_SLOW:
            current_speed = max(0.0, slow_speed)
            if wheels is not None and frame_bgr is not None:
                la = _get_lane_agent()
                left, right = la.compute_commands(frame_bgr, bgr_input=True)
                cmd_l, cmd_r = _scale_drive_commands(left, right, current_speed)
                _publish_viz(frame_bgr, la.last_debug_info, last_event, last_tag_ids, cmd_l, cmd_r)
                wheels.set_wheels_speed(cmd_l, cmd_r)
            elif wheels is not None:
                wheels.set_wheels_speed(current_speed, current_speed)

        else:  # CRUISING
            state         = STATE_CRUISING
            current_speed = max(0.0, cruise_speed)
            if wheels is not None and frame_bgr is not None:
                la = _get_lane_agent()
                left, right = la.compute_commands(frame_bgr, bgr_input=True)
                cmd_l, cmd_r = _scale_drive_commands(left, right, current_speed)
                _publish_viz(frame_bgr, la.last_debug_info, last_event, last_tag_ids, cmd_l, cmd_r)
                wheels.set_wheels_speed(cmd_l, cmd_r)
            elif wheels is not None:
                wheels.set_wheels_speed(current_speed, current_speed)

        if frame_bgr is not None and state in (STATE_STOPPED, STATE_STOPPING):
            _publish_viz(frame_bgr, _get_lane_agent().last_debug_info, last_event, last_tag_ids, 0.0, 0.0)

        if now - last_status_pub >= status_dt:
            payload = build_status_payload(state, current_speed)
            payload.update({"event": last_event, "tag_ids": last_tag_ids})
            set_leader_status(payload)
            last_status_pub = now

        if now - last_log >= 2.0:
            print(
                f"[Project][Leader] state={state} speed={current_speed:.2f} "
                f"event={last_event} tags={last_tag_ids}",
                flush=True,
            )
            last_log = now
        time.sleep(dt)

    set_leader_status(build_status_payload(STATE_STOPPED, 0.0))
    _safe_stop(wheels)
    print("[Project][Leader] Stopped.", flush=True)


# ---------------------------------------------------------------------------
# Follower loop
# ---------------------------------------------------------------------------

def run_follower(camera, wheels, leds, stop_event, cfg: Dict[str, Any]) -> None:
    loop_hz           = max(1.0, float(cfg.get("loop_hz", 20)))
    dt                = 1.0 / loop_hz
    poll_hz           = max(1.0, float(cfg.get("status_poll_hz", 10)))
    poll_dt           = 1.0 / poll_hz
    request_timeout_s = float(cfg.get("request_timeout_s", 0.2))
    leader_timeout_s  = float(cfg.get("leader_timeout_s", 0.4))
    leader_host       = str(cfg.get("leader_host", "127.0.0.1")).strip()
    # Allow convoy_sim.py to inject the actual leader port at runtime so the
    # follower always reaches the right server even if the default port was busy.
    leader_port       = int(os.environ.get("CONVOY_LEADER_PORT", "") or cfg.get("leader_port", 5055))
    cruise_speed      = float(cfg.get("cruise_speed", 0.2))
    slow_speed        = float(cfg.get("slow_speed", 0.12))
    follower_max_speed = float(cfg.get("follower_max_speed", 0.2))
    follower_min_speed = float(cfg.get("follower_min_speed", 0.0))
    distance_target   = float(cfg.get("distance_target", 0.06))
    distance_kp       = float(cfg.get("distance_kp", 0.6))
    decel_time_s      = float(cfg.get("decel_time_s", 0.8))
    decel_steps       = int(cfg.get("decel_steps", 8))
    status_url        = f"http://{leader_host}:{leader_port}/convoy/status"
    leader_tag_ids    = set(int(x) for x in cfg.get("leader_tag_ids", []))
    print(f"[Project][Follower] Polling loop started at {loop_hz:.1f} Hz.")

    last_log        = 0.0
    last_poll       = 0.0
    latest          = build_status_payload(STATE_STOPPED, 0.0)
    mode            = EVENT_TIMEOUT
    target_speed    = 0.0
    commanded_speed = 0.0
    prev_mode       = None
    last_leader_tag_id = None

    while not stop_event.is_set():
        now = time.time()
        distance_signal = None
        leader_tag_id   = None

        frame_bgr, frame_rgb = _read_camera(camera)
        frame_shape = None if frame_bgr is None else frame_bgr.shape[:2]

        if leader_tag_ids and frame_rgb is not None:
            raw_tags = _at_detect(_tag_buf_follower, frame_rgb)
            distance_signal, leader_tag_id = estimate_leader_distance_signal(
                raw_tags, frame_shape, cfg
            )
            if leader_tag_id is not None:
                last_leader_tag_id = leader_tag_id

        if now - last_poll >= poll_dt:
            try:
                resp = requests.get(status_url, timeout=request_timeout_s)
                if resp.ok:
                    data = resp.json()
                    latest = {
                        "state": str(data.get("state", "STOPPED")).upper(),
                        "speed": float(data.get("speed", 0.0)),
                        "ts":    float(data.get("ts", 0.0)),
                    }
            except Exception:
                pass
            last_poll = now

        is_stale = (now - float(latest.get("ts", 0.0))) > leader_timeout_s
        state    = str(latest.get("state", "STOPPED")).upper()

        if is_stale:
            mode, target_speed = EVENT_TIMEOUT, 0.0
        elif state == STATE_STOPPED:
            mode, target_speed = STATE_STOPPED, 0.0
        elif state == STATE_SLOW:
            mode, target_speed = STATE_SLOW, min(slow_speed, follower_max_speed)
        else:
            mode, target_speed = STATE_CRUISING, min(cruise_speed, follower_max_speed)

        if mode in (STATE_CRUISING, STATE_SLOW) and distance_signal is not None:
            distance_error = distance_target - float(distance_signal)
            target_speed  += distance_kp * distance_error
            target_speed   = max(follower_min_speed, min(follower_max_speed, target_speed))

        if mode == EVENT_TIMEOUT:
            commanded_speed = 0.0
            _safe_stop(wheels)
        elif mode == STATE_STOPPED:
            if commanded_speed > 0.0:
                smooth_stop(wheels, commanded_speed, decel_time_s, decel_steps, stop_event)
            commanded_speed = 0.0
            # Actively hold wheels at zero every cycle so the bot doesn't drift.
            _safe_stop(wheels)
        else:
            commanded_speed = target_speed
            if wheels is not None and frame_bgr is not None:
                la = _get_lane_agent()
                left, right  = la.compute_commands(frame_bgr, bgr_input=True)
                cmd_l, cmd_r = _scale_drive_commands(left, right, commanded_speed)
                leader = get_leader_status()
                _publish_viz(
                    frame_bgr,
                    la.last_debug_info,
                    str(leader.get("event", EVENT_NORMAL)),
                    list(leader.get("tag_ids", [])),
                    cmd_l, cmd_r,
                )
                wheels.set_wheels_speed(cmd_l, cmd_r)
            elif wheels is not None:
                wheels.set_wheels_speed(commanded_speed, commanded_speed)

        if mode != prev_mode:
            print(f"[Project][Follower] transition {prev_mode} -> {mode}", flush=True)
            prev_mode = mode

        if now - last_log >= 2.0:
            age = now - float(latest.get("ts", 0.0))
            print(
                f"[Project][Follower] mode={mode} target={target_speed:.2f} "
                f"cmd={commanded_speed:.2f} leader={state} age={age:.2f}s "
                f"dist={distance_signal} ltag={last_leader_tag_id}",
                flush=True,
            )
            last_log = now
        time.sleep(dt)

    _safe_stop(wheels)
    print("[Project][Follower] Stopped.", flush=True)


# ---------------------------------------------------------------------------
# HSV tuning pass-through (used by web UI sliders)
# ---------------------------------------------------------------------------

def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper):
    try:
        from tasks.project.packages import visual_servoing_activity as mod
        mod.set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper)
    except Exception:
        pass


def get_hsv_bounds():
    try:
        from tasks.project.packages import visual_servoing_activity as mod
        return mod.get_hsv_bounds()
    except Exception:
        return {}


def build_project_lane_agent() -> LaneServoingAgent:
    global _lane_agent
    sim_cfg = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "project_lane_sim.yaml")
    )
    use_sim = "sim" in _config_path().lower()
    if use_sim and os.path.isfile(sim_cfg):
        _lane_agent = LaneServoingAgent(config_path=sim_cfg)
    else:
        _lane_agent = LaneServoingAgent()
        _lane_agent.base_speed  = 0.48
        _lane_agent.curve_speed = 0.40
        _lane_agent.d_gain      = 0.79
        _lane_agent.p_gain      = 0.16
        _lane_agent.detection_threshold = 100
    return _lane_agent


def main(camera, wheels, leds, stop_event):
    cfg  = load_config()
    role = cfg.get("role", "leader")
    print(f"[Project] Loaded config from {_config_path()}")
    print(f"[Project] Role: {role}")

    if role == "follower":
        run_follower(camera, wheels, leds, stop_event, cfg)
    else:
        run_leader(camera, wheels, leds, stop_event, cfg)
