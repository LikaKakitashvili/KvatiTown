import sys
import os
import signal
import threading
import argparse
import socket
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)

from flask import Flask, Response, jsonify, request, render_template_string
import cv2
import numpy as np
import yaml

from servers.project.visualization import create_lane_visualization
from servers.templates.convoy import CONVOY_TEMPLATE as HTML_TEMPLATE
import tasks.project.packages.agent as project_agent

from duckiebot.camera_driver import CameraDriver
from duckiebot.wheel_driver import DaguWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from duckiebot.led_driver import LEDDriver
from launcher.ports import find_available_port
from servers.common import shutdown_cleanup, suppress_http_logs


def _configure_line_buffered_stdio():
    """Flush prints immediately when stdout is piped (bot dashboard log panel)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass


_configure_line_buffered_stdio()

LANE_CONFIG_FILE = os.path.join(project_root, "config", "lane_servoing_config.yaml")
LANE_HSV_CONFIG_FILE = os.path.join(project_root, "config", "lane_servoing_hsv_config.yaml")

app = Flask(__name__)
camera = None
wheels = None
leds = None
agent = None
running = True
stop_event = threading.Event()
_agent_thread = None
_camera_ready = False
_camera_start_lock = threading.Lock()


def _stop_camera():
    global camera, _camera_ready
    if camera is not None:
        try:
            camera.stop()
        except Exception:
            pass
    camera = None
    _camera_ready = False
    time.sleep(2.0)


def _try_start_camera(max_attempts: int = 1) -> bool:
    """Start camera with retries; return True on success."""
    global camera, _camera_ready

    with _camera_start_lock:
        if _camera_ready and camera is not None:
            return True

        _stop_camera()

        for attempt in range(1, max_attempts + 1):
            try:
                cam = CameraDriver()
                cam.start()
                camera = cam
                _camera_ready = True
                print("  Camera: ok")
                return True
            except Exception as e:
                print(f"  Camera attempt {attempt}/{max_attempts} failed: {e}")
                _stop_camera()
                if attempt < max_attempts:
                    time.sleep(5.0)
        return False


def _start_agent_thread():
    global _agent_thread
    if not _camera_ready or camera is None:
        return
    if _agent_thread is not None and _agent_thread.is_alive():
        return
    _agent_thread = threading.Thread(target=_project_agent_loop, daemon=True, name="ProjectAgentLoop")
    _agent_thread.start()


def _camera_retry_loop():
    """Keep trying camera init in background so UI can still be opened."""
    while not stop_event.is_set():
        if _camera_ready and camera is not None:
            time.sleep(2.0)
            continue
        print("[Project] Camera retry (waiting for nvargus)...")
        if _try_start_camera(max_attempts=1):
            _start_agent_thread()
        time.sleep(8.0)


def _error_frame(message: str):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    for i, line in enumerate(message.split("\n")[:6]):
        cv2.putText(
            blank,
            line[:70],
            (20, 80 + i * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    return blank


def _visualize(frame):
    if frame is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for camera...", (160, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        return blank
    if agent is None:
        return frame

    frame_bgr, debug_info = project_agent.get_viz_snapshot()
    if frame_bgr is None:
        frame_bgr = frame
    if debug_info is None:
        debug_info = getattr(agent, "last_debug_info", None)

    pwm_left = 0.0
    pwm_right = 0.0
    if isinstance(debug_info, dict):
        pwm_left = float(debug_info.get("pwm_left", 0.0))
        pwm_right = float(debug_info.get("pwm_right", 0.0))

    try:
        if debug_info is None:
            return frame_bgr
        return create_lane_visualization(frame_bgr, debug_info, pwm_left, pwm_right)
    except Exception as e:
        print(f"[Project] visualize error: {e}")
        cv2.putText(frame_bgr, str(e)[:80], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame_bgr


def _generate_project_frames():
    """MJPEG stream from agent-published snapshots (no second camera.read)."""
    while True:
        try:
            if not _camera_ready:
                display = _error_frame(
                    "Camera not ready.\n"
                    "1) Stop other tasks using the camera\n"
                    "2) On bot: sudo systemctl restart nvargus-daemon\n"
                    "3) Wait ~10s, refresh this page"
                )
            else:
                frame_bgr, _ = project_agent.get_viz_snapshot()
                if frame_bgr is None:
                    display = _error_frame("Waiting for first frame from agent...")
                else:
                    display = _visualize(frame_bgr)
            ret, jpeg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 50])
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
        except Exception as e:
            print(f"[Project][VideoStream] Error: {e}")
            time.sleep(0.05)


def _project_agent_loop():
    """Run project task main loop in background thread."""
    global running
    if camera is None:
        print("[Project] Agent loop skipped: camera unavailable")
        return
    running = True
    try:
        project_agent.main(camera, wheels, leds, stop_event)
    except Exception as e:
        print(f"[Project] Agent loop crashed: {e}")
    finally:
        running = False


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, config=agent, hostname=socket.gethostname())


@app.route("/video")
def video():
    return Response(_generate_project_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/shutdown")
def shutdown():
    shutdown_cleanup(wheels, camera, stop_event)
    return jsonify({"status": "ok"})


@app.route("/start", methods=["POST"])
def start():
    global _agent_thread, running
    if _agent_thread is None or not _agent_thread.is_alive():
        return jsonify({"status": "stopped", "message": "restart task process to start again"})
    running = True
    return jsonify({"status": "running"})


@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    stop_event.set()
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)
    return jsonify({"status": "stopped"})


@app.route("/running")
def get_running():
    return jsonify({"running": running})


@app.route("/reset", methods=["POST"])
def reset():
    if hasattr(wheels, "reset_game"):
        wheels.reset_game()
    if wheels is not None and agent is not None:
        wheels.set_wheels_speed(agent.base_speed, agent.base_speed)
    return jsonify({"status": "ok"})


@app.route("/update_config", methods=["POST"])
def update_config():
    data = request.json or {}
    agent.p_gain = float(data.get("k_d", agent.p_gain))
    agent.d_gain = float(data.get("k_phi", agent.d_gain))
    agent.base_speed = float(data.get("const", agent.base_speed))

    try:
        with open(LANE_CONFIG_FILE, "r") as f:
            saved = yaml.safe_load(f) or {}
        saved["p_gain"] = agent.p_gain
        saved["d_gain"] = agent.d_gain
        saved["base_speed"] = agent.base_speed
        with open(LANE_CONFIG_FILE, "w") as f:
            yaml.dump(saved, f, default_flow_style=False)
    except Exception as e:
        print(f"[Project] Could not save config: {e}")
    return jsonify({"status": "ok"})


@app.route("/get_hsv")
def get_hsv():
    return jsonify(project_agent.get_hsv_bounds())


@app.route("/update_hsv", methods=["POST"])
def update_hsv():
    data = request.json or {}
    current = project_agent.get_hsv_bounds()
    current.update({k: int(v) for k, v in data.items()})
    project_agent.set_hsv_bounds(
        [current["yellow_lower_h"], current["yellow_lower_s"], current["yellow_lower_v"]],
        [current["yellow_upper_h"], current["yellow_upper_s"], current["yellow_upper_v"]],
        [current["white_lower_h"], current["white_lower_s"], current["white_lower_v"]],
        [current["white_upper_h"], current["white_upper_s"], current["white_upper_v"]],
    )
    try:
        from tasks.project.packages import visual_servoing_activity as vsa
        with open(LANE_HSV_CONFIG_FILE, "r") as f:
            saved = yaml.safe_load(f) or {}
        saved.update(current)
        with open(LANE_HSV_CONFIG_FILE, "w") as f:
            yaml.dump(saved, f, default_flow_style=False)
        vsa.reload_hsv_config()
    except Exception as e:
        print(f"[Project] Could not save HSV config: {e}")
    return jsonify({"status": "ok"})


@app.route("/status")
def status():
    if agent is None:
        return jsonify({"status": "not_initialized"})
    leader = project_agent.get_leader_status()
    _, _debug_info = project_agent.get_viz_snapshot()
    if _debug_info is None:
        _debug_info = getattr(agent, "last_debug_info", None)
    _frame_count = int(_debug_info.get("frame_count", 0)) if isinstance(_debug_info, dict) else 0
    cfg = project_agent.load_config()
    return jsonify(
        {
            "status": "active",
            "camera_ready": _camera_ready,
            "running": running,
            "frame_count": _frame_count,
            "role": cfg.get("role", "leader"),
            "convoy_state": leader.get("state", "STOPPED"),
            "convoy_speed": float(leader.get("speed", 0.0)),
            "convoy_ts": float(leader.get("ts", 0.0)),
            "event": leader.get("event", "EVENT_NORMAL"),
            "tag_ids": leader.get("tag_ids", []),
            "config": {
                "p_gain": agent.p_gain,
                "d_gain": agent.d_gain,
                "base_speed": agent.base_speed,
                "detection_threshold": agent.detection_threshold,
            },
        }
    )


@app.route("/convoy/status")
def convoy_status():
    return jsonify(project_agent.get_leader_status())


def main():
    global camera, wheels, leds, agent

    ap = argparse.ArgumentParser(description="Project Server — Real Hardware (Lane UI)")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    suppress_http_logs()
    print("=" * 60)
    print("PROJECT SERVER — REAL HARDWARE (VISUAL LANE UI)")
    print("=" * 60)

    print("\n[1/4] Initializing LED driver...")
    try:
        leds = LEDDriver()
        leds.all_off()
        print("  LEDs: ok")
    except Exception as e:
        print(f"  LEDs: not available ({e})")
        leds = None

    print("\n[2/4] Initializing wheels driver...")
    wheels = DaguWheelsDriver(WheelPWMConfiguration(), WheelPWMConfiguration())
    print("  Wheels: ok")

    print("\n[3/4] Initializing camera driver...")
    if not _try_start_camera(max_attempts=2):
        print("  Camera: unavailable now (server will continue, retrying in background)")

    print("\n[4/4] Initializing project lane agent...")
    agent = project_agent.build_project_lane_agent()
    print("  Agent: ready")

    def _shutdown(signum, frame):
        print("\nShutting down...")
        if leds:
            try:
                leds.all_off()
                leds.release()
            except Exception:
                pass
        shutdown_cleanup(wheels, camera, stop_event)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    web_port = find_available_port(args.port)
    print(f"\nWeb Interface: http://localhost:{web_port}")
    print("Press Ctrl+C to stop\n")

    _start_agent_thread()
    threading.Thread(target=_camera_retry_loop, daemon=True, name="ProjectCameraRetry").start()

    try:
        app.run(host="0.0.0.0", port=web_port, debug=False, threaded=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if leds:
            try:
                leds.all_off()
                leds.release()
            except Exception:
                pass
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == "__main__":
    sys.exit(main())
