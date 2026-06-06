import argparse
import os
import signal
import socket
import sys
import threading
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)

from flask import Flask, Response, jsonify, render_template_string, request
import cv2
import numpy as np
import yaml

from duckiebot.camera_driver.godot_camera_driver import GodotCameraConfig, GodotCameraDriver
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from launcher.ports import find_available_port
from servers.common import shutdown_cleanup, suppress_http_logs
from servers.project.visualization import create_lane_visualization
from servers.templates.lane_servoing import LANE_SERVOING_TEMPLATE as HTML_TEMPLATE
import tasks.project.packages.agent as project_agent

SIM_CONFIG = os.path.join(project_root, "config", "project_config_sim.yaml")
LANE_CONFIG_FILE = os.path.join(project_root, "config", "lane_servoing_config.yaml")
LANE_HSV_CONFIG_FILE = os.path.join(project_root, "config", "lane_servoing_hsv_config.yaml")

app = Flask(__name__)
camera = None
wheels = None
agent = None
running = False
stop_event = threading.Event()
_agent_thread = None


def _ensure_sim_config():
    if os.environ.get("PROJECT_CONFIG", "").strip():
        return
    os.environ.setdefault("PROJECT_CONFIG", SIM_CONFIG)


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


def _visualize(frame_bgr):
    if frame_bgr is None:
        return _error_frame("Waiting for first frame from agent...")

    _, debug_info = project_agent.get_viz_snapshot()
    if debug_info is None:
        debug_info = getattr(agent, "last_debug_info", None)

    pwm_left = 0.0
    pwm_right = 0.0
    if isinstance(debug_info, dict):
        pwm_left = float(debug_info.get("pwm_left", 0.0))
        pwm_right = float(debug_info.get("pwm_right", 0.0))

    if debug_info is None:
        return frame_bgr
    try:
        return create_lane_visualization(frame_bgr, debug_info, pwm_left, pwm_right)
    except Exception as e:
        print(f"[Project][Sim] visualize error: {e}")
        return frame_bgr


def _generate_project_frames():
    while True:
        try:
            frame_bgr, _ = project_agent.get_viz_snapshot()
            if frame_bgr is None:
                display = _error_frame("Waiting for agent loop...\nPress Start in the UI if stopped.")
            else:
                display = _visualize(frame_bgr)

            leader = project_agent.get_leader_status()
            cv2.putText(
                display,
                f"Convoy: {leader.get('state', 'STOPPED')}  speed={float(leader.get('speed', 0.0)):.2f}",
                (10, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            ret, jpeg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 50])
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
        except Exception as e:
            print(f"[Project][Sim][VideoStream] Error: {e}")
            time.sleep(0.05)


def _project_agent_loop():
    global running
    if camera is None:
        print("[Project][Sim] Agent loop skipped: camera unavailable")
        return
    running = True
    try:
        project_agent.main(camera, wheels, None, stop_event)
    except Exception as e:
        print(f"[Project][Sim] Agent loop crashed: {e}")
    finally:
        running = False


def _start_agent_thread():
    global _agent_thread
    if camera is None:
        return
    if _agent_thread is not None and _agent_thread.is_alive():
        return
    stop_event.clear()
    _agent_thread = threading.Thread(
        target=_project_agent_loop,
        daemon=True,
        name="ProjectSimAgentLoop",
    )
    _agent_thread.start()


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, config=agent, hostname=socket.gethostname())


@app.route("/video")
def video():
    return Response(_generate_project_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start", methods=["POST"])
def start():
    global running
    _start_agent_thread()
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
    alive = _agent_thread is not None and _agent_thread.is_alive()
    return jsonify({"running": running and alive})


@app.route("/reset", methods=["POST"])
def reset():
    if hasattr(wheels, "reset_game"):
        wheels.reset_game()
    stop_event.clear()
    _start_agent_thread()
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
        print(f"[Project][Sim] Could not save config: {e}")
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
        print(f"[Project][Sim] Could not save HSV config: {e}")
    return jsonify({"status": "ok"})


@app.route("/status")
def status():
    if agent is None:
        return jsonify({"status": "not_initialized"})
    leader = project_agent.get_leader_status()
    _, debug_info = project_agent.get_viz_snapshot()
    if debug_info is None:
        debug_info = getattr(agent, "last_debug_info", None)
    frame_count = int(debug_info.get("frame_count", 0)) if isinstance(debug_info, dict) else 0
    cfg = project_agent.load_config()
    return jsonify(
        {
            "status": "active",
            "camera_ready": camera is not None,
            "running": running,
            "frame_count": frame_count,
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
    global camera, wheels, agent

    _ensure_sim_config()

    ap = argparse.ArgumentParser(description="Project Server — Godot Simulation (Convoy + Signs)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--frame-port", type=int, default=5001)
    ap.add_argument("--wheel-port", type=int, default=5002)
    ap.add_argument("--godot-host", type=str, default="localhost")
    args = ap.parse_args()

    suppress_http_logs()
    cfg = project_agent.load_config()
    print("=" * 60)
    print(f"PROJECT SERVER — SIMULATION ({cfg.get('role', 'leader').upper()})")
    print("=" * 60)
    print(f"Role: {cfg.get('role', 'leader')}")
    print(f"Config: {os.environ.get('PROJECT_CONFIG', 'default')}")
    print(f"Stop tag IDs: {cfg.get('stop_tag_ids', [])}")
    print(f"Slow tag IDs: {cfg.get('slow_tag_ids', [])}")

    print("\n[1/3] Initializing wheels driver...")
    wheels = GodotWheelsDriver(
        WheelPWMConfiguration(pwm_min=0),
        WheelPWMConfiguration(pwm_min=0),
        godot_host=args.godot_host,
        godot_port=args.wheel_port,
    )
    wheels.trim = 0
    print(f"  Wheels: {args.godot_host}:{args.wheel_port}")

    print("\n[2/3] Initializing camera driver...")
    print(f"  Waiting for Godot on port {args.frame_port}...")
    camera = GodotCameraDriver(godot_config=GodotCameraConfig(host="0.0.0.0", port=args.frame_port))
    camera.start()
    print("  Camera: connected")

    print("\n[3/3] Creating project lane agent...")
    agent = project_agent.build_project_lane_agent()
    print("  Agent: ready")

    def _shutdown(signum, frame):
        print("\nShutting down simulation server...")
        shutdown_cleanup(wheels, camera, stop_event)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    web_port = find_available_port(args.port)
    if web_port != args.port:
        print(f"  Port {args.port} busy, using {web_port}")

    print("\n" + "=" * 60)
    print(f"Web Interface: http://localhost:{web_port}")
    print("Convoy status: http://localhost:{}/convoy/status".format(web_port))
    print("=" * 60 + "\n")

    _start_agent_thread()

    try:
        app.run(host="127.0.0.1", port=web_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == "__main__":
    sys.exit(main())
