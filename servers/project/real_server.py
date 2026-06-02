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
from servers.templates.lane_servoing import LANE_SERVOING_TEMPLATE as HTML_TEMPLATE
import tasks.project.packages.agent as project_agent

from duckiebot.camera_driver import CameraDriver
from duckiebot.wheel_driver import DaguWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from duckiebot.led_driver import LEDDriver
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs

LANE_CONFIG_FILE = os.path.join(project_root, "config", "lane_servoing_config.yaml")
LANE_HSV_CONFIG_FILE = os.path.join(project_root, "config", "lane_servoing_hsv_config.yaml")

app = Flask(__name__)
camera = None
wheels = None
leds = None
agent = None
running = True
stop_event = threading.Event()
_state_lock = threading.Lock()
_last_pwm = (0.0, 0.0)
_last_debug_info = None
_control_thread_started = False


def _visualize(frame):
    global running
    if frame is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for camera...", (160, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        return blank
    if agent is None:
        return frame

    # Rendering-only: wheel control happens in the background control thread.
    with _state_lock:
        pwm_left, pwm_right = _last_pwm
        debug_info = _last_debug_info or getattr(agent, "last_debug_info", None)

    # Frame is BGR from real CameraDriver when rgb=False generator is used.
    bgr = frame
    try:
        if debug_info is None:
            return bgr
        return create_lane_visualization(bgr, debug_info, pwm_left, pwm_right)
    except Exception as e:
        print(f"[Project] visualize error: {e}")
        cv2.putText(bgr, str(e)[:80], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return bgr


def _control_loop():
    """Continuously run lane controller and drive wheels.

    This makes the robot start moving immediately after task launch,
    even if nobody opens the UI video stream yet.
    """
    global _last_pwm, _last_debug_info

    # Avoid sending commands before motors are ready.
    while not stop_event.is_set() and (camera is None or wheels is None or agent is None):
        time.sleep(0.01)

    while not stop_event.is_set():
        ok, frame_bgr = camera.read()
        if not ok or frame_bgr is None:
            # Keep it safe on camera failure.
            with _state_lock:
                _last_pwm = (0.0, 0.0)
            if wheels is not None:
                wheels.set_wheels_speed(0.0, 0.0)
            continue

        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pwm_left, pwm_right = agent.compute_commands(frame_rgb)
        except Exception as e:
            print(f"[Project] control_loop compute error: {e}")
            pwm_left, pwm_right = 0.0, 0.0

        with _state_lock:
            _last_pwm = (pwm_left, pwm_right)
            _last_debug_info = getattr(agent, "last_debug_info", None)

        if running and wheels is not None:
            wheels.set_wheels_speed(pwm_left, pwm_right)
        elif wheels is not None:
            wheels.set_wheels_speed(0.0, 0.0)


generate_frames = make_frame_generator(lambda: camera, _visualize, quality=50, rgb=False)


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, config=agent, hostname=socket.gethostname())


@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/shutdown")
def shutdown():
    shutdown_cleanup(wheels, camera, stop_event)
    return jsonify({"status": "ok"})


@app.route("/start", methods=["POST"])
def start():
    global running
    running = True
    return jsonify({"status": "running"})


@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
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
        with open(LANE_HSV_CONFIG_FILE, "w") as f:
            yaml.dump(current, f, default_flow_style=False)
    except Exception as e:
        print(f"[Project] Could not save HSV config: {e}")
    return jsonify({"status": "ok"})


@app.route("/status")
def status():
    if agent is None:
        return jsonify({"status": "not_initialized"})
    return jsonify(
        {
            "status": "active",
            "frame_count": agent.frame_count,
            "sign_state": getattr(agent, "sign_state", "unknown"),
            "config": {
                "p_gain": agent.p_gain,
                "d_gain": agent.d_gain,
                "base_speed": agent.base_speed,
                "detection_threshold": agent.detection_threshold,
            },
        }
    )


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
    camera = CameraDriver()
    camera.start()
    print("  Camera: ok")

    print("\n[4/4] Initializing project lane agent...")
    agent = project_agent.build_project_lane_agent()
    print("  Agent: ready")

    def _start_control_thread():
        global _control_thread_started
        if _control_thread_started:
            return
        _control_thread_started = True
        threading.Thread(target=_control_loop, daemon=True, name="ProjectControlLoop").start()

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

    _start_control_thread()

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
