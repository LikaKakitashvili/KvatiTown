"""Launch leader + follower virtual servers against one Godot convoy scene."""

import os
import platform
import subprocess
import sys
import tempfile
import time

from launcher.ports import find_available_port, wait_for_port_file

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEADER_CONFIG = os.path.join(PROJECT_ROOT, "config", "project_config_sim.yaml")
FOLLOWER_CONFIG = os.path.join(PROJECT_ROOT, "config", "project_config_sim_follower.yaml")
VIRTUAL_SERVER = os.path.join(PROJECT_ROOT, "servers", "project", "virtual_server.py")


def _start_server(
    config_path: str,
    web_port: int,
    frame_port: int,
    wheel_port: int,
    extra_env: dict | None = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["PROJECT_CONFIG"] = config_path
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        [
            sys.executable,
            VIRTUAL_SERVER,
            "--port",
            str(web_port),
            "--frame-port",
            str(frame_port),
            "--wheel-port",
            str(wheel_port),
        ],
        cwd=PROJECT_ROOT,
        env=env,
    )


def run_convoy_simulation(args, launch_godot_fn, stop_godot_fn, godot_scene: str) -> int:
    print("\n" + "=" * 60)
    print("RUN CONVOY SIMULATION (LEADER + FOLLOWER)")
    print("=" * 60)

    used_ports: set[int] = set()

    leader_cam = find_available_port(5001, exclude=used_ports)
    used_ports.add(leader_cam)
    leader_wheel_hint = find_available_port(5002, exclude=used_ports)
    used_ports.add(leader_wheel_hint)

    follower_cam = find_available_port(5011, exclude=used_ports)
    used_ports.add(follower_cam)
    follower_wheel_hint = find_available_port(5012, exclude=used_ports)
    used_ports.add(follower_wheel_hint)

    leader_web = find_available_port(args.port, exclude=used_ports)
    used_ports.add(leader_web)
    follower_web = find_available_port(5055, exclude=used_ports)
    used_ports.add(follower_web)

    print(f"  Leader  camera={leader_cam} wheel_hint={leader_wheel_hint} web={leader_web}")
    print(f"  Follower camera={follower_cam} wheel_hint={follower_wheel_hint} web={follower_web}")

    tmp_dir = tempfile.mkdtemp(prefix="ducky_convoy_")
    port_file_path = os.path.join(tmp_dir, "ports.json")

    leader_proc = None
    follower_proc = None

    try:
        print("\n[1/4] Launching Godot convoy scene...")
        if not launch_godot_fn(
            args.godot_path,
            args.debug,
            camera_port=leader_cam,
            wheel_port_hint=leader_wheel_hint,
            port_file_path=port_file_path,
            scene=godot_scene,
            follower_camera_port=follower_cam,
            follower_wheel_port=follower_wheel_hint,
        ):
            return 1

        godot_init_timeout = 60 if platform.system() == "Darwin" else 20
        print(f"Waiting for Godot wheel ports (timeout: {godot_init_timeout}s)...")
        try:
            port_data = wait_for_port_file(
                port_file_path,
                timeout=godot_init_timeout,
                required_keys=["wheel_port", "follower_wheel_port"],
            )
            leader_wheel = int(port_data.get("wheel_port", leader_wheel_hint))
            follower_wheel = int(port_data.get("follower_wheel_port", follower_wheel_hint))
            print(f"  Leader wheel port:   {leader_wheel}")
            print(f"  Follower wheel port: {follower_wheel}")
        except TimeoutError as exc:
            print(f"❌ ERROR: {exc}")
            return 1

        print("\n[2/4] Starting leader virtual server...")
        leader_proc = _start_server(LEADER_CONFIG, leader_web, leader_cam, leader_wheel)
        time.sleep(2.0)

        print("[3/4] Starting follower virtual server...")
        # Tell the follower the exact port the leader web server is on so
        # the convoy/status poll URL is always correct regardless of port availability.
        follower_proc = _start_server(
            FOLLOWER_CONFIG, follower_web, follower_cam, follower_wheel,
            extra_env={"CONVOY_LEADER_PORT": str(leader_web)},
        )
        time.sleep(1.0)

        print("\n[4/4] Convoy simulation running")
        print("=" * 60)
        print(f"Leader UI:   http://localhost:{leader_web}")
        print(f"Follower UI: http://localhost:{follower_web}")
        print(f"Leader status API: http://localhost:{leader_web}/convoy/status")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        while True:
            if leader_proc.poll() is not None:
                print("Leader server exited unexpectedly.")
                return 1
            if follower_proc.poll() is not None:
                print("Follower server exited unexpectedly.")
                return 1
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping convoy simulation...")
        return 0
    finally:
        for proc in (leader_proc, follower_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        stop_godot_fn()
        try:
            os.remove(port_file_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass
