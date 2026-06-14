import os
import sys
import threading
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tasks.project.packages import agent


class DummyWheels:
    def __init__(self):
        self.commands = []

    def set_wheels_speed(self, left: float, right: float):
        self.commands.append((left, right))


def _run_and_stop(loop_fn, timeout_s: float = 0.2):
    stop_event = threading.Event()
    wheels = DummyWheels()
    cfg = {"loop_hz": 20}

    t = threading.Thread(
        target=loop_fn,
        args=(None, wheels, None, stop_event, cfg),
        daemon=True,
    )
    t.start()
    time.sleep(timeout_s)
    stop_event.set()
    t.join(timeout=1.0)

    assert not t.is_alive(), f"{loop_fn.__name__} did not exit after stop_event"
    assert wheels.commands, f"{loop_fn.__name__} never sent wheel command"
    assert wheels.commands[-1] == (0.0, 0.0), f"{loop_fn.__name__} final command was not stop"


def test_stop_pass_through_triggers_on_sign_exit():
    agent._stop_pass_runtime.update({"tracking": False, "confirm_count": 0, "cooldown_until": 0.0})
    cfg = {
        "stop_tag_ids": [26],
        "sign_confirm_frames": 2,
        "sign_cooldown_s": 0.0,
        "sign_center_roi": 1.0,
    }

    class Det:
        def __init__(self, tag_id):
            self.tag_id = tag_id
            self.center = (100.0, 100.0)

    visible = [Det(26)]
    shape = (480, 640)

    assert not agent.detect_stop_pass_trigger(visible, shape, cfg)
    assert not agent.detect_stop_pass_trigger(visible, shape, cfg)
    assert agent.is_stop_sign_tracking()
    assert not agent.detect_stop_pass_trigger(visible, shape, cfg)

    assert agent.detect_stop_pass_trigger([], shape, cfg)
    assert not agent.is_stop_sign_tracking()
    print("OK: stop pass-through triggers when sign leaves frame")


def test_next_state_transitions():
    assert agent.next_state(agent.STATE_CRUISING, agent.EVENT_SLOW_SIGN) == agent.STATE_SLOW
    assert agent.next_state(agent.STATE_SLOW, agent.EVENT_NORMAL) == agent.STATE_CRUISING
    assert agent.next_state(agent.STATE_CRUISING, agent.EVENT_STOP_SIGN) == agent.STATE_STOPPING
    assert agent.next_state(agent.STATE_CRUISING, agent.EVENT_TIMEOUT) == agent.STATE_STOPPING
    print("OK: FSM transitions")


def test_smooth_stop():
    wheels = DummyWheels()
    stop_event = threading.Event()
    agent.smooth_stop(wheels, current_speed=0.4, decel_time_s=0.05, decel_steps=4, stop_event=stop_event)
    assert wheels.commands, "smooth_stop produced no wheel commands"
    assert wheels.commands[-1] == (0.0, 0.0), "smooth_stop did not end at full stop"
    print("OK: smooth_stop reaches zero")


def test_convoy_leds():
    class MockLeds:
        def __init__(self):
            self.colors = {}

        def set_rgb(self, idx, color):
            self.colors[idx] = list(color)

        def all_off(self):
            self.colors.clear()

    leds = MockLeds()
    agent._last_led_state = None
    agent.apply_convoy_leds(leds, agent.STATE_CRUISING)
    assert leds.colors[0] == [0.0, 0.0, 0.0]
    agent.apply_convoy_leds(leds, agent.STATE_SLOW)
    assert leds.colors[0] == [1.0, 1.0, 0.0]
    agent.apply_convoy_leds(leds, agent.STATE_STOPPED)
    assert leds.colors[0] == [1.0, 0.0, 0.0]
    agent.apply_convoy_leds(leds, agent.STATE_STOPPED)
    assert len(leds.colors) == 4
    agent._leds_off(leds)
    assert leds.colors == {}
    print("OK: convoy LEDs")


def test_vision_speed_cap():
    from tasks.project.packages.leader_detection import speed_cap_from_vision

    cfg = {"vision_stop_ratio": 0.12, "vision_slow_ratio": 0.06}
    assert speed_cap_from_vision(0.03, cfg, 0.5) == 0.5
    assert speed_cap_from_vision(0.12, cfg, 0.5) == 0.0
    mid = speed_cap_from_vision(0.09, cfg, 0.5)
    assert 0.0 < mid < 0.5
    print("OK: vision speed cap")


def test_config_loads():
    cfg = agent.load_config()
    assert isinstance(cfg, dict), "Config must be a dictionary"
    assert "role" in cfg, "Config missing role"
    assert "loop_hz" in cfg, "Config missing loop_hz"
    print("OK: config loads")


def test_role_dispatch():
    called = {"leader": 0, "follower": 0}
    original_load = agent.load_config
    original_leader = agent.run_leader
    original_follower = agent.run_follower

    def fake_leader(*args, **kwargs):
        called["leader"] += 1

    def fake_follower(*args, **kwargs):
        called["follower"] += 1

    try:
        agent.run_leader = fake_leader
        agent.run_follower = fake_follower

        agent.load_config = lambda: {"role": "leader", "loop_hz": 20}
        agent.main(None, None, None, threading.Event())
        assert called["leader"] == 1 and called["follower"] == 0, "Leader dispatch failed"

        called["leader"] = 0
        called["follower"] = 0
        agent.load_config = lambda: {"role": "follower", "loop_hz": 20}
        agent.main(None, None, None, threading.Event())
        assert called["follower"] == 1 and called["leader"] == 0, "Follower dispatch failed"
        print("OK: role dispatch works")
    finally:
        agent.load_config = original_load
        agent.run_leader = original_leader
        agent.run_follower = original_follower


def test_loops_exit_cleanly():
    _run_and_stop(agent.run_leader)
    _run_and_stop(agent.run_follower)
    print("OK: loops exit cleanly with fake stop_event")


if __name__ == "__main__":
    test_stop_pass_through_triggers_on_sign_exit()
    test_next_state_transitions()
    test_smooth_stop()
    test_convoy_leds()
    test_vision_speed_cap()
    test_config_loads()
    test_role_dispatch()
    test_loops_exit_cleanly()
    print("All sanity checks passed.")
