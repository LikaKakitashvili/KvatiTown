import logging
import re
import socket
import sys
import time
import cv2


def configure_line_buffered_stdio() -> None:
    """Flush each print line immediately when stdout is piped (e.g. bot dashboard)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass


class _HttpErrorsOnly(logging.Filter):
    """Pass werkzeug request lines only when status code >= 400."""
    _STATUS_RE = re.compile(r'" (\d{3}) ')

    def filter(self, record):
        m = self._STATUS_RE.search(record.getMessage())
        if m:
            return int(m.group(1)) >= 400
        return True  # non-request lines (startup, errors) always shown


def suppress_http_logs():
    """Call once at server startup to hide 2xx/3xx request noise."""
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.INFO)
    log.addFilter(_HttpErrorsOnly())


def make_frame_generator(get_camera, visualize, quality=70, rgb=True):
    """Return an MJPEG generator. rgb=True calls read_rgb(), False calls read()."""
    def generate():
        while True:
            try:
                cam = get_camera()
                if cam is None:
                    time.sleep(0.05)
                    continue

                ok, frame = cam.read_rgb() if rgb else cam.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                display = visualize(frame)
                ret, jpeg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ret:
                    continue

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n'
                       + jpeg.tobytes() + b'\r\n')

            except Exception as e:
                print(f'[VideoStream] Error: {e}')
                time.sleep(0.05)

    return generate


def find_available_port(start=5000, max_attempts=20, exclude=None):
    """Find a free TCP port by test-binding, starting from `start`."""
    if exclude is None:
        exclude = set()
    for offset in range(max_attempts):
        port = start + offset
        if port in exclude:
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"Could not find a free port in range {start}-{start + max_attempts - 1}"
    )


def shutdown_cleanup(wheels, camera, stop_event):
    """Stop motors, stop camera, set stop_event."""
    stop_event.set()

    if wheels:
        try:
            print("Stopping motors...")
            wheels.set_wheels_speed(0, 0)
            time.sleep(0.1)
            wheels.set_wheels_speed(0, 0)
        except Exception as e:
            print(f"  Error: {e}")

    if camera:
        try:
            print("Stopping camera...")
            camera.stop()
        except Exception as e:
            print(f"  Error: {e}")

    print("\nShutdown complete!")
