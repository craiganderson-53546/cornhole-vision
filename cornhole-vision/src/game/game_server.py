#!/usr/bin/env python3
"""
game_server.py

The vision-based cornhole scorer's entry point. Runs a background
thread that repeatedly captures a frame, runs detect_bags against it,
and feeds the result into a GameEngine (see game_engine.py for all the
actual game logic). Serves the resulting web UI with Python's
standard-library http.server -- no Flask, no pip install required on
the Pi, consistent with the capture pipeline's no-dependency stance.

Usage:
    python3 game_server.py --calib session.cal [--port 8080]

Requires calibrate_teams' full workflow (background, team A, team B,
hole) to have already been run against session.cal -- detect_bags
will refuse to run and say exactly which key is missing if not.
"""

import argparse
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from game_engine import GameEngine, parse_detect_bags_output, build_html

# ── Paths -- adjust for your layout, or pass via CLI flags below ──
DEFAULT_DETECT_BAGS_BIN = "./detect_bags"
DEFAULT_FRAME_PATH = "/tmp/cornhole_frame.yuv420"
DEFAULT_POLL_INTERVAL_SEC = 1.5
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


def load_roi_and_hole(calib_path: str) -> dict:
    """Reads just the ROI + hole fields out of the calibration file --
    everything detect_bags itself needs (color references) stays in
    the file and is read by detect_bags directly; this is only for
    the SVG renderer, which needs to know the board's extent and hole
    position to draw the picture."""
    values = {}
    with open(calib_path) as f:
        for line in f:
            if "=" not in line:
                continue
            key, _, val = line.strip().partition("=")
            try:
                values[key] = float(val)
            except ValueError:
                continue
    required = ["roi_x0", "roi_y0", "roi_x1", "roi_y1"]
    missing = [k for k in required if k not in values]
    if missing:
        raise SystemExit(
            f"'{calib_path}' is missing {missing} -- run calibrate_teams's "
            f"'background' step first."
        )
    roi = {
        "x0": int(values["roi_x0"]), "y0": int(values["roi_y0"]),
        "x1": int(values["roi_x1"]), "y1": int(values["roi_y1"]),
    }
    corner_keys = ["roi_c0x", "roi_c0y", "roi_c1x", "roi_c1y",
                   "roi_c2x", "roi_c2y", "roi_c3x", "roi_c3y"]
    if all(k in values for k in corner_keys):
        roi["corners"] = [
            (values["roi_c0x"], values["roi_c0y"]),
            (values["roi_c1x"], values["roi_c1y"]),
            (values["roi_c2x"], values["roi_c2y"]),
            (values["roi_c3x"], values["roi_c3y"]),
        ]
    else:
        roi["corners"] = None
        print("warning: no 4-corner calibration found -- board SVG will use "
              "the older, less accurate bounding-box mapping. Recalibrate "
              "with calibrate_teams's updated 'background' step (now takes "
              "4 corners instead of 2) to fix this.", file=sys.stderr)
    if all(k in values for k in ("hole_cx", "hole_cy", "hole_radius")):
        roi["hole_cx"] = int(values["hole_cx"])
        roi["hole_cy"] = int(values["hole_cy"])
        roi["hole_radius"] = int(values["hole_radius"])
    else:
        roi["hole_cx"] = None
        print("warning: no hole calibration found -- board SVG will render "
              "without a hole marker. Run 'calibrate_teams hole ...' when "
              "convenient; detect_bags itself still requires it.",
              file=sys.stderr)
    return roi


def capture_frame(frame_path: str) -> bool:
    """Grabs exactly one frame via rpicam-vid. Returns False (and logs
    to stderr) on failure rather than raising, so a single bad poll
    doesn't take the whole server down."""
    try:
        subprocess.run(
            ["rpicam-vid", "--codec", "yuv420",
             "--width", str(FRAME_WIDTH), "--height", str(FRAME_HEIGHT),
             "--frames", "1", "-n", "-o", frame_path],
            check=True, capture_output=True, timeout=10,
        )
        return True
    except subprocess.TimeoutExpired:
        print("warning: rpicam-vid timed out -- is the camera busy or "
              "disconnected?", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"warning: rpicam-vid failed: {e.stderr.decode(errors='replace')}",
              file=sys.stderr)
    except FileNotFoundError:
        print("error: rpicam-vid not found on PATH.", file=sys.stderr)
    return False


def run_detect_bags(detect_bin: str, frame_path: str, calib_path: str):
    """Returns a list of BlobReading, or None on failure."""
    try:
        result = subprocess.run(
            [detect_bin, frame_path, calib_path],
            capture_output=True, text=True, timeout=5,
        )
    except subprocess.TimeoutExpired:
        print("warning: detect_bags timed out", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"error: '{detect_bin}' not found -- build it first "
              f"(gcc -std=c17 -O2 -o detect_bags detect_bags.c -lm)",
              file=sys.stderr)
        return None
    if result.returncode != 0:
        print(f"warning: detect_bags exited {result.returncode}: "
              f"{result.stderr.strip()}", file=sys.stderr)
        return None
    return parse_detect_bags_output(result.stdout)


def poll_loop(engine: GameEngine, detect_bin: str, frame_path: str,
              calib_path: str, interval: float, stop_event: threading.Event):
    while not stop_event.is_set():
        start = time.time()
        if capture_frame(frame_path):
            readings = run_detect_bags(detect_bin, frame_path, calib_path)
            if readings is not None:
                engine.process_poll(readings)
        elapsed = time.time() - start
        stop_event.wait(max(0.0, interval - elapsed))


def make_handler(engine: GameEngine, roi: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # keep stdout quiet; rely on the poll loop's own warnings

        def do_GET(self):
            if self.path == "/" or self.path == "":
                state = engine.snapshot()
                html = build_html(state, roi)
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/newgame":
                engine.new_game()
                self.send_response(303)
                self.send_header("Location", "/")
                self.end_headers()
            elif self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_response(404)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Not found")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib", required=True, help="calibration file from calibrate_teams")
    parser.add_argument("--detect-bin", default=DEFAULT_DETECT_BAGS_BIN)
    parser.add_argument("--frame-path", default=DEFAULT_FRAME_PATH)
    parser.add_argument("--interval", type=float, default=DEFAULT_POLL_INTERVAL_SEC,
                         help="seconds between camera polls")
    parser.add_argument("--port", type=int, default=8080,
                         help="ports below 1024 need root or "
                              "'sudo setcap cap_net_bind_service=+ep $(which python3)'")
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    roi = load_roi_and_hole(args.calib)

    engine = GameEngine()
    stop_event = threading.Event()
    poller = threading.Thread(
        target=poll_loop,
        args=(engine, args.detect_bin, args.frame_path, args.calib,
              args.interval, stop_event),
        daemon=True,
    )
    poller.start()

    handler_cls = make_handler(engine, roi)
    httpd = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"Cornhole scorer running at http://{args.host}:{args.port}/ "
          f"(polling every {args.interval}s)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        httpd.shutdown()


if __name__ == "__main__":
    main()
