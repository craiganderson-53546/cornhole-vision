#!/usr/bin/env python3
"""
calibrate_web.py

A guided, click-through field calibration wizard for the cornhole
vision system -- meant to be run by a player before a game, or
whenever detection performance falls off, without needing to SSH in
and type pixel coordinates by hand (the old calibrate_teams CLI
workflow, which is still what actually does the calibration -- this
just drives it).

What it automates:
  - Board corners + hole position/radius: captured automatically by
    detect_board from one click on the board's border color, instead
    of measuring and typing 8 coordinates (see the bowtie-corner-order
    incident this avoids entirely). The detected geometry is always
    shown overlaid on the photo for a human to confirm before it's
    used -- auto-detection here is a suggestion, not a silent trust.
  - Background and team color capture: already just "capture a frame,
    calibrate_teams figures out the rest" -- this just wraps that in
    buttons instead of CLI arguments.
  - Cross-team separation check: runs calibrate_teams validate
    automatically after both team captures and surfaces PASS/FAIL
    with the actual distances, rather than requiring a separate
    manual step a field user might not know to run.

What it deliberately does NOT do: recognize non-solid/patterned bags,
or replace calibrate_teams/detect_bags/detect_board -- it only drives
them. If detect_board can't find the border confidently, or the hole
isn't found, this falls back to manual corner/hole clicking on the
same photo -- same underlying mechanism (a click producing pixel
coordinates), just without the auto-detection step succeeding first.

Usage:
    python3 calibrate_web.py [--port 8090] [--calib session.cal]
                              [--detect-board-bin ./detect_board]
                              [--calibrate-teams-bin ./calibrate_teams]

Requires detect_board and calibrate_teams already built
(gcc -std=c17 -O2 -o detect_board detect_board.c -lm, and the same
for calibrate_teams.c).
"""

import argparse
import html
import http.server
import socketserver
import struct
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse, parse_qs

WIDTH = 1280
HEIGHT = 720
CHROMA_W = WIDTH // 2
CHROMA_H = HEIGHT // 2
Y_SIZE = WIDTH * HEIGHT
UV_SIZE = CHROMA_W * CHROMA_H
FRAME_SIZE = Y_SIZE + 2 * UV_SIZE

FRAME_PATH = "/tmp/cornhole_calib_frame.yuv420"


# ── frame capture (same approach as game_server.py) ─────────────────

def capture_frame(frame_path: str) -> str | None:
    """Grabs one frame via rpicam-vid. Returns an error message on
    failure, or None on success -- inverted from a bool so the caller
    can show the actual reason rather than just 'capture failed'."""
    try:
        subprocess.run(
            ["rpicam-vid", "--codec", "yuv420",
             "--width", str(WIDTH), "--height", str(HEIGHT),
             "--frames", "1", "-n", "-o", frame_path],
            check=True, capture_output=True, timeout=10,
        )
        return None
    except subprocess.TimeoutExpired:
        return "rpicam-vid timed out -- is the camera busy or disconnected?"
    except subprocess.CalledProcessError as e:
        return f"rpicam-vid failed: {e.stderr.decode(errors='replace')}"
    except FileNotFoundError:
        return "rpicam-vid not found on PATH."


# ── YUV420 -> BMP, pure Python, no dependencies ─────────────────────

def yuv420_to_bmp(yuv_path: str) -> bytes:
    """Converts a captured YUV420 (4:2:0 planar) frame to an
    uncompressed 24-bit BMP so it can be shown in an <img> tag with no
    image library on either end -- consistent with this project's
    no-dependency stance (see game_server.py's own docstring)."""
    with open(yuv_path, "rb") as f:
        data = f.read()
    if len(data) != FRAME_SIZE:
        raise ValueError(
            f"'{yuv_path}' is {len(data)} bytes, expected {FRAME_SIZE}"
        )
    y_plane = data[:Y_SIZE]
    u_plane = data[Y_SIZE:Y_SIZE + UV_SIZE]
    v_plane = data[Y_SIZE + UV_SIZE:]

    row_size = WIDTH * 3  # already a multiple of 4, no row padding needed
    pixel_data = bytearray(row_size * HEIGHT)

    for row in range(HEIGHT):
        crow = row // 2
        y_row_off = row * WIDTH
        c_row_off = crow * CHROMA_W
        # BMP rows are stored bottom-to-top
        out_row = (HEIGHT - 1 - row) * row_size
        for col in range(WIDTH):
            yv = y_plane[y_row_off + col]
            uv = u_plane[c_row_off + (col // 2)] - 128
            vv = v_plane[c_row_off + (col // 2)] - 128
            r = yv + 1.402 * vv
            g = yv - 0.344136 * uv - 0.714136 * vv
            b = yv + 1.772 * uv
            o = out_row + col * 3
            pixel_data[o] = 0 if b < 0 else (255 if b > 255 else int(b))
            pixel_data[o + 1] = 0 if g < 0 else (255 if g > 255 else int(g))
            pixel_data[o + 2] = 0 if r < 0 else (255 if r > 255 else int(r))

    file_size = 54 + len(pixel_data)
    header = struct.pack(
        "<2sIHHI IiiHHIIiiII",
        b"BM", file_size, 0, 0, 54,
        40, WIDTH, HEIGHT, 1, 24, 0, len(pixel_data), 0, 0, 0, 0,
    )
    return header + bytes(pixel_data)


# ── wizard state ─────────────────────────────────────────────────────

class WizardState:
    """All mutable state for the calibration flow, single global
    instance -- this tool is meant to be used by one person at a time
    walking through a physical board, not multiple concurrent
    sessions."""

    def __init__(self, calib_path: str, detect_board_bin: str,
                 calibrate_teams_bin: str):
        self.lock = threading.Lock()
        self.calib_path = calib_path
        self.detect_board_bin = detect_board_bin
        self.calibrate_teams_bin = calibrate_teams_bin
        self.reset()

    def reset(self):
        self.step = "intro"
        self.frame_captured = False
        self.outer_corners = None  # [(x,y)]*4, perimeter order, or None
        self.inner_corners = None  # [(x,y)]*4, perimeter order, or None
        self.hole = None          # (cx, cy, radius) or None
        self.manual_clicks = []   # accumulator for manual corner/hole entry
        self.log = []             # list of (ok: bool, message: str)
        self.team_results = {}    # 'A'/'B' -> message string
        self.validate_result = None  # (passed: bool, output: str)

    def add_log(self, ok: bool, message: str):
        self.log.append((ok, message))


def run_tool(args: list) -> tuple:
    """Runs a subprocess, returns (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return 1, "", f"'{args[0]}' not found -- build it first."
    except subprocess.TimeoutExpired:
        return 1, "", f"'{args[0]}' timed out."


# ── HTML shell (kept close in style to game_engine.py's dark theme) ─

PAGE_STYLE = """
body{background:#0f1424;color:#eee;font-family:-apple-system,sans-serif;
margin:0;padding:16px}
.wrap{max-width:900px;margin:0 auto}
h1{color:#e94560;font-size:1.3rem;margin:0 0 4px}
.sub{color:#888;font-size:0.85rem;margin:0 0 16px}
.card{background:#16213e;border-radius:12px;padding:16px;margin-bottom:14px}
.step-title{font-size:1.05rem;font-weight:bold;margin-bottom:8px}
.imgwrap{position:relative;display:inline-block;max-width:100%}
.imgwrap img{max-width:100%;display:block;border-radius:8px;cursor:crosshair}
.marker{position:absolute;width:14px;height:14px;margin:-7px 0 0 -7px;
border-radius:50%;border:2px solid #fff;background:#e94560;
font-size:0.6rem;color:#fff;display:flex;align-items:center;
justify-content:center;pointer-events:none}
.marker.hole{background:#f0b429}
.marker.outer{background:#4d9de0}
.btn{display:inline-block;background:#0f3460;color:#eee;border:1px solid
#2a3f66;border-radius:8px;padding:10px 16px;font-size:0.9rem;
text-decoration:none;margin:4px 6px 4px 0;cursor:pointer}
.btn:active{background:#2a3f66}
.btn.primary{background:#e94560;border-color:#e94560}
.btn.primary:active{background:#c13350}
.logline{font-size:0.85rem;padding:4px 0;border-bottom:1px solid #223055}
.logline.ok{color:#5ce38f}
.logline.fail{color:#e97b8f}
pre{background:#0f1b32;padding:10px;border-radius:6px;font-size:0.78rem;
overflow-x:auto;white-space:pre-wrap}
.progress{display:flex;gap:4px;margin-bottom:14px}
.progress div{flex:1;height:4px;border-radius:2px;background:#223055}
.progress div.done{background:#5ce38f}
.progress div.current{background:#e94560}
"""

STEPS_ORDER = ["intro", "geometry", "confirm_geometry", "background",
               "team_a", "team_b", "validate", "done"]


def progress_html(current: str) -> str:
    idx = STEPS_ORDER.index(current) if current in STEPS_ORDER else 0
    bars = []
    for i, _ in enumerate(STEPS_ORDER):
        cls = "done" if i < idx else ("current" if i == idx else "")
        bars.append(f"<div class='{cls}'></div>")
    return f"<div class='progress'>{''.join(bars)}</div>"


def log_html(state: WizardState) -> str:
    if not state.log:
        return ""
    lines = []
    for ok, msg in state.log[-6:]:
        cls = "ok" if ok else "fail"
        lines.append(f"<div class='logline {cls}'>{html.escape(msg)}</div>")
    return f"<div class='card'>{''.join(lines)}</div>"


def page_shell(state: WizardState, body: str) -> bytes:
    html_doc = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Cornhole Field Calibration</title><style>{PAGE_STYLE}</style></head>
<body><div class='wrap'>
<h1>&#127919; Field Calibration</h1>
<p class='sub'>Guided setup for {html.escape(state.calib_path)}</p>
{progress_html(state.step)}
{body}
{log_html(state)}
</div></body></html>"""
    return html_doc.encode("utf-8")


def image_with_markers(markers: list) -> str:
    """markers: list of (x_full, y_full, label, css_class). Coordinates
    are full-res; JS below scales clicks back the same way."""
    marker_html = "".join(
        f"<div class='marker {cls}' data-fx='{x}' data-fy='{y}'>{label}</div>"
        for (x, y, label, cls) in markers
    )
    return f"""
<div class='imgwrap' id='imgwrap'>
<img id='frameimg' src='/frame.bmp?t={int(time.time())}'
 onload='placeMarkers()'>
{marker_html}
</div>
<script>
function placeMarkers() {{
    var img = document.getElementById('frameimg');
    var scaleX = img.clientWidth / {WIDTH};
    var scaleY = img.clientHeight / {HEIGHT};
    document.querySelectorAll('#imgwrap .marker').forEach(function(m) {{
        var fx = parseFloat(m.dataset.fx), fy = parseFloat(m.dataset.fy);
        m.style.left = (fx * scaleX) + 'px';
        m.style.top = (fy * scaleY) + 'px';
    }});
}}
window.addEventListener('resize', placeMarkers);
document.getElementById('frameimg').addEventListener('click', function(e) {{
    var img = e.target;
    var scaleX = {WIDTH} / img.clientWidth;
    var scaleY = {HEIGHT} / img.clientHeight;
    var fx = Math.round(e.offsetX * scaleX);
    var fy = Math.round(e.offsetY * scaleY);
    window.location.href = '/click?x=' + fx + '&y=' + fy;
}});
</script>
"""


# ── per-step page rendering ──────────────────────────────────────────

def render_intro(state: WizardState) -> str:
    return """
<div class='card'>
<div class='step-title'>Before you start</div>
<p>Clear the board of all bags. Make sure the board's high-contrast
border is fully visible to the camera (nothing resting on it, no
shadow cutting across it).</p>
<a class='btn primary' href='/begin'>&#9654; Capture &amp; Start</a>
</div>
"""


def render_geometry(state: WizardState) -> str:
    img = image_with_markers([])
    return f"""
<div class='card'>
<div class='step-title'>Click once on the border</div>
<p>Click anywhere directly on the board's colored border (the tape),
not on the white surface or the floor. This is the only click needed
-- the 4 corners and the hole are found automatically from it.</p>
{img}
<p><a class='btn' href='/capture'>&#8635; Recapture frame</a></p>
</div>
"""


def render_confirm_geometry(state: WizardState) -> str:
    markers = []
    if state.outer_corners:
        for i, (x, y) in enumerate(state.outer_corners):
            markers.append((x, y, "O" + str(i), "outer"))
    if state.inner_corners:
        for i, (x, y) in enumerate(state.inner_corners):
            markers.append((x, y, "I" + str(i), ""))
    if state.hole:
        cx, cy, r = state.hole
        markers.append((cx, cy, "H", "hole"))
    img = image_with_markers(markers)
    hole_note = ""
    if not state.hole:
        hole_note = (
            "<p style='color:#e97b8f'>No hole was detected. You can set it "
            "manually, or continue without it -- but detect_bags will "
            "refuse to run until a hole is calibrated.</p>"
            "<a class='btn' href='/goto?step=manual_hole'>Set hole manually</a>"
        )
    return f"""
<div class='card'>
<div class='step-title'>Confirm board geometry</div>
<p><b>O0-O3</b> (blue) mark the outer boundary -- tape-to-carpet.
<b>I0-I3</b> (red) mark the inner boundary -- interior-to-tape. Both
matter: the rules score a bag on the tape the same as one on the
interior, so both boundaries need to be right, not just one. The hole
(if found) is marked H. Check all of these against the actual board
before continuing -- a wrong corner here throws off everything
downstream.</p>
{img}
{hole_note}
<p>
<a class='btn primary' href='/confirm_geometry_continue'>&#10003; Looks correct, continue</a><br>
<a class='btn' href='/goto?step=manual_corners'>Corners wrong -- enter manually</a>
<a class='btn' href='/goto?step=geometry'>Start over (reclick border)</a>
</p>
</div>
"""


def render_manual_corners(state: WizardState) -> str:
    n = len(state.manual_clicks)
    img = image_with_markers([
        (x, y, "O" + str(i), "outer") for i, (x, y) in enumerate(state.manual_clicks)
    ])
    return f"""
<div class='card'>
<div class='step-title'>Manual entry: OUTER boundary ({n} of 4)</div>
<p>Click the board's 4 outer corners -- the tape-to-carpet edge -- in
order, walking around the perimeter (clockwise or counter-clockwise --
either is fine, just don't jump diagonally across). You'll enter the
inner (interior-to-tape) boundary next.</p>
{img}
<p><a class='btn' href='/goto?step=confirm_geometry'>Cancel</a></p>
</div>
"""


def render_manual_corners_inner(state: WizardState) -> str:
    n = len(state.manual_clicks)
    markers = [(x, y, "O" + str(i), "outer")
               for i, (x, y) in enumerate(state.outer_corners or [])]
    markers += [(x, y, "I" + str(i), "")
                for i, (x, y) in enumerate(state.manual_clicks)]
    img = image_with_markers(markers)
    return f"""
<div class='card'>
<div class='step-title'>Manual entry: INNER boundary ({n} of 4)</div>
<p>Now click the 4 inner corners -- the interior-to-tape edge -- same
perimeter order. If your board has no visually distinct tape, click
the same 4 spots as the outer boundary.</p>
{img}
<p><a class='btn' href='/goto?step=confirm_geometry'>Cancel</a></p>
</div>
"""


def render_manual_hole(state: WizardState) -> str:
    n = len(state.manual_clicks)
    label = "center" if n == 0 else "a point on its edge"
    img = image_with_markers([
        (x, y, str(i), "hole") for i, (x, y) in enumerate(state.manual_clicks)
    ])
    return f"""
<div class='card'>
<div class='step-title'>Manual hole entry ({n} of 2)</div>
<p>Click the hole's {label}.</p>
{img}
<p><a class='btn' href='/goto?step=confirm_geometry'>Cancel</a></p>
</div>
"""


def render_team_capture(state: WizardState, team: str) -> str:
    name = "Team A" if team == "A" else "Team B"
    other = "" if team == "A" else (
        "<p>Team A is already calibrated -- pick a bag color clearly "
        "different from it.</p>"
    )
    return f"""
<div class='card'>
<div class='step-title'>Calibrate {name}</div>
<p>Place a single {name} bag anywhere on the board -- flat, not on its
edge, not overlapping another bag.</p>
{other}
<a class='btn primary' href='/capture'>&#128247; Capture &amp; calibrate {name}</a>
</div>
"""


def render_validate(state: WizardState) -> str:
    if state.validate_result is None:
        return "<div class='card'>Running validation...</div>"
    passed, output = state.validate_result
    verdict = "PASS" if passed else "FAIL"
    color = "#5ce38f" if passed else "#e97b8f"
    retry = ""
    if not passed:
        retry = """
<p>
<a class='btn' href='/goto?step=team_a'>Redo Team A</a>
<a class='btn' href='/goto?step=team_b'>Redo Team B</a>
</p>
"""
    finish = (
        "<a class='btn primary' href='/finish'>&#10003; Finish</a>"
        if passed else ""
    )
    return f"""
<div class='card'>
<div class='step-title'>Validation: <span style='color:{color}'>{verdict}</span></div>
<pre>{html.escape(output)}</pre>
{retry}
{finish}
</div>
"""


def render_done(state: WizardState) -> str:
    return f"""
<div class='card'>
<div class='step-title'>&#9989; Calibration complete</div>
<p><code>{html.escape(state.calib_path)}</code> is ready to use.</p>
<p>Start (or restart) the scorer with it:</p>
<pre>python3 game_server.py --calib {html.escape(state.calib_path)}</pre>
<a class='btn' href='/reset'>Run calibration again</a>
</div>
"""


STEP_RENDERERS = {
    "intro": render_intro,
    "geometry": render_geometry,
    "confirm_geometry": render_confirm_geometry,
    "manual_corners": render_manual_corners,
    "manual_corners_inner": render_manual_corners_inner,
    "manual_hole": render_manual_hole,
    "team_a": lambda s: render_team_capture(s, "A"),
    "team_b": lambda s: render_team_capture(s, "B"),
    "validate": render_validate,
    "done": render_done,
}


# ── actions ──────────────────────────────────────────────────────────

def run_background_and_hole(state: WizardState):
    args = [state.calibrate_teams_bin, "background", FRAME_PATH]
    for (x, y) in state.outer_corners:
        args += [str(int(x)), str(int(y))]
    for (x, y) in state.inner_corners:
        args += [str(int(x)), str(int(y))]
    args += [state.calib_path]
    rc, out, err = run_tool(args)
    state.add_log(rc == 0, (out + err).strip() or "background: no output")
    if rc != 0:
        return False
    if state.hole:
        cx, cy, r = state.hole
        rc2, out2, err2 = run_tool([
            state.calibrate_teams_bin, "hole",
            str(int(cx)), str(int(cy)), str(int(r)), state.calib_path,
        ])
        state.add_log(rc2 == 0, (out2 + err2).strip() or "hole: no output")
    return True


def run_team_capture(state: WizardState, team: str):
    rc, out, err = run_tool([
        state.calibrate_teams_bin, "team", team, FRAME_PATH, state.calib_path,
    ])
    state.add_log(rc == 0, (out + err).strip() or f"team {team}: no output")
    return rc == 0


def run_validate(state: WizardState):
    rc, out, err = run_tool([state.calibrate_teams_bin, "validate", state.calib_path])
    state.validate_result = (rc == 0, (out + err).strip())


def handle_click(state: WizardState, fx: int, fy: int):
    if state.step == "geometry":
        if not state.frame_captured:
            state.add_log(False, "No frame captured yet.")
            return
        rc, out, err = run_tool([
            state.detect_board_bin, FRAME_PATH, str(fx), str(fy),
        ])
        if rc != 0:
            state.add_log(False, (err or out).strip() or "detect_board failed")
            return  # stay on geometry step, let them click again
        inner = [None] * 4
        outer = [None] * 4
        hole = None
        for line in out.splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "inner":
                inner[int(parts[1])] = (int(parts[2]), int(parts[3]))
            elif parts[0] == "outer":
                outer[int(parts[1])] = (int(parts[2]), int(parts[3]))
            elif parts[0] == "hole":
                hole = (int(parts[1]), int(parts[2]), int(parts[3]))
        if any(c is None for c in inner) or any(c is None for c in outer):
            state.add_log(False, "detect_board didn't return all 8 corners.")
            return
        state.inner_corners = inner
        state.outer_corners = outer
        state.hole = hole
        state.add_log(True, "Board geometry detected -- review below.")
        if hole is None:
            state.add_log(False, "No hole detected -- set it manually if needed.")
        state.step = "confirm_geometry"

    elif state.step == "manual_corners":
        state.manual_clicks.append((fx, fy))
        if len(state.manual_clicks) >= 4:
            state.outer_corners = state.manual_clicks[:4]
            state.manual_clicks = []
            state.add_log(True, "4 outer corners entered -- now the inner boundary.")
            state.step = "manual_corners_inner"

    elif state.step == "manual_corners_inner":
        state.manual_clicks.append((fx, fy))
        if len(state.manual_clicks) >= 4:
            state.inner_corners = state.manual_clicks[:4]
            state.manual_clicks = []
            state.add_log(True, "4 inner corners entered manually.")
            state.step = "confirm_geometry"

    elif state.step == "manual_hole":
        state.manual_clicks.append((fx, fy))
        if len(state.manual_clicks) >= 2:
            (cx, cy), (ex, ey) = state.manual_clicks
            radius = int(((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5)
            state.hole = (cx, cy, radius)
            state.manual_clicks = []
            state.add_log(True, f"Hole set manually: center=({cx},{cy}) r={radius}")
            state.step = "confirm_geometry"


# ── HTTP handler ─────────────────────────────────────────────────────

def make_handler(state: WizardState):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _redirect(self, path="/"):
            self.send_response(303)
            self.send_header("Location", path)
            self.end_headers()

        def _send_html(self, body: bytes):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            q = parse_qs(parsed.query)

            with state.lock:
                if path == "/" or path == "":
                    renderer = STEP_RENDERERS.get(state.step, render_intro)
                    self._send_html(page_shell(state, renderer(state)))

                elif path == "/frame.bmp":
                    if not state.frame_captured:
                        self.send_response(404)
                        self.end_headers()
                        return
                    try:
                        bmp = yuv420_to_bmp(FRAME_PATH)
                    except Exception as e:
                        self.send_response(500)
                        self.end_headers()
                        self.wfile.write(str(e).encode())
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/bmp")
                    self.send_header("Content-Length", str(len(bmp)))
                    self.end_headers()
                    self.wfile.write(bmp)

                elif path == "/begin":
                    err = capture_frame(FRAME_PATH)
                    state.frame_captured = err is None
                    state.add_log(err is None, err or "Frame captured.")
                    if err is None:
                        state.step = "geometry"
                    self._redirect()

                elif path == "/capture":
                    err = capture_frame(FRAME_PATH)
                    state.frame_captured = err is None
                    state.add_log(err is None, err or "Frame captured.")
                    if err is None and state.step in ("team_a", "team_b"):
                        team = "A" if state.step == "team_a" else "B"
                        if run_team_capture(state, team):
                            if team == "A":
                                state.step = "team_b"
                            else:
                                state.step = "validate"
                                run_validate(state)
                    self._redirect()

                elif path == "/click":
                    try:
                        fx = int(q["x"][0])
                        fy = int(q["y"][0])
                    except (KeyError, ValueError, IndexError):
                        self._redirect()
                        return
                    handle_click(state, fx, fy)
                    self._redirect()

                elif path == "/confirm_geometry_continue":
                    if run_background_and_hole(state):
                        state.step = "team_a"
                    self._redirect()

                elif path == "/goto":
                    target = q.get("step", [""])[0]
                    if target in STEP_RENDERERS:
                        state.manual_clicks = []
                        state.step = target
                        if target == "validate":
                            run_validate(state)
                    self._redirect()

                elif path == "/finish":
                    state.step = "done"
                    self._redirect()

                elif path == "/reset":
                    state.reset()
                    self._redirect()

                elif path == "/favicon.ico":
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
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--calib", default="session.cal")
    parser.add_argument("--detect-board-bin", default="./detect_board")
    parser.add_argument("--calibrate-teams-bin", default="./calibrate_teams")
    args = parser.parse_args()

    state = WizardState(args.calib, args.detect_board_bin, args.calibrate_teams_bin)
    handler_cls = make_handler(state)
    httpd = socketserver.ThreadingTCPServer((args.host, args.port), handler_cls)
    print(f"Field calibration wizard running at "
          f"http://{args.host}:{args.port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
    