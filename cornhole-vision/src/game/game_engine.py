"""
game_engine.py

Pure game-state logic for the vision-based cornhole scorer -- the
Python-side counterpart to the ESP32 load-cell game's GameState/
scoreRound()/buildBoardSVG(). Deliberately has no hardware or network
code in it: everything here is a plain function or class you can call
with synthetic data and check the output, the same way calibrate_teams
and detect_bags get smoke-tested with synthetic frames before trusting
them on real hardware.

How this differs from the load-cell version, and why
------------------------------------------------------
- Team ID: nearest-color-reference instead of a weight threshold.
  Comes pre-computed from `detect_bags` (team='A'|'B' per blob).
- Position: no inches, no x/y triangulation. `detect_bags` already
  tells us in_hole directly (a point-in-circle test done in pixel
  space). This module just carries that boolean through.
- Bag add/remove detection: load cells watched for a step change in
  total weight, sampled continuously. A camera poll only sees "here
  are the blobs right now" with no history, so this module debounces
  across consecutive polls itself (ADD_CONFIRM_POLLS /
  REMOVE_CONFIRM_POLLS) rather than trusting a single reading -- a
  bag mid-flight, a hand lingering over the board, or one noisy frame
  shouldn't register as an event.
- Scoring trigger: the ESP32 version only calls scoreRound() after a
  3-minute hold following the 8th bag, or the 's'/'n' serial commands
  -- if players cleared the board by hand before that timer elapsed,
  the round's score was silently lost. That looked like an oversight
  rather than something worth preserving, so this version scores the
  moment the board goes empty (bag_count back to 0), whichever way
  that happens, and keeps the 3-minute hold only as a fallback for
  "everyone walked away and never picked the bags back up." Flagging
  this explicitly since it's a real behavioral difference, not just a
  sensing-technology swap.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import re
import time

# ── Tunables ──────────────────────────────────────────────────────
MAX_BAGS = 8                 # 4 per team per round, same as the ESP32 version
MATCH_DISTANCE_PX = 80       # how close (full-res px) a reading must be to an
                              # existing bag to count as "still that bag"
ADD_CONFIRM_POLLS = 2        # consecutive polls a new blob must appear before
                              # it's counted as a thrown bag
REMOVE_CONFIRM_POLLS = 2     # consecutive polls a bag must be missing before
                              # it's counted as picked up
ROUND_END_HOLD_SEC = 180     # fallback auto-score if the board is never
                              # manually cleared (matches ESP32's 3 minutes)
PENDING_GRID_PX = 20         # bucket size for matching a new blob to itself
                              # across polls despite a few px of jitter

TEAM_DISPLAY = {  # purely cosmetic, matches the ESP32 version's palette
    'A': {'name': 'RED',  'fill': '#e94560', 'ring': '#ffaabb'},
    'B': {'name': 'BLUE', 'fill': '#4d9de0', 'ring': '#a8d8ff'},
}


@dataclass
class BlobReading:
    """One line of detect_bags output for the current poll."""
    team: str        # 'A' or 'B'
    in_hole: bool
    cx: int           # full-res (Y-plane) pixel coordinates
    cy: int
    area: int


_DETECT_LINE_RE = re.compile(
    r"bag team=(?P<team>[AB]) in_hole=(?P<hole>[01]) "
    r"cx=(?P<cx>-?\d+) cy=(?P<cy>-?\d+) area=(?P<area>\d+)"
)


def parse_detect_bags_output(stdout_text: str) -> List[BlobReading]:
    """Parses detect_bags' stdout into a list of BlobReading.
    Unrecognized lines are ignored rather than raising -- stderr
    diagnostics from the same tool shouldn't crash a poll cycle."""
    readings = []
    for line in stdout_text.splitlines():
        m = _DETECT_LINE_RE.match(line.strip())
        if not m:
            continue
        readings.append(BlobReading(
            team=m.group('team'),
            in_hole=(m.group('hole') == '1'),
            cx=int(m.group('cx')),
            cy=int(m.group('cy')),
            area=int(m.group('area')),
        ))
    return readings


@dataclass
class Bag:
    """One thrown bag, for the lifetime of the round (mirrors the
    ESP32 version's fixed 8-slot bags[] array, including keeping
    inactive/removed entries around so the UI can show them as
    struck-through "ghost" rows, same as the original)."""
    id: int          # 1-based throw order, used as the table row number
    team: str
    in_hole: bool
    cx: int
    cy: int
    active: bool = True


@dataclass
class GameState:
    bags: List[Bag] = field(default_factory=list)
    bag_count: int = 0            # active bags currently on the board
    total_thrown: int = 0         # cumulative confirmed adds this round
    score_a: int = 0              # accumulated across rounds
    score_b: int = 0
    round_ending: bool = False
    round_end_time: float = 0.0
    status_msg: str = "Ready for new round."


class GameEngine:
    """Owns GameState and all the debounce bookkeeping around it.
    Thread-safe: process_poll() is meant to be called from a single
    background polling thread, snapshot() from the web handler thread."""

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self.state = GameState()
        self._next_id = 1
        self._pending_add: Dict[Tuple[str, int, int], dict] = {}
        self._missing_streak: Dict[int, int] = {}

    def snapshot(self) -> GameState:
        """Returns a shallow copy safe to read from another thread
        without holding the lock while rendering HTML."""
        with self._lock:
            return GameState(
                bags=list(self.state.bags),
                bag_count=self.state.bag_count,
                total_thrown=self.state.total_thrown,
                score_a=self.state.score_a,
                score_b=self.state.score_b,
                round_ending=self.state.round_ending,
                round_end_time=self.state.round_end_time,
                status_msg=self.state.status_msg,
            )

    def process_poll(self, raw: List[BlobReading]) -> None:
        with self._lock:
            self._match_existing(raw)
            self._confirm_new(raw)
            self._check_round_end()

    def new_game(self) -> None:
        """Hard reset from the web UI's New Game button -- discards
        any round in progress and zeroes accumulated scores, same as
        the ESP32 version's handleNewGame()."""
        with self._lock:
            self.state.score_a = 0
            self.state.score_b = 0
            self._reset_round()
            self.state.status_msg = "New game started."

    # ── internals ────────────────────────────────────────────────

    def _match_existing(self, raw: List[BlobReading]) -> None:
        unmatched = list(raw)
        for bag in self.state.bags:
            if not bag.active:
                continue
            match = self._nearest(bag, unmatched)
            if match is not None:
                bag.cx, bag.cy, bag.in_hole = match.cx, match.cy, match.in_hole
                unmatched.remove(match)
                self._missing_streak.pop(bag.id, None)
            else:
                streak = self._missing_streak.get(bag.id, 0) + 1
                if streak >= REMOVE_CONFIRM_POLLS:
                    bag.active = False
                    self.state.bag_count -= 1
                    self.state.status_msg = (
                        f"Bag {bag.id} removed. {self.state.bag_count} "
                        f"bag(s) remain."
                    )
                    self._missing_streak.pop(bag.id, None)
                else:
                    self._missing_streak[bag.id] = streak
        # whatever's left in `unmatched` is handled by _confirm_new
        self._unmatched = unmatched

    def _confirm_new(self, raw: List[BlobReading]) -> None:
        still_pending = {}
        active_count = sum(1 for b in self.state.bags if b.active)
        for reading in self._unmatched:
            key = (reading.team, reading.cx // PENDING_GRID_PX,
                   reading.cy // PENDING_GRID_PX)
            prev = self._pending_add.get(key)
            streak = (prev['streak'] + 1) if prev else 1
            if streak >= ADD_CONFIRM_POLLS:
                if active_count < MAX_BAGS:
                    self._add_bag(reading)
                    active_count += 1
                # else: board already full -- ignore extra detections
                # rather than raising, since 9+ blobs usually means
                # noise/glare, not a 9th bag in a standard game.
            else:
                still_pending[key] = {'streak': streak}
        self._pending_add = still_pending

    def _nearest(self, bag: Bag, candidates: List[BlobReading]) -> Optional[BlobReading]:
        best, best_d = None, MATCH_DISTANCE_PX
        for c in candidates:
            if c.team != bag.team:
                continue
            d = ((c.cx - bag.cx) ** 2 + (c.cy - bag.cy) ** 2) ** 0.5
            if d < best_d:
                best, best_d = c, d
        return best

    def _add_bag(self, reading: BlobReading) -> None:
        bag = Bag(id=self._next_id, team=reading.team, in_hole=reading.in_hole,
                   cx=reading.cx, cy=reading.cy, active=True)
        self._next_id += 1
        self.state.bags.append(bag)
        self.state.bag_count += 1
        self.state.total_thrown += 1

        pts = 3 if bag.in_hole else 1
        team_name = TEAM_DISPLAY[bag.team]['name']
        self.state.status_msg = (
            f"{team_name} bag {self.state.total_thrown}: "
            f"{'HOLE!' if bag.in_hole else 'on board'} (+{pts})"
        )

        if self.state.total_thrown >= MAX_BAGS and not self.state.round_ending:
            self.state.round_ending = True
            self.state.round_end_time = time.time()

    def _check_round_end(self) -> None:
        # Board physically cleared -> score whatever was on it and
        # reset, regardless of whether all 8 had been thrown. See the
        # module docstring for why this differs from the ESP32 source.
        if self.state.total_thrown > 0 and self.state.bag_count == 0:
            self._score_round()
            self._reset_round()
            return
        # Fallback: nobody cleared the board after the 8th bag -- 
        # auto-score after the same hold the ESP32 version uses.
        if self.state.round_ending and \
           (time.time() - self.state.round_end_time >= ROUND_END_HOLD_SEC):
            self._score_round()
            self._reset_round()

    def _score_round(self) -> None:
        # Sum every bag thrown this round, not just ones still flagged
        # `active` -- by the time the board-clear trigger fires, every
        # bag has already been marked removed in this same poll cycle,
        # so filtering on `active` here would always score 0-0. A bag
        # counts toward the round's total once it's been confirmed
        # placed; removal ends the round, it doesn't erase the throw.
        pts_a = sum((3 if b.in_hole else 1)
                    for b in self.state.bags if b.team == 'A')
        pts_b = sum((3 if b.in_hole else 1)
                    for b in self.state.bags if b.team == 'B')
        net_a = max(0, pts_a - pts_b)
        net_b = max(0, pts_b - pts_a)
        self.state.score_a += net_a
        self.state.score_b += net_b
        self.state.status_msg = (
            f"Round scored: RED +{net_a} (raw {pts_a})  "
            f"BLUE +{net_b} (raw {pts_b})  |  "
            f"Total RED {self.state.score_a} \u2013 BLUE {self.state.score_b}"
        )

    def _reset_round(self) -> None:
        self.state.bags = []
        self.state.bag_count = 0
        self.state.total_thrown = 0
        self.state.round_ending = False
        self.state.round_end_time = 0.0
        self._pending_add = {}
        self._missing_streak = {}
        self._next_id = 1


# ── Rendering ─────────────────────────────────────────────────────
# Layout/CSS/colors are carried over from the ESP32 version almost
# verbatim. Removed: weight stat card, corner-weight grid (both
# load-cell-specific). Board SVG now sizes itself from the ROI's own
# aspect ratio instead of a hardcoded 24x48" board, and bag/hole
# placement is the bag's position as a fraction of the ROI -- no
# inches anywhere.

def build_board_svg(state: GameState, roi: dict) -> str:
    roi_w = max(1, roi['x1'] - roi['x0'])
    roi_h = max(1, roi['y1'] - roi['y0'])
    target_w = 320.0
    scale = target_w / roi_w
    sw, sh = int(roi_w * scale), int(roi_h * scale)

    def to_svg(cx: int, cy: int) -> Tuple[int, int]:
        fx = (cx - roi['x0']) / roi_w
        fy = (cy - roi['y0']) / roi_h
        return (int(max(9, min(sw - 9, fx * sw))),
                int(max(9, min(sh - 9, fy * sh))))

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {sw} {sh}' "
        f"style='width:100%;max-width:380px;display:block;margin:0 auto;"
        f"border-radius:8px;'>",
        f"<rect width='{sw}' height='{sh}' rx='6' fill='#6b4c2a'/>",
        f"<rect width='{sw}' height='{sh}' rx='6' fill='none' "
        f"stroke='#9b7a50' stroke-width='3'/>",
    ]

    if roi.get('hole_cx') is not None:
        hx, hy = to_svg(roi['hole_cx'], roi['hole_cy'])
        hr = max(4, int(roi['hole_radius'] * scale))
        parts.append(f"<circle cx='{hx}' cy='{hy}' r='{hr}' "
                      f"fill='#111' stroke='#555' stroke-width='2'/>")

    for bag in state.bags:
        bx, by = to_svg(bag.cx, bag.cy)
        disp = TEAM_DISPLAY[bag.team]
        if bag.active:
            stroke = '#f0b429' if bag.in_hole else disp['ring']
            sw_ = 3 if bag.in_hole else 2
            parts.append(f"<circle cx='{bx}' cy='{by}' r='9' fill='{disp['fill']}' "
                         f"stroke='{stroke}' stroke-width='{sw_}' opacity='0.93'/>")
            if bag.in_hole:
                parts.append(f"<circle cx='{bx}' cy='{by}' r='12' fill='none' "
                             f"stroke='#f0b429' stroke-width='2' opacity='0.7' "
                             f"stroke-dasharray='3,2'/>")
            parts.append(f"<text x='{bx}' y='{by+4}' text-anchor='middle' "
                         f"fill='white' font-size='8' font-weight='bold' "
                         f"font-family='sans-serif'>{bag.id}</text>")
        else:
            g = disp['fill']
            parts.append(f"<circle cx='{bx}' cy='{by}' r='9' fill='none' "
                         f"stroke='{g}' stroke-width='2' opacity='0.35' "
                         f"stroke-dasharray='4,2'/>")
            parts.append(f"<text x='{bx}' y='{by+4}' text-anchor='middle' "
                         f"fill='{g}' font-size='8' font-weight='bold' "
                         f"font-family='sans-serif' opacity='0.35'>{bag.id}</text>")
    parts.append("</svg>")
    return "".join(parts)


def build_html(state: GameState, roi: dict) -> str:
    bags_each_side = MAX_BAGS // 2
    red_thrown = sum(1 for b in state.bags if b.team == 'A')
    blue_thrown = sum(1 for b in state.bags if b.team == 'B')
    red_dots = "".join("&#9679;" if i < red_thrown else "&#9675;"
                        for i in range(bags_each_side))
    blue_dots = "".join("&#9679;" if i < blue_thrown else "&#9675;"
                         for i in range(bags_each_side))

    remaining = MAX_BAGS - state.total_thrown
    if state.round_ending:
        elapsed = time.time() - state.round_end_time
        secs_left = int(max(0, ROUND_END_HOLD_SEC - elapsed))
        prompt = (f"&#8987; Round complete! Auto-clearing in "
                  f"{secs_left // 60}:{secs_left % 60:02d}&hellip; "
                  f"(or just pick up the bags)")
    elif remaining <= 0:
        prompt = "&#127937; Round complete! Remove all bags to start a new round."
    else:
        prompt = f"&#127919; Throw bag {state.total_thrown + 1} of {MAX_BAGS}"

    rows = []
    for b in state.bags:
        disp = TEAM_DISPLAY[b.team]
        row_style = "" if b.active else " style='opacity:0.35;text-decoration:line-through'"
        result = ("<td style='color:#f0b429;font-weight:bold'>HOLE &mdash; 3 pts</td>"
                   if b.in_hole else "<td>Board &mdash; 1 pt</td>")
        rows.append(
            f"<tr{row_style}><td>{b.id}{'' if b.active else ' &#10007;'}</td>"
            f"<td style='color:{disp['fill']};font-weight:bold'>{disp['name']}</td>"
            f"{result}</tr>"
        )
    bag_rows_html = "".join(rows) if rows else \
        "<tr><td colspan='3' class='empty'>No bags thrown yet</td></tr>"

    board_svg = build_board_svg(state, roi)

    return f"""<!DOCTYPE html><html lang='en'><head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width,initial-scale=1.0'>
<meta http-equiv='refresh' content='2'>
<title>Cornhole Scorer</title><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
background:#1a1a2e;color:#eee;padding:14px;max-width:480px;margin:0 auto}}
h1{{text-align:center;color:#e94560;font-size:1.5rem;margin-bottom:3px}}
.sub{{text-align:center;color:#666;font-size:0.78rem;margin-bottom:14px}}
.scoreboard{{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;
gap:8px;background:#16213e;border-radius:12px;padding:14px 12px;
margin-bottom:14px;border:2px solid #0f3460}}
.score-team{{text-align:center}}
.score-name{{font-size:0.7rem;font-weight:800;letter-spacing:2px;margin-bottom:4px}}
.score-name.red{{color:#e94560}} .score-name.blue{{color:#4d9de0}}
.score-val{{font-size:2.8rem;font-weight:900;line-height:1}}
.score-val.red{{color:#e94560}} .score-val.blue{{color:#4d9de0}}
.score-sep{{font-size:1.6rem;color:#444;font-weight:300;text-align:center}}
.score-lbl{{font-size:0.6rem;color:#555;text-align:center;margin-top:4px;letter-spacing:1px}}
.prompt{{background:#16213e;border-left:4px solid #e94560;
padding:13px 15px;border-radius:8px;font-size:1.05rem;margin-bottom:14px}}
.teams{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}}
.tcard{{border-radius:10px;padding:12px;text-align:center}}
.tcard.red{{background:#200a10;border:2px solid #e94560}}
.tcard.blue{{background:#091525;border:2px solid #4d9de0}}
.tname{{font-size:0.75rem;font-weight:800;letter-spacing:2px;margin-bottom:5px}}
.tname.red{{color:#e94560}} .tname.blue{{color:#4d9de0}}
.dots{{font-size:1.1rem;letter-spacing:3px}}
.dots.red{{color:#e94560}} .dots.blue{{color:#4d9de0}}
.tbags{{font-size:0.68rem;color:#666;margin-top:4px}}
.cards{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}}
.card{{background:#16213e;border-radius:10px;padding:12px;text-align:center}}
.card .val{{font-size:1.9rem;font-weight:bold;color:#e94560}}
.card .lbl{{font-size:0.72rem;color:#aaa;margin-top:2px}}
.bw{{background:#16213e;border-radius:10px;padding:10px;margin-bottom:14px}}
.bt{{font-size:0.7rem;color:#666;text-align:center;margin-bottom:6px}}
.leg{{display:flex;justify-content:center;gap:16px;font-size:0.7rem;
margin-top:6px;color:#aaa;flex-wrap:wrap}}
.dot-red{{display:inline-block;width:10px;height:10px;border-radius:50%;
background:#e94560;vertical-align:middle;margin-right:3px}}
.dot-blue{{display:inline-block;width:10px;height:10px;border-radius:50%;
background:#4d9de0;vertical-align:middle;margin-right:3px}}
.dot-hole{{display:inline-block;width:10px;height:10px;border-radius:50%;
background:#e94560;border:2px solid #f0b429;vertical-align:middle;margin-right:3px}}
.dot-off{{display:inline-block;width:10px;height:10px;border-radius:50%;
border:2px dashed #888;vertical-align:middle;margin-right:3px;opacity:0.4}}
table{{width:100%;border-collapse:collapse;background:#16213e;
border-radius:10px;overflow:hidden;margin-bottom:14px}}
th{{background:#0f3460;padding:9px 8px;font-size:0.78rem;color:#aaa}}
td{{padding:8px;text-align:center;font-size:0.88rem;border-bottom:1px solid #0f3460}}
tr:last-child td{{border-bottom:none}}
.empty{{color:#555;font-style:italic;text-align:center;padding:14px}}
.status{{text-align:center;color:#444;font-size:0.7rem;margin-top:4px}}
.btn-newgame{{display:block;width:100%;padding:14px;margin-top:14px;
background:#1a1a2e;color:#e94560;font-size:1rem;font-weight:700;
border:2px solid #e94560;border-radius:10px;cursor:pointer;
letter-spacing:1px;text-align:center;text-decoration:none;
-webkit-tap-highlight-color:transparent}}
.btn-newgame:active{{background:#e94560;color:#fff}}
</style></head><body>
<h1>&#127919; Cornhole Scorer</h1>
<p class='sub'>Auto-refreshes every 2 seconds</p>
<div class='scoreboard'>
<div class='score-team'><div class='score-name red'>RED</div>
<div class='score-val red'>{state.score_a}</div></div>
<div><div class='score-sep'>&#8211;</div><div class='score-lbl'>SCORE</div></div>
<div class='score-team'><div class='score-name blue'>BLUE</div>
<div class='score-val blue'>{state.score_b}</div></div>
</div>
<div class='prompt'>{prompt}</div>
<div class='teams'>
<div class='tcard red'><div class='tname red'>RED</div>
<div class='dots red'>{red_dots}</div>
<div class='tbags'>{red_thrown} of {bags_each_side} thrown</div></div>
<div class='tcard blue'><div class='tname blue'>BLUE</div>
<div class='dots blue'>{blue_dots}</div>
<div class='tbags'>{blue_thrown} of {bags_each_side} thrown</div></div>
</div>
<div class='cards'>
<div class='card'><div class='val'>{state.bag_count}</div><div class='lbl'>Bags on Board</div></div>
<div class='card'><div class='val'>{state.total_thrown}</div><div class='lbl'>Bags Thrown</div></div>
</div>
<div class='bw'>
<div class='bt'>BOARD (camera view)</div>
{board_svg}
<div class='leg'>
<span><span class='dot-red'></span>Red (1pt)</span>
<span><span class='dot-blue'></span>Blue (1pt)</span>
<span><span class='dot-hole'></span>Hole (3pts)</span>
<span><span class='dot-off'></span>Removed</span>
</div></div>
<table><thead><tr><th>#</th><th>Team</th><th>Score</th></tr></thead>
<tbody>{bag_rows_html}</tbody></table>
<p class='status'>{state.status_msg}</p>
<a href='/newgame' class='btn-newgame'
onclick="return confirm('Reset scores and start a new game?')">&#9654; NEW GAME</a>
</body></html>"""
