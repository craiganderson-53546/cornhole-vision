# Cornhole Vision — Calibration Procedure

Run this whenever the board, camera position, or lighting changes
significantly enough that the old `session.cal` can't be trusted.
Every command below assumes you're in the directory with the compiled
`calibrate_teams` and `detect_bags` binaries.

**Board prerequisites before starting:** playing surface is a single
blank color (no printing/pattern — that's what caused the >32-blob
issue), green tape marks the interior border, hole is unobstructed
and visible to the camera.

---

## 1. Capture a reference frame

```
rpicam-vid --codec yuv420 --width 1280 --height 720 --frames 1 -n -o ref_frame.yuv420
ffmpeg -f rawvideo -pix_fmt yuv420p -s 1280x720 -i ref_frame.yuv420 -frames:v 1 ref_frame.png
```

Board should be empty (no bags) for this one — it's what you'll use
to pick coordinates and to calibrate the background in step 3.

`--frames 1` is required, not `-t <ms>` — a short timeout can still
emit several frames, which `ffmpeg`/`calibrate_teams` will reject or
silently mis-happy about since they expect exactly one frame's worth
of bytes.

---

## 2. Find the ROI corners

Open `ref_frame.png` in GIMP or IrfanView (needs live cursor-position
readout — VS Code's built-in preview doesn't show this).

Hover each of the four corners of the green tape's **inner edge** and
note (x, y) for each.

Because the camera is angled, the interior isn't a perfect rectangle.
Inscribe a safe axis-aligned rectangle inside all four corners:

- `roi_x0` = larger of the two **left** corners' x
- `roi_x1` = smaller of the two **right** corners' x
- `roi_y0` = larger of the two **top** corners' y
- `roi_y1` = smaller of the two **bottom** corners' y

Then shrink all four inward another 10–20px as a margin, so the
rectangle sits entirely on board surface even if your corner picks
were slightly off.

---

## 3. Calibrate the background

Board still empty, using the raw frame from step 1:

```
./calibrate_teams background ref_frame.yuv420 <roi_x0> <roi_y0> <roi_x1> <roi_y1> session.cal
```

Check the printed MAD (median absolute deviation) isn't suspiciously
large — that would mean the ROI is catching a shadow, seam, or the
tape itself, and you should shrink the rectangle and recapture.

---

## 4. Calibrate the hole

```
rpicam-vid --codec yuv420 --width 1280 --height 720 --frames 1 -n -o hole_frame.yuv420
ffmpeg -f rawvideo -pix_fmt yuv420p -s 1280x720 -i hole_frame.yuv420 -frames:v 1 hole_frame.png
```

In `hole_frame.png`, hover the hole's center and note (x, y). Hover
one point on the hole's edge and note (x, y). Radius is the straight-
line pixel distance between the two — doesn't need to be exact.

```
./calibrate_teams hole <center_x> <center_y> <radius> session.cal
```

---

## 5. Calibrate Team A

Place **one** Team A bag on the board — nothing else on the board at
this point, since this step keeps only the single largest detected
blob as the reference.

```
rpicam-vid --codec yuv420 --width 1280 --height 720 --frames 1 -n -o teamA_frame.yuv420
./calibrate_teams team A teamA_frame.yuv420 session.cal
```

Use a distinct filename per capture (`teamA_frame.yuv420`, not a
reused `frame.yuv420`) — reusing one filename across steps is what
caused the earlier identical-team-values mixup.

---

## 6. Calibrate Team B

Remove the Team A bag, place **one** Team B bag:

```
rpicam-vid --codec yuv420 --width 1280 --height 720 --frames 1 -n -o teamB_frame.yuv420
./calibrate_teams team B teamB_frame.yuv420 session.cal
```

---

## 7. Validate

```
./calibrate_teams validate session.cal
```

Confirm:
- All three pairwise distances (background/A, background/B, A/B) say
  **OK**, not TOO CLOSE.
- Hole center/radius print and look right.

If anything's TOO CLOSE, it's a real contrast problem with these
colors/lighting, not a tool bug — see Troubleshooting below.

---

## 8. Sanity-check with `detect_bags`

Place a known number of bags (mix of both teams, one in the hole) and
capture a fresh frame:

```
rpicam-vid --codec yuv420 --width 1280 --height 720 --frames 1 -n -o check_frame.yuv420
./detect_bags check_frame.yuv420 session.cal
```

The number of `bag` lines printed should exactly match the number of
bags you placed, with correct `team=` and `in_hole=` for each. This
is the step that will immediately show whether covering the board
actually fixed the >32-blob issue — if you still see far more bags
reported than you placed, the board surface still has enough visual
variation to trip the deviation threshold and needs a flatter/more
uniform cover.

---

## Troubleshooting

**Way more blobs than bags placed (the issue you just hit):** board
surface isn't visually uniform enough — printing, texture, glare, or
shadows are each registering as their own blob. Cover more
thoroughly, or reposition lighting to kill glare/shadow, then redo
steps 1–3 (background) and 8 (recheck).

**`validate` reports TOO CLOSE:** the two colors involved are
genuinely close in U/V space under this lighting. Try more visually
distinct bag colors, or check whether the board color sits between
the two bag colors in chroma space.

**`team` step reports "no blob found meeting minimum size":** either
the bag isn't inside the ROI rectangle, or it isn't different enough
from the background to register at all — recheck bag placement and
lighting.

**Hole falls outside the ROI (a note, not an error):** expected if
the ROI was drawn conservatively away from the board edges. Just
double check the hole coordinates themselves are correct.
