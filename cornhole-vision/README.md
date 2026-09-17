# Cornhole vision tracking

Feasibility project: detect and count red/blue cornhole bags resting on a
regulation 2'x4' board using an overhead Raspberry Pi 5 + IMX708 camera,
written in C. See the full phased project plan for context and status.

## Layout

- `src/main.c` — current entry point. Right now this is the Phase 1
  capture/frame-rate test program; it'll get split into `src/capture/`,
  `src/calibration/`, and `src/vision/` as those phases land.
- `scripts/start_capture.sh` — creates the FIFO and starts `rpicam-vid`
  writing raw YUV420 into it.
- `scripts/verify_frame.sh` — converts a dumped raw frame to PNG with
  ffmpeg so you can eyeball it.
- `data/calibration/` — tracked in git: board-corner coordinates, color
  threshold configs, anything small you'll want history on.
- `data/captures/` — gitignored: raw YUV dumps and test frames, these get
  large fast and don't belong in version control.

## Build & run

```bash
make                              # builds build/cornhole_vision
./scripts/start_capture.sh        # starts rpicam-vid -> FIFO (defaults 1280x720)
./build/cornhole_vision /tmp/cornhole_cam.fifo 1280 720
```

Or just press Ctrl+Shift+B in VS Code to build, then F5 to build-and-debug
under gdb (Remote-SSH session runs the debugger on the Pi directly, so
there's no cross-compile debugging complexity).

To sanity-check a dumped frame:

```bash
./scripts/verify_frame.sh first_frame.yuv 1280 720 frame.png
```
