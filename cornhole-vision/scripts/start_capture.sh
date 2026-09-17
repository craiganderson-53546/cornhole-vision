#!/usr/bin/env bash
# Creates the FIFO if needed and starts rpicam-vid writing raw YUV420 into it.
# Usage: ./scripts/start_capture.sh [width] [height]
set -euo pipefail

FIFO=/tmp/cornhole_cam.fifo
WIDTH=${1:-1280}
HEIGHT=${2:-720}

if [[ ! -p "$FIFO" ]]; then
    mkfifo "$FIFO"
    echo "Created FIFO at $FIFO"
fi

echo "Starting rpicam-vid -> $FIFO (${WIDTH}x${HEIGHT}, yuv420)"
rpicam-vid -n -t 0 --width "$WIDTH" --height "$HEIGHT" --codec yuv420 -o "$FIFO" &
echo "rpicam-vid PID: $!  (kill with: kill $!)"
