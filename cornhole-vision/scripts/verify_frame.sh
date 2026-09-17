#!/usr/bin/env bash
# Converts a dumped raw YUV420 frame to PNG for visual inspection.
# Usage: ./scripts/verify_frame.sh [in.yuv] [width] [height] [out.png]
set -euo pipefail

IN=${1:-first_frame.yuv}
WIDTH=${2:-1280}
HEIGHT=${3:-720}
OUT=${4:-frame.png}

ffmpeg -y -f rawvideo -pix_fmt yuv420p -s "${WIDTH}x${HEIGHT}" -i "$IN" "$OUT"
echo "Wrote $OUT"
