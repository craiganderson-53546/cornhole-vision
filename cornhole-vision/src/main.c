// yuv_frame_reader.c
//
// Phase 1 feasibility test: reads raw YUV420 (I420 planar) frames from a
// named pipe fed by rpicam-vid, verifies frame geometry, dumps the first
// frame to disk for visual inspection, and reports sustained frame rate.
//
// Usage:
//   mkfifo /tmp/cornhole_cam.fifo
//   rpicam-vid -n -t 0 --width 1280 --height 720 --codec yuv420 -o /tmp/cornhole_cam.fifo &
//   ./yuv_frame_reader /tmp/cornhole_cam.fifo 1280 720
//
// Assumes tightly-packed I420: Y plane is width*height bytes, U and V
// planes are each (width/2)*(height/2) bytes, with no row padding. This
// holds for rpicam-vid's --codec yuv420 raw output at common resolutions
// (1280x720, 1920x1080). If the dumped frame looks skewed/garbled when
// converted, that's the first sign of stride padding you'll need to
// account for -- see the note in the project plan's Phase 1 write-up.

// Needed under strict -std=c17: without this, glibc hides clock_gettime,
// CLOCK_MONOTONIC, and other POSIX declarations that aren't part of the
// bare ISO C standard.
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>

typedef struct {
    int width;
    int height;
    size_t y_size;
    size_t uv_size;
    size_t frame_size;
    uint8_t *buf;
} frame_geometry_t;

static int init_geometry(frame_geometry_t *g, int width, int height) {
    g->width = width;
    g->height = height;
    g->y_size = (size_t)width * (size_t)height;
    g->uv_size = (size_t)(width / 2) * (size_t)(height / 2);
    g->frame_size = g->y_size + 2 * g->uv_size;
    g->buf = malloc(g->frame_size);
    return g->buf != NULL;
}

// Blocking read of exactly frame_size bytes, looping over short reads.
// Returns 1 on a complete frame, 0 on clean EOF (writer closed pipe),
// -1 on error.
static int read_full_frame(int fd, uint8_t *buf, size_t frame_size) {
    size_t total = 0;
    while (total < frame_size) {
        ssize_t n = read(fd, buf + total, frame_size - total);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("read");
            return -1;
        }
        if (n == 0) {
            return 0; // writer closed the pipe (rpicam-vid exited/stream ended)
        }
        total += (size_t)n;
    }
    return 1;
}

static double timespec_diff_s(const struct timespec *a, const struct timespec *b) {
    return (double)(b->tv_sec - a->tv_sec) + (double)(b->tv_nsec - a->tv_nsec) / 1e9;
}

static void dump_frame(const frame_geometry_t *g, const char *out_path) {
    FILE *f = fopen(out_path, "wb");
    if (!f) {
        perror("fopen (dump)");
        return;
    }
    fwrite(g->buf, 1, g->frame_size, f);
    fclose(f);
    fprintf(stderr,
        "[dump] wrote one frame to %s (%dx%d, %zu bytes). Verify with:\n"
        "  ffmpeg -f rawvideo -pix_fmt yuv420p -s %dx%d -i %s frame.png\n",
        out_path, g->width, g->height, g->frame_size, g->width, g->height, out_path);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <fifo_path> <width> <height> [dump_path]\n", argv[0]);
        fprintf(stderr, "Example: %s /tmp/cornhole_cam.fifo 1280 720 first_frame.yuv\n", argv[0]);
        return 1;
    }

    const char *fifo_path = argv[1];
    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    const char *dump_path = (argc >= 5) ? argv[4] : "first_frame.yuv";

    if (width <= 0 || height <= 0 || (width % 2) || (height % 2)) {
        fprintf(stderr, "Width/height must be positive and even (4:2:0 subsampling).\n");
        return 1;
    }

    frame_geometry_t g;
    if (!init_geometry(&g, width, height)) {
        fprintf(stderr, "Failed to allocate %zu bytes for frame buffer.\n",
                (size_t)width * (size_t)height * 3 / 2);
        return 1;
    }

    fprintf(stderr, "[init] %dx%d I420 -> Y=%zu U=V=%zu total=%zu bytes/frame\n",
            width, height, g.y_size, g.uv_size, g.frame_size);

    fprintf(stderr, "[init] opening FIFO %s (blocks until a writer connects)...\n", fifo_path);
    int fd = open(fifo_path, O_RDONLY);
    if (fd < 0) {
        perror("open (fifo)");
        fprintf(stderr, "Did you create it first? mkfifo %s\n", fifo_path);
        free(g.buf);
        return 1;
    }
    fprintf(stderr, "[init] writer connected, reading frames.\n");

    int dumped = 0;
    long frame_count = 0;
    long window_count = 0;
    struct timespec t_start, t_now, t_window;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    t_window = t_start;

    for (;;) {
        int rc = read_full_frame(fd, g.buf, g.frame_size);
        if (rc < 0) {
            fprintf(stderr, "[error] read failed, aborting.\n");
            break;
        }
        if (rc == 0) {
            fprintf(stderr, "[eof] writer closed the pipe after %ld frames.\n", frame_count);
            break;
        }

        frame_count++;
        window_count++;

        if (!dumped) {
            dump_frame(&g, dump_path);
            dumped = 1;
        }

        clock_gettime(CLOCK_MONOTONIC, &t_now);
        double window_elapsed = timespec_diff_s(&t_window, &t_now);
        if (window_elapsed >= 1.0) {
            double overall_elapsed = timespec_diff_s(&t_start, &t_now);
            fprintf(stderr,
                "[stats] frame %-6ld  instant fps=%.1f  avg fps=%.1f\n",
                frame_count,
                (double)window_count / window_elapsed,
                (double)frame_count / overall_elapsed);
            window_count = 0;
            t_window = t_now;
        }
    }

    close(fd);
    free(g.buf);
    return 0;
}

