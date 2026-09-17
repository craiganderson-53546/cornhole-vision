// yuv_line_probe.c
//
// Phase 2 step 4 tool: samples Y, U, V values along a single row or column
// of a dumped I420 frame and prints them as CSV. Point this across the
// board's contrasting border to see the actual luminance/chroma transition
// profile -- tells you (a) how many pixels wide the border is at your
// mount distance/angle, and (b) which plane (Y, U, or V) gives the
// cleanest separation from the board surface. Feed the CSV into Excel or
// gnuplot and look for a clean spike/step where the line crosses.
//
// Usage:
//   ./yuv_line_probe <file.yuv> <width> <height> row <y>   > row.csv
//   ./yuv_line_probe <file.yuv> <width> <height> col <x>   > col.csv
//
// Example: board's front edge is roughly 400px down the frame ->
//   ./yuv_line_probe first_frame.yuv 1280 720 row 400 > row400.csv

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

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

static int load_frame(const frame_geometry_t *g, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("fopen");
        return -1;
    }
    size_t n = fread(g->buf, 1, g->frame_size, f);
    fclose(f);
    if (n != g->frame_size) {
        fprintf(stderr, "Expected %zu bytes, read %zu -- wrong width/height for this file?\n",
                g->frame_size, n);
        return -1;
    }
    return 0;
}

// Y is full resolution. U/V are subsampled 2x in each dimension (4:2:0),
// so every 2x2 block of Y pixels shares one U and one V sample.
static void sample_pixel(const frame_geometry_t *g, int x, int y,
                          int *out_y, int *out_u, int *out_v) {
    *out_y = g->buf[(size_t)y * g->width + x];
    int uv_w = g->width / 2;
    size_t uv_index = (size_t)(y / 2) * uv_w + (x / 2);
    *out_u = g->buf[g->y_size + uv_index];
    *out_v = g->buf[g->y_size + g->uv_size + uv_index];
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <file.yuv> <width> <height> row|col <index>\n", argv[0]);
        fprintf(stderr, "Example: %s first_frame.yuv 1280 720 row 400 > row400.csv\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    const char *mode = argv[4];
    int index = atoi(argv[5]);

    if (width <= 0 || height <= 0 || (width % 2) || (height % 2)) {
        fprintf(stderr, "Width/height must be positive and even.\n");
        return 1;
    }
    if (strcmp(mode, "row") != 0 && strcmp(mode, "col") != 0) {
        fprintf(stderr, "Mode must be 'row' or 'col'.\n");
        return 1;
    }

    frame_geometry_t g;
    if (!init_geometry(&g, width, height)) {
        fprintf(stderr, "Failed to allocate frame buffer.\n");
        return 1;
    }
    if (load_frame(&g, path) != 0) {
        free(g.buf);
        return 1;
    }

    int is_row = (strcmp(mode, "row") == 0);
    int limit = is_row ? width : height;

    if (is_row && (index < 0 || index >= height)) {
        fprintf(stderr, "Row index out of range (0-%d).\n", height - 1);
        free(g.buf);
        return 1;
    }
    if (!is_row && (index < 0 || index >= width)) {
        fprintf(stderr, "Column index out of range (0-%d).\n", width - 1);
        free(g.buf);
        return 1;
    }

    printf("position,Y,U,V\n");
    for (int i = 0; i < limit; i++) {
        int x = is_row ? i : index;
        int y = is_row ? index : i;
        int yv, uv, vv;
        sample_pixel(&g, x, y, &yv, &uv, &vv);
        printf("%d,%d,%d,%d\n", i, yv, uv, vv);
    }

    free(g.buf);
    return 0;
}
