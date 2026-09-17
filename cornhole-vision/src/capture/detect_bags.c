/*
 * detect_bags.c
 *
 * Single-frame bag detector for the cornhole vision project. Given one
 * captured frame and a calibration file produced by calibrate_teams
 * (background + Team A + Team B color references, ROI, and hole
 * region), finds every bag-sized blob in the ROI, classifies each by
 * team color, and reports whether its centroid falls inside the hole.
 *
 * This is a stateless, one-shot tool -- same philosophy as
 * calibrate_teams and border_detect. It does not track bags across
 * frames or decide when a "new bag" event happened; that's the job of
 * whatever polls this repeatedly (the Python game/web layer) and
 * diffs the bag list between polls, the same way the load-cell
 * version watched for step changes in weight instead of polling a
 * camera.
 *
 * Usage:
 *   detect_bags <frame.yuv420> <calib_file>
 *
 * Output (one line per bag found, to stdout):
 *   bag team=A in_hole=0 cx=612 cy=430 area=812
 *   bag team=B in_hole=1 cx=780 cy=415 area=790
 *
 * cx/cy are full-resolution (Y-plane) pixel coordinates of the blob's
 * centroid -- same coordinate system as the ROI and hole calibration
 * -- so the caller can place a marker on a display without doing any
 * inches conversion. area is in chroma-plane pixels, useful mostly
 * for sanity-checking against MIN_BLOB_AREA_FRACTION.
 *
 * Diagnostics go to stderr so stdout stays clean and easy for a
 * calling script to parse.
 *
 * Build:
 *   gcc -std=c17 -O2 -o detect_bags detect_bags.c -lm
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ---- Frame geometry: must match calibrate_teams / the capture pipeline ---- */
#define WIDTH   1280
#define HEIGHT  720
#define CHROMA_W (WIDTH  / 2)
#define CHROMA_H (HEIGHT / 2)
#define Y_SIZE   (WIDTH * HEIGHT)
#define UV_SIZE  (CHROMA_W * CHROMA_H)
#define FRAME_SIZE (Y_SIZE + 2 * UV_SIZE)

/* Same deviation/blob-size tuning as calibrate_teams -- keep these two
 * files in sync if you retune one of them. */
#define DEVIATION_MAD_MULTIPLIER 4.0
#define DEVIATION_FLOOR          8.0
#define MIN_BLOB_AREA_FRACTION   0.01

typedef struct {
    uint8_t *y, *u, *v;
} Frame;

typedef struct {
    int x0, y0, x1, y1; /* full-res (Y-plane) coordinates */
} Rect;

typedef struct {
    double u_median, v_median;
} ColorRef;

/* ---- Frame loading (identical to calibrate_teams) ---- */

static int load_frame(const char *path, Frame *f) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "error: could not open frame file '%s'\n", path);
        return -1;
    }
    uint8_t *buf = malloc(FRAME_SIZE);
    if (!buf) {
        fprintf(stderr, "error: out of memory reading frame\n");
        fclose(fp);
        return -1;
    }
    size_t n = fread(buf, 1, FRAME_SIZE, fp);
    fclose(fp);
    if (n != FRAME_SIZE) {
        fprintf(stderr,
            "error: '%s' is %zu bytes, expected %d (WIDTH=%d HEIGHT=%d) -- "
            "wrong resolution, or more than one frame in the file? "
            "(capture with --frames 1)\n",
            path, n, FRAME_SIZE, WIDTH, HEIGHT);
        free(buf);
        return -1;
    }
    f->y = buf;
    f->u = buf + Y_SIZE;
    f->v = buf + Y_SIZE + UV_SIZE;
    return 0;
}

static void free_frame(Frame *f) {
    free(f->y);
    f->y = f->u = f->v = NULL;
}

static Rect roi_to_chroma(Rect roi_yplane) {
    Rect c;
    c.x0 = roi_yplane.x0 / 2;
    c.y0 = roi_yplane.y0 / 2;
    c.x1 = roi_yplane.x1 / 2;
    c.y1 = roi_yplane.y1 / 2;
    if (c.x0 < 0) c.x0 = 0;
    if (c.y0 < 0) c.y0 = 0;
    if (c.x1 > CHROMA_W) c.x1 = CHROMA_W;
    if (c.y1 > CHROMA_H) c.y1 = CHROMA_H;
    return c;
}

/* ---- Calibration file I/O: flat key=value text (same format as calibrate_teams) ---- */

#define MAX_KEYS 64
#define KEY_LEN 32

typedef struct {
    char keys[MAX_KEYS][KEY_LEN];
    double vals[MAX_KEYS];
    int count;
} KVStore;

static void kv_load(KVStore *kv, const char *path) {
    kv->count = 0;
    FILE *fp = fopen(path, "r");
    if (!fp) return;
    char line[128];
    while (fgets(line, sizeof(line), fp)) {
        char key[KEY_LEN];
        double val;
        if (sscanf(line, "%31[^=]=%lf", key, &val) == 2) {
            if (kv->count < MAX_KEYS) {
                snprintf(kv->keys[kv->count], KEY_LEN, "%s", key);
                kv->vals[kv->count] = val;
                kv->count++;
            }
        }
    }
    fclose(fp);
}

static int kv_get(const KVStore *kv, const char *key, double *out) {
    for (int i = 0; i < kv->count; i++) {
        if (strcmp(kv->keys[i], key) == 0) {
            *out = kv->vals[i];
            return 1;
        }
    }
    return 0;
}

static int kv_require(const KVStore *kv, const char *key, double *out,
                       const char *calib_path) {
    if (kv_get(kv, key, out)) return 1;
    fprintf(stderr,
        "error: '%s' is missing '%s' -- run the full calibrate_teams "
        "workflow (background, team A, team B, hole) before using "
        "detect_bags.\n", calib_path, key);
    return 0;
}

/* ---- Blob finding: deviation mask + ALL connected components ---- */

typedef struct {
    ColorRef color;   /* median U/V of the blob */
    int cx_full, cy_full; /* centroid, full-res Y-plane coordinates */
    int area;         /* pixel count, chroma-plane */
} Blob;

#define MAX_BLOBS 32

/* Returns the number of blobs found (up to MAX_BLOBS), or -1 on error. */
static int find_all_blobs(const Frame *f, Rect roi_c, const ColorRef *bg,
                           double bg_u_mad, double bg_v_mad, Blob *blobs_out) {
    int w = roi_c.x1 - roi_c.x0;
    int h = roi_c.y1 - roi_c.y0;
    int n = w * h;
    if (n <= 0) {
        fprintf(stderr, "error: ROI is empty after clamping to frame bounds\n");
        return -1;
    }

    double combined_mad = (bg_u_mad + bg_v_mad) / 2.0;
    double threshold = combined_mad * DEVIATION_MAD_MULTIPLIER;
    if (threshold < DEVIATION_FLOOR) threshold = DEVIATION_FLOOR;

    uint8_t *mask = calloc(n, 1);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int fy = roi_c.y0 + y, fx = roi_c.x0 + x;
            double du = (double)f->u[fy * CHROMA_W + fx] - bg->u_median;
            double dv = (double)f->v[fy * CHROMA_W + fx] - bg->v_median;
            double dist = sqrt(du * du + dv * dv);
            mask[y * w + x] = (dist > threshold) ? 1 : 0;
        }
    }

    int *label = calloc(n, sizeof(int));
    int *sizes = calloc(n + 1, sizeof(int));
    int *stack = malloc(sizeof(int) * n);
    int next_label = 0;

    for (int i = 0; i < n; i++) {
        if (!mask[i] || label[i] != 0) continue;
        next_label++;
        int sp = 0;
        stack[sp++] = i;
        label[i] = next_label;
        while (sp > 0) {
            int p = stack[--sp];
            sizes[next_label]++;
            int py = p / w, px = p % w;
            static const int dx[4] = {-1, 1, 0, 0};
            static const int dy[4] = {0, 0, -1, 1};
            for (int k = 0; k < 4; k++) {
                int nx = px + dx[k], ny = py + dy[k];
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                int ni = ny * w + nx;
                if (mask[ni] && label[ni] == 0) {
                    label[ni] = next_label;
                    stack[sp++] = ni;
                }
            }
        }
    }

    int min_area = (int)(n * MIN_BLOB_AREA_FRACTION);
    int found = 0;

    for (int l = 1; l <= next_label && found < MAX_BLOBS; l++) {
        if (sizes[l] < min_area) continue; /* too small: noise, not a bag */

        long u_sum = 0, v_sum = 0, x_sum = 0, y_sum = 0;
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (label[i] != l) continue;
            int y = i / w, x = i % w;
            int fy = roi_c.y0 + y, fx = roi_c.x0 + x;
            u_sum += f->u[fy * CHROMA_W + fx];
            v_sum += f->v[fy * CHROMA_W + fx];
            x_sum += fx;
            y_sum += fy;
            count++;
        }
        /* Mean, not median, here: this is a per-blob centroid/color
         * summary for classification, not a calibration reference, and
         * a plain mean over the whole blob is fine for that -- the
         * outlier-robustness that matters (rejecting noise pixels) is
         * already handled by keeping only labeled blob members. */
        blobs_out[found].color.u_median = (double)u_sum / count;
        blobs_out[found].color.v_median = (double)v_sum / count;
        /* centroid in chroma-plane, then to full-res by *2 */
        blobs_out[found].cx_full = (int)((double)x_sum / count) * 2;
        blobs_out[found].cy_full = (int)((double)y_sum / count) * 2;
        blobs_out[found].area = count;
        found++;
    }

    if (next_label >= MAX_BLOBS) {
        fprintf(stderr,
            "warning: found more than %d distinct blobs; only reporting the "
            "first %d. Check for noise (shadows, glare) creating spurious "
            "blobs.\n", MAX_BLOBS, MAX_BLOBS);
    }

    free(mask);
    free(label);
    free(sizes);
    free(stack);
    return found;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: detect_bags <frame.yuv420> <calib_file>\n");
        return 1;
    }
    const char *frame_path = argv[1];
    const char *calib_path = argv[2];

    KVStore kv;
    kv_load(&kv, calib_path);

    double rx0, ry0, rx1, ry1;
    double bg_u, bg_v, bg_u_mad, bg_v_mad;
    double a_u, a_v, b_u, b_v;
    double hx, hy, hr;
    int ok = 1;
    ok &= kv_require(&kv, "roi_x0", &rx0, calib_path);
    ok &= kv_require(&kv, "roi_y0", &ry0, calib_path);
    ok &= kv_require(&kv, "roi_x1", &rx1, calib_path);
    ok &= kv_require(&kv, "roi_y1", &ry1, calib_path);
    ok &= kv_require(&kv, "bg_u_median", &bg_u, calib_path);
    ok &= kv_require(&kv, "bg_v_median", &bg_v, calib_path);
    ok &= kv_require(&kv, "bg_u_mad", &bg_u_mad, calib_path);
    ok &= kv_require(&kv, "bg_v_mad", &bg_v_mad, calib_path);
    ok &= kv_require(&kv, "teamA_u_median", &a_u, calib_path);
    ok &= kv_require(&kv, "teamA_v_median", &a_v, calib_path);
    ok &= kv_require(&kv, "teamB_u_median", &b_u, calib_path);
    ok &= kv_require(&kv, "teamB_v_median", &b_v, calib_path);
    ok &= kv_require(&kv, "hole_cx", &hx, calib_path);
    ok &= kv_require(&kv, "hole_cy", &hy, calib_path);
    ok &= kv_require(&kv, "hole_radius", &hr, calib_path);
    if (!ok) return 1;

    Rect roi_y = { .x0 = (int)rx0, .y0 = (int)ry0, .x1 = (int)rx1, .y1 = (int)ry1 };
    Rect roi_c = roi_to_chroma(roi_y);
    ColorRef bg = { .u_median = bg_u, .v_median = bg_v };
    ColorRef teamA = { .u_median = a_u, .v_median = a_v };
    ColorRef teamB = { .u_median = b_u, .v_median = b_v };

    Frame f;
    if (load_frame(frame_path, &f) != 0) return 1;

    Blob blobs[MAX_BLOBS];
    int count = find_all_blobs(&f, roi_c, &bg, bg_u_mad, bg_v_mad, blobs);
    free_frame(&f);
    if (count < 0) return 1;

    for (int i = 0; i < count; i++) {
        double da = hypot(blobs[i].color.u_median - teamA.u_median,
                           blobs[i].color.v_median - teamA.v_median);
        double db = hypot(blobs[i].color.u_median - teamB.u_median,
                           blobs[i].color.v_median - teamB.v_median);
        const char *team = (da <= db) ? "A" : "B";

        double dx = blobs[i].cx_full - hx;
        double dy = blobs[i].cy_full - hy;
        int in_hole = (dx * dx + dy * dy) <= (hr * hr);

        printf("bag team=%s in_hole=%d cx=%d cy=%d area=%d\n",
               team, in_hole, blobs[i].cx_full, blobs[i].cy_full, blobs[i].area);
    }

    fprintf(stderr, "info: %d bag(s) detected\n", count);
    return 0;
}
