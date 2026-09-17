// border_detect.c (v2)
//
// Phase 2 step 5: detects the board's contrasting border in the Y (luma)
// plane and refines four rough corner estimates into precise pixel
// coordinates.
//
// Why v2: the original design used a fixed search window applied
// uniformly across an entire edge. Real captures can have edges that
// slant significantly across the frame (camera perspective), which broke
// that assumption -- a single window either missed part of the edge or
// exceeded the scan buffer size. v2 instead takes your four rough corner
// estimates (easy to read off frame.png by eye) and, for each edge,
// interpolates the *expected* line position at each sample point along
// it -- so each individual search window stays small regardless of how
// much the edge slants overall.
//
// Usage:
//   ./border_detect <frame.yuv> <width> <height> <corners.txt> [n_samples] [search_radius]
//
// corners.txt: four lines, name + rough pixel coordinates, any order:
//   TL 92 151
//   TR 1205 33
//   BR 1234 719
//   BL 87 674
//
// n_samples (default 8): how many scanlines to place along each edge.
// search_radius (default 30): how far on either side of the interpolated
// expected position to search on each sample -- widen this if a lot of
// samples come back "no hit", narrow it if you're picking up noise.

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#define DEVIATION_THRESHOLD 30
#define MIN_RUN_LENGTH 3
#define MAX_ACCEPTABLE_WIDTH 30
#define MAX_SAMPLES 128
#define MAX_POINTS_PER_SIDE 32
#define MAX_RUNS 24
#define DEFAULT_N_SAMPLES 8
#define DEFAULT_SEARCH_RADIUS 30
// Fraction of the edge to inset from each true corner before placing the
// first/last sample, so samples don't land on the noisy corner junction.
#define EDGE_INSET 0.12

typedef struct {
    int width, height;
    size_t y_size, uv_size, frame_size;
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
    if (!f) { perror("fopen"); return -1; }
    size_t n = fread(g->buf, 1, g->frame_size, f);
    fclose(f);
    if (n != g->frame_size) {
        fprintf(stderr, "Expected %zu bytes, read %zu -- wrong width/height?\n", g->frame_size, n);
        return -1;
    }
    return 0;
}

// ---- border detection on a single scanline ----

typedef struct { int found; int position; int width; int peak_deviation; } edge_hit_t;

static int cmp_uint8(const void *a, const void *b) {
    return (int)(*(const uint8_t *)a) - (int)(*(const uint8_t *)b);
}

static uint8_t median_of(const uint8_t *values, int count) {
    uint8_t tmp[MAX_SAMPLES];
    memcpy(tmp, values, (size_t)count);
    qsort(tmp, (size_t)count, sizeof(uint8_t), cmp_uint8);
    return tmp[count / 2];
}

static edge_hit_t find_transition(const uint8_t *values, int count) {
    edge_hit_t hit = {0, 0, 0, 0};
    if (count < MIN_RUN_LENGTH || count > MAX_SAMPLES) return hit;

    uint8_t med = median_of(values, count);

    // Collect every qualifying run (>= MIN_RUN_LENGTH consecutive samples
    // deviating from the median), rather than only tracking the longest.
    // A wide glare/reflection streak is often longer than the real border,
    // so "longest wins" is the wrong selection rule outdoors.
    int run_starts[MAX_RUNS], run_ends[MAX_RUNS];
    int n_runs = 0;
    int run_start = -1;

    for (int i = 0; i < count; i++) {
        int dev = abs((int)values[i] - (int)med);
        if (dev >= DEVIATION_THRESHOLD) {
            if (run_start < 0) run_start = i;
        } else if (run_start >= 0) {
            int len = i - run_start;
            if (len >= MIN_RUN_LENGTH && n_runs < MAX_RUNS) {
                run_starts[n_runs] = run_start;
                run_ends[n_runs] = i - 1;
                n_runs++;
            }
            run_start = -1;
        }
    }
    if (run_start >= 0) {
        int len = count - run_start;
        if (len >= MIN_RUN_LENGTH && n_runs < MAX_RUNS) {
            run_starts[n_runs] = run_start;
            run_ends[n_runs] = count - 1;
            n_runs++;
        }
    }

    if (n_runs == 0) return hit;

    // Pick the run with the strongest peak deviation, among runs whose
    // width isn't implausibly large (rejects wide glare/reflection
    // streaks). Deviation strength is a better signal than width -- real
    // border width varies with viewing angle across the frame, so
    // matching a fixed expected width actively discards good detections
    // on sides where perspective foreshortens the border differently.
    int best_run = -1, best_peak_dev = -1;
    for (int r = 0; r < n_runs; r++) {
        int width = run_ends[r] - run_starts[r] + 1;
        if (width > MAX_ACCEPTABLE_WIDTH) continue;

        int peak_dev = 0;
        for (int i = run_starts[r]; i <= run_ends[r]; i++) {
            int dev = abs((int)values[i] - (int)med);
            if (dev > peak_dev) peak_dev = dev;
        }
        if (peak_dev > best_peak_dev) { best_peak_dev = peak_dev; best_run = r; }
    }
    if (best_run < 0) return hit; // every run exceeded the width sanity bound

    int best_start = run_starts[best_run], best_end = run_ends[best_run];
    int peak_idx = best_start, peak_dev = 0;
    for (int i = best_start; i <= best_end; i++) {
        int dev = abs((int)values[i] - (int)med);
        if (dev > peak_dev) { peak_dev = dev; peak_idx = i; }
    }
    hit.found = 1;
    hit.position = peak_idx;
    hit.width = best_end - best_start + 1;
    hit.peak_deviation = peak_dev;
    return hit;
}

static edge_hit_t scan_row(const frame_geometry_t *g, int y, int x0, int x1) {
    if (y < 0 || y >= g->height) { edge_hit_t h = {0,0,0,0}; return h; }
    if (x0 < 0) x0 = 0;
    if (x1 > g->width) x1 = g->width;
    int count = x1 - x0;
    if (count > MAX_SAMPLES) count = MAX_SAMPLES;
    if (count < MIN_RUN_LENGTH) { edge_hit_t h = {0,0,0,0}; return h; }
    uint8_t buf[MAX_SAMPLES];
    for (int i = 0; i < count; i++) buf[i] = g->buf[(size_t)y * g->width + (x0 + i)];
    edge_hit_t hit = find_transition(buf, count);
    if (hit.found) hit.position += x0;
    return hit;
}

static edge_hit_t scan_col(const frame_geometry_t *g, int x, int y0, int y1) {
    if (x < 0 || x >= g->width) { edge_hit_t h = {0,0,0,0}; return h; }
    if (y0 < 0) y0 = 0;
    if (y1 > g->height) y1 = g->height;
    int count = y1 - y0;
    if (count > MAX_SAMPLES) count = MAX_SAMPLES;
    if (count < MIN_RUN_LENGTH) { edge_hit_t h = {0,0,0,0}; return h; }
    uint8_t buf[MAX_SAMPLES];
    for (int i = 0; i < count; i++) buf[i] = g->buf[(size_t)(y0 + i) * g->width + x];
    edge_hit_t hit = find_transition(buf, count);
    if (hit.found) hit.position += y0;
    return hit;
}

// ---- line fitting (total least squares, works at any orientation) ----

typedef struct { double a, b, c; } line_t; // a*x + b*y = c

static int fit_line(const double *xs, const double *ys, int n, line_t *out) {
    if (n < 2) return 0;
    double mx = 0, my = 0;
    for (int i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
    mx /= n; my /= n;

    double sxx = 0, syy = 0, sxy = 0;
    for (int i = 0; i < n; i++) {
        double dx = xs[i] - mx, dy = ys[i] - my;
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }

    double t2 = (sxx + syy) / 2.0;
    double det = sxx * syy - sxy * sxy;
    double under = t2 * t2 - det;
    double disc = sqrt(under < 0 ? 0 : under);
    double lambda_min = t2 - disc;

    double a, b;
    if (fabs(sxy) > 1e-9) {
        a = 1.0;
        b = -(sxx - lambda_min) / sxy;
    } else {
        if (sxx < syy) { a = 1.0; b = 0.0; } else { a = 0.0; b = 1.0; }
    }
    double norm = sqrt(a * a + b * b);
    a /= norm; b /= norm;

    out->a = a; out->b = b; out->c = a * mx + b * my;
    return 1;
}

static int line_intersect(line_t l1, line_t l2, double *x, double *y) {
    double det = l1.a * l2.b - l2.a * l1.b;
    if (fabs(det) < 1e-9) return 0;
    *x = (l1.c * l2.b - l2.c * l1.b) / det;
    *y = (l1.a * l2.c - l2.a * l1.c) / det;
    return 1;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

// Fits an initial line, then discards points whose perpendicular distance
// from it is unusually large (more than 3x the median residual, floored
// at 10px so a naturally tight fit doesn't over-reject) and refits from
// the remainder. A single bad point near one end of an edge can otherwise
// drag the whole line -- and the corner it's extrapolated to -- well off.
static int fit_line_robust(const double *xs, const double *ys, int n, const char *name, line_t *out) {
    line_t initial;
    if (!fit_line(xs, ys, n, &initial)) return 0;
    if (n <= 2) { *out = initial; return 1; }

    double residuals[MAX_POINTS_PER_SIDE];
    double sorted[MAX_POINTS_PER_SIDE];
    for (int i = 0; i < n; i++) {
        residuals[i] = fabs(initial.a * xs[i] + initial.b * ys[i] - initial.c);
        sorted[i] = residuals[i];
    }
    qsort(sorted, (size_t)n, sizeof(double), cmp_double);
    double median_res = sorted[n / 2];
    double threshold = median_res * 3.0;
    if (threshold < 10.0) threshold = 10.0;

    double fxs[MAX_POINTS_PER_SIDE], fys[MAX_POINTS_PER_SIDE];
    int fn = 0;
    for (int i = 0; i < n; i++) {
        if (residuals[i] <= threshold) {
            fxs[fn] = xs[i]; fys[fn] = ys[i]; fn++;
        } else {
            fprintf(stderr, "  [%s] rejecting outlier point (%.0f,%.0f), residual=%.1f (threshold=%.1f)\n",
                    name, xs[i], ys[i], residuals[i], threshold);
        }
    }

    if (fn >= 2 && fn < n) {
        fit_line(fxs, fys, fn, out);
        fprintf(stderr, "[%s] refit after outlier rejection: kept %d/%d points\n", name, fn, n);
        return 1;
    }
    *out = initial;
    return 1;
}

// ---- edge scanning driven by two rough corner points ----

typedef struct { double x, y; } point_t;

// Scans one edge between two rough corner estimates, interpolating the
// expected line position at each sample so the search window stays
// narrow even if the edge slants a lot across the frame.
static int scan_edge(const frame_geometry_t *g, point_t c0, point_t c1,
                      int n_samples, int radius, const char *name, line_t *out_line) {
    double dx = c1.x - c0.x, dy = c1.y - c0.y;
    int mostly_horizontal = fabs(dx) >= fabs(dy);

    double xs[MAX_POINTS_PER_SIDE], ys[MAX_POINTS_PER_SIDE];
    int n = 0;

    for (int i = 0; i < n_samples && n < MAX_POINTS_PER_SIDE; i++) {
        double t = EDGE_INSET + (1.0 - 2.0 * EDGE_INSET) * i / (n_samples - 1);
        double ex = c0.x + dx * t;
        double ey = c0.y + dy * t;

        edge_hit_t hit;
        double px, py;
        if (mostly_horizontal) {
            // edge runs mostly left-right -> scan vertically at this x
            int x = (int)lround(ex);
            hit = scan_col(g, x, (int)lround(ey - radius), (int)lround(ey + radius));
            px = x; py = hit.position;
        } else {
            // edge runs mostly top-bottom -> scan horizontally at this y
            int y = (int)lround(ey);
            hit = scan_row(g, y, (int)lround(ex - radius), (int)lround(ex + radius));
            px = hit.position; py = y;
        }

        if (hit.found) {
            fprintf(stderr, "  [%s] expected~(%.0f,%.0f) -> hit at (%.0f,%.0f) width=%d dev=%d\n",
                    name, ex, ey, px, py, hit.width, hit.peak_deviation);
            xs[n] = px; ys[n] = py; n++;
        } else {
            fprintf(stderr, "  [%s] expected~(%.0f,%.0f) -> no hit (radius=%d)\n", name, ex, ey, radius);
        }
    }

    if (n < 2) {
        fprintf(stderr, "[%s] only %d valid points, need at least 2. Try widening search_radius.\n", name, n);
        return 0;
    }
    fit_line_robust(xs, ys, n, name, out_line);
    fprintf(stderr, "[%s] fitted from %d/%d points: %.4fx + %.4fy = %.2f\n",
            name, n, n_samples, out_line->a, out_line->b, out_line->c);
    return 1;
}

static int parse_corners(const char *path, point_t *tl, point_t *tr, point_t *br, point_t *bl) {
    FILE *f = fopen(path, "r");
    if (!f) { perror("fopen (corners file)"); return -1; }

    int found = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        char name[8];
        double x, y;
        if (sscanf(line, "%7s %lf %lf", name, &x, &y) != 3) continue;

        if (strcmp(name, "TL") == 0) { tl->x = x; tl->y = y; found |= 1; }
        else if (strcmp(name, "TR") == 0) { tr->x = x; tr->y = y; found |= 2; }
        else if (strcmp(name, "BR") == 0) { br->x = x; br->y = y; found |= 4; }
        else if (strcmp(name, "BL") == 0) { bl->x = x; bl->y = y; found |= 8; }
    }
    fclose(f);

    if (found != 15) {
        fprintf(stderr, "corners file must define TL, TR, BR, BL (found mask=%d).\n", found);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 7) {
        fprintf(stderr, "Usage: %s <frame.yuv> <width> <height> <corners.txt> [n_samples] [search_radius]\n", argv[0]);
        return 1;
    }

    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    int n_samples = (argc >= 6) ? atoi(argv[5]) : DEFAULT_N_SAMPLES;
    int radius = (argc >= 7) ? atoi(argv[6]) : DEFAULT_SEARCH_RADIUS;

    if (n_samples < 2 || n_samples > MAX_POINTS_PER_SIDE) {
        fprintf(stderr, "n_samples must be between 2 and %d.\n", MAX_POINTS_PER_SIDE);
        return 1;
    }
    if (radius < 5 || radius * 2 > MAX_SAMPLES) {
        fprintf(stderr, "search_radius must be between 5 and %d.\n", MAX_SAMPLES / 2);
        return 1;
    }

    frame_geometry_t g;
    if (!init_geometry(&g, width, height)) { fprintf(stderr, "alloc failed\n"); return 1; }
    if (load_frame(&g, argv[1]) != 0) { free(g.buf); return 1; }

    point_t tl, tr, br, bl;
    if (parse_corners(argv[4], &tl, &tr, &br, &bl) != 0) { free(g.buf); return 1; }

    line_t top, right, bottom, left;
    int ok = 1;
    fprintf(stderr, "\n--- scanning: top (TL->TR) ---\n");
    ok &= scan_edge(&g, tl, tr, n_samples, radius, "top", &top);
    fprintf(stderr, "\n--- scanning: right (TR->BR) ---\n");
    ok &= scan_edge(&g, tr, br, n_samples, radius, "right", &right);
    fprintf(stderr, "\n--- scanning: bottom (BR->BL) ---\n");
    ok &= scan_edge(&g, br, bl, n_samples, radius, "bottom", &bottom);
    fprintf(stderr, "\n--- scanning: left (BL->TL) ---\n");
    ok &= scan_edge(&g, bl, tl, n_samples, radius, "left", &left);
    free(g.buf);

    if (!ok) {
        fprintf(stderr, "\nOne or more edges failed to fit -- widen search_radius and retry.\n");
        return 1;
    }

    double x, y;
    printf("\n--- refined board corners (pixel coordinates) ---\n");
    if (line_intersect(top, left, &x, &y)) printf("TL: (%.1f, %.1f)\n", x, y);
    if (line_intersect(top, right, &x, &y)) printf("TR: (%.1f, %.1f)\n", x, y);
    if (line_intersect(bottom, right, &x, &y)) printf("BR: (%.1f, %.1f)\n", x, y);
    if (line_intersect(bottom, left, &x, &y)) printf("BL: (%.1f, %.1f)\n", x, y);

    return 0;
}
