/*
 * calibrate_teams.c
 *
 * Field calibration routine for the cornhole vision project.
 *
 * Problem it solves: development so far has used red/blue bags on a
 * solid-color board, which made U/V thresholds easy to pick by hand.
 * A real end user can show up with any bag colors and any board color.
 * This tool builds the U/V reference for "board background", "Team A
 * bag", and "Team B bag" from real captured frames -- no assumptions
 * about which colors are involved -- and then reports whether there's
 * enough chroma separation between all three to classify reliably.
 *
 * Field workflow (three captures, in order):
 *   1. Empty board, no bags:
 *        calibrate_teams background frame_empty.yuv420 X0 Y0 X1 Y1 X2 Y2 X3 Y3 session.cal
 *   2. One Team A bag placed anywhere on the board:
 *        calibrate_teams team A frame_teamA.yuv420 session.cal
 *   3. One Team B bag placed anywhere on the board:
 *        calibrate_teams team B frame_teamB.yuv420 session.cal
 *   4. Check the result:
 *        calibrate_teams validate session.cal
 *
 * X0,Y0 .. X3,Y3 are the board's 4 interior corners, in full-resolution
 * (Y-plane) pixel coordinates -- the same interior boundary that
 * border_detect already locates -- given in order around the perimeter
 * (clockwise or counter-clockwise, doesn't matter which, just don't
 * cross them). Wire border_detect's output into step 1 instead of
 * clicking coordinates by hand once that hookup is convenient. A
 * photographed board is rarely axis-aligned in the raw frame (camera
 * mount angle, perspective), so all 4 corners matter: collapsing to 2
 * opposite corners of a bounding box -- the old behavior here --
 * quietly stretches the sampled/scanned region away from the board's
 * true shape, worse the further a point sits from wherever those 2
 * corners happened to be. The tool still derives and stores that
 * bounding box too (as roi_x0/y0/x1/y1), since it's a cheap, useful
 * outer limit for iteration and for any older tooling that only reads
 * the rectangle -- but background/bag sampling itself is now masked
 * to the true quad, and detect_bags does the same for on-board
 * scanning.
 *
 * Input frames: raw I420 YUV420 planar, WIDTH x HEIGHT, same format
 * everything else in this project reads off the FIFO. Grab one frame
 * to a file however you're already doing that (verify_frame.sh, a
 * one-shot rpicam-vid capture, etc.) and pass the path in.
 *
 * Design notes:
 *  - Uses median + median-absolute-deviation (MAD) rather than mean/
 *    stddev for the same reason border_detect's line fit rejects
 *    outliers by median residual: a handful of glare/shadow pixels at
 *    a bag's edge shouldn't drag the reference color off target.
 *  - "Is this pixel a bag?" is decided by U/V distance from the
 *    background reference, not by any assumed color. This is what
 *    makes the tool color-agnostic.
 *  - A raw deviation mask alone is noisy (stray pixels, board texture,
 *    shadow edges). This tool runs connected-component labeling
 *    (iterative flood fill, 4-connected, no OpenCV) and keeps only the
 *    single largest blob, which should be the bag itself. A minimum
 *    blob size is enforced so a few scattered noisy pixels can't pass
 *    as a bag.
 *  - The calibration file is a flat key=value text file so it's easy
 *    to eyeball, diff, or read from the C capture loop later.
 *
 * Build:
 *   gcc -std=c17 -O2 -D_POSIX_C_SOURCE=199309L -o calibrate_teams calibrate_teams.c -lm
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ---- Frame geometry: must match the rest of the capture pipeline ---- */
#define WIDTH   1280
#define HEIGHT  720
#define CHROMA_W (WIDTH  / 2)
#define CHROMA_H (HEIGHT / 2)
#define Y_SIZE   (WIDTH * HEIGHT)
#define UV_SIZE  (CHROMA_W * CHROMA_H)
#define FRAME_SIZE (Y_SIZE + 2 * UV_SIZE)

/* ---- Tunable thresholds ---- */

/* A pixel counts as "deviant from background" if its U/V distance from
 * the background median exceeds this many combined-MAD units, floored
 * so a suspiciously flat/clean background frame can't make the
 * detector oversensitive. Same floor-and-multiply pattern as
 * border_detect's DEVIATION_THRESHOLD. */
#define DEVIATION_MAD_MULTIPLIER 4.0
#define DEVIATION_FLOOR          8.0

/* Minimum blob size, as a fraction of the ROI area, to be accepted as
 * "the bag" rather than noise. A regulation bag is roughly 6"x6"; this
 * is deliberately loose since ROI size in pixels depends on camera
 * mount distance -- tune after seeing real numbers from your rig. */
#define MIN_BLOB_AREA_FRACTION 0.01

/* Minimum acceptable U/V distance between any two reference colors
 * (background, Team A, Team B) for the field setup to be considered
 * usable. This is a starting point, not a physical constant -- adjust
 * once you've seen how the classifier behaves near this margin. */
#define MIN_SEPARATION 20.0

/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t *y, *u, *v;
} Frame;

typedef struct {
    int x0, y0, x1, y1; /* full-res (Y-plane) coordinates */
} Rect;

typedef struct {
    double x, y;
} Pt;

/* Point-in-convex-quad test. `quad` is 4 points given in order around
 * the perimeter (either winding direction, just not crossed/diagonal
 * order) -- true for any real board's corners, however skewed by
 * camera perspective, since a photographed rectangle is still convex.
 * Works by checking the test point falls on the same side of every
 * edge; a mismatched sign on any edge means it's outside. */
static int point_in_quad(double px, double py, const Pt quad[4]) {
    double sign = 0.0;
    for (int i = 0; i < 4; i++) {
        Pt a = quad[i], b = quad[(i + 1) % 4];
        double ex = b.x - a.x, ey = b.y - a.y;
        double cx = px - a.x, cy = py - a.y;
        double cross = ex * cy - ey * cx;
        if (i == 0) {
            sign = cross;
        } else if (cross * sign < 0.0) {
            return 0;
        }
    }
    return 1;
}

/* Same halving the ROI rect already does, applied to each quad corner
 * so point_in_quad can be evaluated directly against chroma-plane
 * pixel coordinates without re-scaling on every call. */
static void quad_to_chroma(const Pt full[4], Pt chroma_out[4]) {
    for (int i = 0; i < 4; i++) {
        chroma_out[i].x = full[i].x / 2.0;
        chroma_out[i].y = full[i].y / 2.0;
    }
}

typedef struct {
    double u_median, v_median;
    double u_mad, v_mad;
    int has_data;
} ColorRef;

/* ---- Frame loading ---- */

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
            "wrong resolution or truncated capture?\n",
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
    free(f->y); /* single allocation backs y/u/v */
    f->y = f->u = f->v = NULL;
}

/* ---- Robust stats: median and median-absolute-deviation ---- */

static int cmp_int(const void *a, const void *b) {
    return (*(const int *)a) - (*(const int *)b);
}

static double median_of(int *vals, int n) {
    if (n <= 0) return 0.0;
    qsort(vals, n, sizeof(int), cmp_int);
    if (n % 2 == 1) return vals[n / 2];
    return (vals[n / 2 - 1] + vals[n / 2]) / 2.0;
}

/* Computes median and MAD of vals[] in one call. vals[] is sorted as a
 * side effect (median_of sorts it); a scratch buffer is used for the
 * absolute-deviation pass so the caller's array order for vals itself
 * doesn't need to survive. */
static void median_and_mad(int *vals, int n, double *median_out, double *mad_out) {
    double med = median_of(vals, n);
    int *dev = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        dev[i] = (int)fabs((double)vals[i] - med);
    }
    double mad = median_of(dev, n);
    free(dev);
    *median_out = med;
    *mad_out = mad;
}

/* ---- ROI helpers ---- */

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

/* ---- Background sampling (whole ROI assumed to be board surface) ---- */

static void sample_background(const Frame *f, Rect roi_c, const Pt *quad_c,
                               ColorRef *out) {
    int w = roi_c.x1 - roi_c.x0;
    int h = roi_c.y1 - roi_c.y0;
    int n = w * h;
    if (n <= 0) {
        fprintf(stderr, "error: ROI is empty after clamping to frame bounds\n");
        out->has_data = 0;
        return;
    }
    int *uvals = malloc(sizeof(int) * n);
    int *vvals = malloc(sizeof(int) * n);
    int idx = 0;
    for (int y = roi_c.y0; y < roi_c.y1; y++) {
        for (int x = roi_c.x0; x < roi_c.x1; x++) {
            /* The bounding box is a safe outer limit for iteration, but
             * for a board that isn't axis-aligned in the frame (camera
             * perspective), its own corners can fall outside the true
             * board quad -- skip those so a sliver of carpet/border
             * near a bbox corner never drags the background median. */
            if (quad_c != NULL && !point_in_quad((double)x, (double)y, quad_c))
                continue;
            uvals[idx] = f->u[y * CHROMA_W + x];
            vvals[idx] = f->v[y * CHROMA_W + x];
            idx++;
        }
    }
    if (idx == 0) {
        fprintf(stderr,
            "error: no pixels fell inside the board quad -- check the "
            "corner order/coordinates\n");
        out->has_data = 0;
        free(uvals);
        free(vvals);
        return;
    }
    median_and_mad(uvals, idx, &out->u_median, &out->u_mad);
    median_and_mad(vvals, idx, &out->v_median, &out->v_mad);
    free(uvals);
    free(vvals);
    out->has_data = 1;
}

/* ---- Bag blob extraction: deviation mask + largest connected component ---- */

static int find_bag_blob(const Frame *f, Rect roi_c, const ColorRef *bg,
                          const Pt *quad_c, ColorRef *out, int *area_out) {
    int w = roi_c.x1 - roi_c.x0;
    int h = roi_c.y1 - roi_c.y0;
    int n = w * h;
    if (n <= 0) {
        fprintf(stderr, "error: ROI is empty after clamping to frame bounds\n");
        return -1;
    }

    double combined_mad = (bg->u_mad + bg->v_mad) / 2.0;
    double threshold = combined_mad * DEVIATION_MAD_MULTIPLIER;
    if (threshold < DEVIATION_FLOOR) threshold = DEVIATION_FLOOR;

    uint8_t *mask = calloc(n, 1);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int fy = roi_c.y0 + y, fx = roi_c.x0 + x;
            if (quad_c != NULL && !point_in_quad((double)fx, (double)fy, quad_c))
                continue; /* outside the true board quad -- never a bag pixel */
            double du = (double)f->u[fy * CHROMA_W + fx] - bg->u_median;
            double dv = (double)f->v[fy * CHROMA_W + fx] - bg->v_median;
            double dist = sqrt(du * du + dv * dv);
            mask[y * w + x] = (dist > threshold) ? 1 : 0;
        }
    }

    /* Connected components via iterative flood fill (4-connected). */
    int *label = calloc(n, sizeof(int));
    int *sizes = calloc(n + 1, sizeof(int)); /* sizes[label] = pixel count */
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

    int best_label = 0, best_size = 0;
    for (int l = 1; l <= next_label; l++) {
        if (sizes[l] > best_size) {
            best_size = sizes[l];
            best_label = l;
        }
    }

    int min_area = (int)(n * MIN_BLOB_AREA_FRACTION);
    int result = 0;
    if (best_label == 0 || best_size < min_area) {
        fprintf(stderr,
            "warning: no blob found meeting minimum size (%d px, need >= %d px "
            "out of %d ROI px). Is the bag actually inside the ROI, and is it "
            "different enough from the board to trigger detection at all?\n",
            best_size, min_area, n);
        result = -1;
    } else {
        int *uvals = malloc(sizeof(int) * best_size);
        int *vvals = malloc(sizeof(int) * best_size);
        int idx = 0;
        for (int i = 0; i < n; i++) {
            if (label[i] == best_label) {
                int y = i / w, x = i % w;
                int fy = roi_c.y0 + y, fx = roi_c.x0 + x;
                uvals[idx] = f->u[fy * CHROMA_W + fx];
                vvals[idx] = f->v[fy * CHROMA_W + fx];
                idx++;
            }
        }
        median_and_mad(uvals, best_size, &out->u_median, &out->u_mad);
        median_and_mad(vvals, best_size, &out->v_median, &out->v_mad);
        free(uvals);
        free(vvals);
        out->has_data = 1;
        if (area_out != NULL) *area_out = best_size;
        fprintf(stderr, "info: bag blob found, %d px (%.1f%% of ROI)\n",
                best_size, 100.0 * best_size / n);
        result = 0;
    }

    free(mask);
    free(label);
    free(sizes);
    free(stack);
    return result;
}

/* ---- Calibration file I/O: flat key=value text ---- */

#define MAX_KEYS 64
#define KEY_LEN 32

typedef struct {
    char keys[MAX_KEYS][KEY_LEN];
    double vals[MAX_KEYS];
    int count;
} KVStore;

static void kv_set(KVStore *kv, const char *key, double val) {
    for (int i = 0; i < kv->count; i++) {
        if (strcmp(kv->keys[i], key) == 0) {
            kv->vals[i] = val;
            return;
        }
    }
    if (kv->count >= MAX_KEYS) {
        fprintf(stderr, "error: calibration file has too many keys\n");
        return;
    }
    strncpy(kv->keys[kv->count], key, KEY_LEN - 1);
    kv->keys[kv->count][KEY_LEN - 1] = '\0';
    kv->vals[kv->count] = val;
    kv->count++;
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

/* The board's true 4 corners (full-res Y-plane coords), in order
 * around the perimeter -- as opposed to roi_x0/y0/x1/y1, which is
 * just their bounding box, kept for any older tooling that only
 * knows the rectangle. Returns 0 (and leaves out[] untouched) if a
 * calibration file predates this and only has the bounding box --
 * callers should treat that as "no quad available, fall back to the
 * bbox" rather than an error. */
static int kv_get_corners(const KVStore *kv, Pt out[4]) {
    static const char *names[4][2] = {
        {"roi_c0x", "roi_c0y"}, {"roi_c1x", "roi_c1y"},
        {"roi_c2x", "roi_c2y"}, {"roi_c3x", "roi_c3y"},
    };
    for (int i = 0; i < 4; i++) {
        if (!kv_get(kv, names[i][0], &out[i].x)) return 0;
        if (!kv_get(kv, names[i][1], &out[i].y)) return 0;
    }
    return 1;
}

static void kv_set_corners(KVStore *kv, const Pt corners[4]) {
    static const char *names[4][2] = {
        {"roi_c0x", "roi_c0y"}, {"roi_c1x", "roi_c1y"},
        {"roi_c2x", "roi_c2y"}, {"roi_c3x", "roi_c3y"},
    };
    for (int i = 0; i < 4; i++) {
        kv_set(kv, names[i][0], corners[i].x);
        kv_set(kv, names[i][1], corners[i].y);
    }
}

static void kv_load(KVStore *kv, const char *path) {
    kv->count = 0;
    FILE *fp = fopen(path, "r");
    if (!fp) return; /* fine if it doesn't exist yet (first call: background) */
    char line[128];
    while (fgets(line, sizeof(line), fp)) {
        char key[KEY_LEN];
        double val;
        if (sscanf(line, "%31[^=]=%lf", key, &val) == 2) {
            kv_set(kv, key, val);
        }
    }
    fclose(fp);
}

static void kv_save(const KVStore *kv, const char *path) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "error: could not write calibration file '%s'\n", path);
        return;
    }
    for (int i = 0; i < kv->count; i++) {
        fprintf(fp, "%s=%.4f\n", kv->keys[i], kv->vals[i]);
    }
    fclose(fp);
}

static void colorref_to_kv(KVStore *kv, const char *prefix, const ColorRef *c) {
    char key[KEY_LEN];
    snprintf(key, sizeof(key), "%s_u_median", prefix); kv_set(kv, key, c->u_median);
    snprintf(key, sizeof(key), "%s_v_median", prefix); kv_set(kv, key, c->v_median);
    snprintf(key, sizeof(key), "%s_u_mad", prefix);    kv_set(kv, key, c->u_mad);
    snprintf(key, sizeof(key), "%s_v_mad", prefix);    kv_set(kv, key, c->v_mad);
}

static int kv_to_colorref(const KVStore *kv, const char *prefix, ColorRef *c) {
    char key[KEY_LEN];
    double u_med, v_med, u_mad, v_mad;
    snprintf(key, sizeof(key), "%s_u_median", prefix);
    if (!kv_get(kv, key, &u_med)) return 0;
    snprintf(key, sizeof(key), "%s_v_median", prefix);
    if (!kv_get(kv, key, &v_med)) return 0;
    snprintf(key, sizeof(key), "%s_u_mad", prefix);
    if (!kv_get(kv, key, &u_mad)) return 0;
    snprintf(key, sizeof(key), "%s_v_mad", prefix);
    if (!kv_get(kv, key, &v_mad)) return 0;
    c->u_median = u_med; c->v_median = v_med;
    c->u_mad = u_mad;    c->v_mad = v_mad;
    c->has_data = 1;
    return 1;
}

static double chroma_distance(const ColorRef *a, const ColorRef *b) {
    double du = a->u_median - b->u_median;
    double dv = a->v_median - b->v_median;
    return sqrt(du * du + dv * dv);
}

/* ---- Subcommands ---- */

/* Detects a self-intersecting ("bowtie") corner order -- e.g. entering
 * TL,TR,BL,BR instead of walking the perimeter TL,TR,BR,BL -- which
 * point_in_quad can't interpret as a simple interior/exterior split.
 * Checks that every consecutive pair of edges turns the same
 * direction (all cross products the same sign); a valid quad, walked
 * consistently clockwise or counter-clockwise, always does. */
static int quad_is_simple(const Pt q[4]) {
    double sign = 0.0;
    for (int i = 0; i < 4; i++) {
        Pt a = q[i], b = q[(i + 1) % 4], c = q[(i + 2) % 4];
        double e1x = b.x - a.x, e1y = b.y - a.y;
        double e2x = c.x - b.x, e2y = c.y - b.y;
        double turn = e1x * e2y - e1y * e2x;
        if (i == 0) {
            sign = turn;
        } else if (turn * sign < 0.0) {
            return 0;
        }
    }
    return 1;
}

static int cmd_background(int argc, char **argv) {
    if (argc != 12) {
        fprintf(stderr,
            "usage: calibrate_teams background <frame.yuv420> "
            "<x0> <y0> <x1> <y1> <x2> <y2> <x3> <y3> <calib_file>\n"
            "  x0,y0 .. x3,y3 are the board's 4 interior corners, in "
            "full-resolution (Y-plane) pixel coordinates, given in order "
            "around the perimeter (clockwise or counter-clockwise -- just "
            "not crossed). A rectangle photographed off-axis is still a "
            "convex quad, so all 4 corners matter even if the board looks "
            "trapezoidal in the raw frame; don't collapse it to 2 opposite "
            "corners of a bounding box.\n");
        return 1;
    }
    const char *frame_path = argv[2];
    Pt corners_full[4] = {
        { atof(argv[3]),  atof(argv[4])  },
        { atof(argv[5]),  atof(argv[6])  },
        { atof(argv[7]),  atof(argv[8])  },
        { atof(argv[9]),  atof(argv[10]) },
    };
    const char *calib_path = argv[11];

    if (!quad_is_simple(corners_full)) {
        fprintf(stderr,
            "error: these 4 corners cross over themselves (a 'bowtie'), "
            "not a simple board outline -- most likely two of them are "
            "out of order. Walk the perimeter in one direction, e.g. "
            "top-left, top-right, bottom-right, bottom-left -- don't "
            "jump diagonally (top-left, top-right, bottom-LEFT, "
            "bottom-right is the mistake this usually is).\n");
        return 1;
    }

    Rect roi_y = {
        .x0 = (int)corners_full[0].x, .y0 = (int)corners_full[0].y,
        .x1 = (int)corners_full[0].x, .y1 = (int)corners_full[0].y,
    };
    for (int i = 1; i < 4; i++) {
        if (corners_full[i].x < roi_y.x0) roi_y.x0 = (int)corners_full[i].x;
        if (corners_full[i].x > roi_y.x1) roi_y.x1 = (int)corners_full[i].x;
        if (corners_full[i].y < roi_y.y0) roi_y.y0 = (int)corners_full[i].y;
        if (corners_full[i].y > roi_y.y1) roi_y.y1 = (int)corners_full[i].y;
    }

    Frame f;
    if (load_frame(frame_path, &f) != 0) return 1;

    Rect roi_c = roi_to_chroma(roi_y);
    Pt quad_c[4];
    quad_to_chroma(corners_full, quad_c);
    ColorRef bg;
    sample_background(&f, roi_c, quad_c, &bg);
    free_frame(&f);
    if (!bg.has_data) return 1;

    KVStore kv;
    kv_load(&kv, calib_path); /* start fresh or reuse existing file if present */
    /* Bounding box: kept for any tooling that only reads the rectangle
     * and for iteration bounds; it's derived from the true corners
     * below, not independently entered, so it can never drift out of
     * sync with them the way it could when both were typed by hand. */
    kv_set(&kv, "roi_x0", roi_y.x0);
    kv_set(&kv, "roi_y0", roi_y.y0);
    kv_set(&kv, "roi_x1", roi_y.x1);
    kv_set(&kv, "roi_y1", roi_y.y1);
    kv_set_corners(&kv, corners_full);
    colorref_to_kv(&kv, "bg", &bg);
    kv_save(&kv, calib_path);

    printf("background: U median=%.1f (MAD %.1f)  V median=%.1f (MAD %.1f)\n",
           bg.u_median, bg.u_mad, bg.v_median, bg.v_mad);
    printf("saved to '%s'. Next: place a Team A bag and run the 'team A' step.\n",
           calib_path);
    return 0;
}

static int cmd_team(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr,
            "usage: calibrate_teams team <A|B> <frame.yuv420> <calib_file>\n"
            "  requires the 'background' step to have already been run "
            "against this calib_file.\n");
        return 1;
    }
    const char *team = argv[2];
    if (strcmp(team, "A") != 0 && strcmp(team, "B") != 0) {
        fprintf(stderr, "error: team must be 'A' or 'B', got '%s'\n", team);
        return 1;
    }
    const char *frame_path = argv[3];
    const char *calib_path = argv[4];

    KVStore kv;
    kv_load(&kv, calib_path);

    double rx0, ry0, rx1, ry1;
    if (!kv_get(&kv, "roi_x0", &rx0) || !kv_get(&kv, "roi_y0", &ry0) ||
        !kv_get(&kv, "roi_x1", &rx1) || !kv_get(&kv, "roi_y1", &ry1)) {
        fprintf(stderr,
            "error: '%s' has no ROI -- run the 'background' step first.\n",
            calib_path);
        return 1;
    }
    ColorRef bg;
    if (!kv_to_colorref(&kv, "bg", &bg)) {
        fprintf(stderr,
            "error: '%s' has no background reference -- run the 'background' "
            "step first.\n", calib_path);
        return 1;
    }

    Rect roi_y = { .x0 = (int)rx0, .y0 = (int)ry0, .x1 = (int)rx1, .y1 = (int)ry1 };
    Rect roi_c = roi_to_chroma(roi_y);
    Pt corners_full[4], quad_c[4];
    int have_quad = kv_get_corners(&kv, corners_full);
    if (have_quad) quad_to_chroma(corners_full, quad_c);

    Frame f;
    if (load_frame(frame_path, &f) != 0) return 1;

    ColorRef bag;
    int bag_area = 0;
    int rc = find_bag_blob(&f, roi_c, &bg, have_quad ? quad_c : NULL, &bag, &bag_area);
    free_frame(&f);
    if (rc != 0) {
        fprintf(stderr,
            "calibration for Team %s FAILED -- no usable bag blob. Confirm "
            "the bag is inside the ROI and try again.\n", team);
        return 1;
    }

    char prefix[8];
    snprintf(prefix, sizeof(prefix), "team%s", team);
    colorref_to_kv(&kv, prefix, &bag);
    /* This single bag's pixel area becomes the reference detect_bags
     * uses to tell "one bag" from "two or three touching bags flood-
     * filled into one blob" -- see MAX_BLOB_AREA handling there. Place
     * the calibration bag flat and by itself, same as for color, or
     * this reference (and every split it drives) will be off. */
    char area_key[16];
    snprintf(area_key, sizeof(area_key), "team%s_ref_area", team);
    kv_set(&kv, area_key, (double)bag_area);
    kv_save(&kv, calib_path);

    printf("Team %s: U median=%.1f (MAD %.1f)  V median=%.1f (MAD %.1f)\n",
           team, bag.u_median, bag.u_mad, bag.v_median, bag.v_mad);
    printf("single-bag reference area: %d px -- used to detect touching/"
           "overlapping bags of the same color.\n", bag_area);
    printf("saved to '%s'.\n", calib_path);
    return 0;
}

static void report_pair(const char *name_a, const ColorRef *a,
                         const char *name_b, const ColorRef *b, int *all_ok) {
    double d = chroma_distance(a, b);
    int ok = d >= MIN_SEPARATION;
    if (!ok) *all_ok = 0;
    printf("  %-14s vs %-14s: distance=%.1f  %s\n",
           name_a, name_b, d, ok ? "OK" : "TOO CLOSE");
}

static int cmd_validate(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: calibrate_teams validate <calib_file>\n");
        return 1;
    }
    const char *calib_path = argv[2];
    KVStore kv;
    kv_load(&kv, calib_path);

    ColorRef bg, teamA, teamB;
    int have_bg = kv_to_colorref(&kv, "bg", &bg);
    int have_a  = kv_to_colorref(&kv, "teamA", &teamA);
    int have_b  = kv_to_colorref(&kv, "teamB", &teamB);

    if (!have_bg || !have_a || !have_b) {
        fprintf(stderr,
            "error: '%s' is incomplete (background=%s, teamA=%s, teamB=%s). "
            "Run all three calibration steps first.\n", calib_path,
            have_bg ? "yes" : "missing", have_a ? "yes" : "missing",
            have_b ? "yes" : "missing");
        return 1;
    }

    printf("Chroma separation (minimum required: %.1f):\n", MIN_SEPARATION);
    int all_ok = 1;
    report_pair("background", &bg, "Team A", &teamA, &all_ok);
    report_pair("background", &bg, "Team B", &teamB, &all_ok);
    report_pair("Team A", &teamA, "Team B", &teamB, &all_ok);

    printf("\nField calibration: %s\n", all_ok ? "PASS" : "FAIL");

    double hx, hy, hr;
    if (kv_get(&kv, "hole_cx", &hx) && kv_get(&kv, "hole_cy", &hy) &&
        kv_get(&kv, "hole_radius", &hr)) {
        printf("Hole: center=(%.0f, %.0f) radius=%.0f px\n", hx, hy, hr);
    } else {
        printf("Hole: not calibrated yet (run the 'hole' step before using "
               "detect_bags).\n");
    }

    if (!all_ok) {
        printf(
            "One or more pairs are too close in U/V space to classify "
            "reliably. Options: pick more visually distinct bag colors, "
            "check for a board color that happens to sit between the two "
            "bag colors in chroma space, or lower MIN_SEPARATION only after "
            "confirming the classifier still behaves acceptably at that "
            "margin in real play.\n");
    }
    return all_ok ? 0 : 1;
}

/* ---- main ---- */

static int cmd_hole(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr,
            "usage: calibrate_teams hole <cx> <cy> <radius> <calib_file>\n"
            "  cx, cy, radius are full-resolution (Y-plane) pixel coordinates, "
            "same coordinate system as the ROI corners. Pick them the same way: "
            "open the PNG, hover the hole's center and one point on its edge, "
            "and use the distance between them as the radius.\n");
        return 1;
    }
    double cx = atof(argv[2]);
    double cy = atof(argv[3]);
    double radius = atof(argv[4]);
    const char *calib_path = argv[5];

    if (radius <= 0) {
        fprintf(stderr, "error: radius must be positive, got %.1f\n", radius);
        return 1;
    }

    KVStore kv;
    kv_load(&kv, calib_path);

    /* Sanity check against the ROI if one is already set, since the hole
     * should sit inside the board interior the ROI was drawn from. This is
     * just a warning -- the ROI is deliberately conservative and may not
     * itself reach all the way to the hole, so it's not an error. Prefer
     * the true quad when the calibration file has one; it's a tighter,
     * more accurate check than the bounding box for a board that isn't
     * axis-aligned in the frame. */
    double rx0, ry0, rx1, ry1;
    Pt corners_full[4];
    int have_quad = kv_get_corners(&kv, corners_full);
    if (have_quad) {
        if (!point_in_quad(cx, cy, corners_full)) {
            fprintf(stderr,
                "note: hole center (%.0f, %.0f) falls outside the calibrated "
                "board quad. That's fine if the ROI is drawn conservatively "
                "away from the hole -- just confirm the hole coordinates "
                "themselves are right.\n", cx, cy);
        }
    } else if (kv_get(&kv, "roi_x0", &rx0) && kv_get(&kv, "roi_y0", &ry0) &&
        kv_get(&kv, "roi_x1", &rx1) && kv_get(&kv, "roi_y1", &ry1)) {
        if (cx < rx0 || cx > rx1 || cy < ry0 || cy > ry1) {
            fprintf(stderr,
                "note: hole center (%.0f, %.0f) falls outside the calibration "
                "ROI (%.0f,%.0f)-(%.0f,%.0f). That's fine if the ROI is drawn "
                "conservatively away from the hole -- just confirm the hole "
                "coordinates themselves are right.\n", cx, cy, rx0, ry0, rx1, ry1);
        }
    }

    kv_set(&kv, "hole_cx", cx);
    kv_set(&kv, "hole_cy", cy);
    kv_set(&kv, "hole_radius", radius);
    kv_save(&kv, calib_path);

    printf("hole: center=(%.0f, %.0f) radius=%.0f px (full-res)\n", cx, cy, radius);
    printf("saved to '%s'.\n", calib_path);
    return 0;
}

static void print_top_usage(const char *prog) {
    fprintf(stderr,
        "usage:\n"
        "  %s background <frame.yuv420> <x0> <y0> <x1> <y1> <x2> <y2> <x3> <y3> <calib_file>\n"
        "  %s team <A|B> <frame.yuv420> <calib_file>\n"
        "  %s hole <cx> <cy> <radius> <calib_file>\n"
        "  %s validate <calib_file>\n",
        prog, prog, prog, prog);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_top_usage(argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "background") == 0) return cmd_background(argc, argv);
    if (strcmp(argv[1], "team") == 0)       return cmd_team(argc, argv);
    if (strcmp(argv[1], "hole") == 0)       return cmd_hole(argc, argv);
    if (strcmp(argv[1], "validate") == 0)   return cmd_validate(argc, argv);

    fprintf(stderr, "error: unknown subcommand '%s'\n", argv[1]);
    print_top_usage(argv[0]);
    return 1;
}