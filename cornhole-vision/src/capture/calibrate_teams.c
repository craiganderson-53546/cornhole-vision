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
 *        calibrate_teams background frame_empty.yuv420 ROI_X0 ROI_Y0 ROI_X1 ROI_Y1 session.cal
 *   2. One Team A bag placed anywhere on the board:
 *        calibrate_teams team A frame_teamA.yuv420 session.cal
 *   3. One Team B bag placed anywhere on the board:
 *        calibrate_teams team B frame_teamB.yuv420 session.cal
 *   4. Check the result:
 *        calibrate_teams validate session.cal
 *
 * ROI_X0/Y0/X1/Y1 are the board-interior corners in full-resolution
 * (Y-plane) pixel coordinates -- i.e. the same interior boundary that
 * border_detect already locates. Wire border_detect's output into step 1
 * instead of typing coordinates by hand once that hookup is convenient;
 * this tool takes a rectangle for now rather than assuming knowledge of
 * border_detect's exact data structure. A rectangular ROI is a safe
 * conservative choice: it should sit fully inside the true board
 * interior polygon so every sampled pixel is guaranteed to be board
 * surface or bag, never background outside the board.
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

static void sample_background(const Frame *f, Rect roi_c, ColorRef *out) {
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
            uvals[idx] = f->u[y * CHROMA_W + x];
            vvals[idx] = f->v[y * CHROMA_W + x];
            idx++;
        }
    }
    median_and_mad(uvals, n, &out->u_median, &out->u_mad);
    median_and_mad(vvals, n, &out->v_median, &out->v_mad);
    free(uvals);
    free(vvals);
    out->has_data = 1;
}

/* ---- Bag blob extraction: deviation mask + largest connected component ---- */

static int find_bag_blob(const Frame *f, Rect roi_c, const ColorRef *bg,
                          ColorRef *out) {
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

static int cmd_background(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr,
            "usage: calibrate_teams background <frame.yuv420> "
            "<roi_x0> <roi_y0> <roi_x1> <roi_y1> <calib_file>\n"
            "  roi_* are full-resolution (Y-plane) pixel coordinates of a "
            "rectangle that sits safely inside the board interior.\n");
        return 1;
    }
    const char *frame_path = argv[2];
    Rect roi_y = {
        .x0 = atoi(argv[3]), .y0 = atoi(argv[4]),
        .x1 = atoi(argv[5]), .y1 = atoi(argv[6])
    };
    const char *calib_path = argv[7];

    Frame f;
    if (load_frame(frame_path, &f) != 0) return 1;

    Rect roi_c = roi_to_chroma(roi_y);
    ColorRef bg;
    sample_background(&f, roi_c, &bg);
    free_frame(&f);
    if (!bg.has_data) return 1;

    KVStore kv;
    kv_load(&kv, calib_path); /* start fresh or reuse existing file if present */
    kv_set(&kv, "roi_x0", roi_y.x0);
    kv_set(&kv, "roi_y0", roi_y.y0);
    kv_set(&kv, "roi_x1", roi_y.x1);
    kv_set(&kv, "roi_y1", roi_y.y1);
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

    Frame f;
    if (load_frame(frame_path, &f) != 0) return 1;

    ColorRef bag;
    int rc = find_bag_blob(&f, roi_c, &bg, &bag);
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
    kv_save(&kv, calib_path);

    printf("Team %s: U median=%.1f (MAD %.1f)  V median=%.1f (MAD %.1f)\n",
           team, bag.u_median, bag.u_mad, bag.v_median, bag.v_mad);
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
     * just a warning -- the ROI rectangle is deliberately conservative and
     * may not itself reach all the way to the hole, so it's not an error. */
    double rx0, ry0, rx1, ry1;
    if (kv_get(&kv, "roi_x0", &rx0) && kv_get(&kv, "roi_y0", &ry0) &&
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
        "  %s background <frame.yuv420> <roi_x0> <roi_y0> <roi_x1> <roi_y1> <calib_file>\n"
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
