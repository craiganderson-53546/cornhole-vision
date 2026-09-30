/*
 * detect_board.c
 *
 * Automated board-geometry detection for field calibration (Tier 2).
 * Given one captured frame and a single point clicked on the board's
 * high-contrast border, this finds:
 *
 *   - the board's 4 true interior corners, by locating every pixel
 *     that matches the border's color and taking the 4 extremal
 *     points of that region (closest to each of the 4 diagonal
 *     directions) -- the same corner format calibrate_teams'
 *     `background` step already takes by hand
 *   - the hole's center and radius, by locating the largest
 *     contiguous dark region strictly inside those 4 corners
 *
 * This replaces measuring and typing 8 coordinates and a hole
 * position/radius by hand -- the previous, error-prone (see the
 * bowtie-corner-order incident) way of getting calibrate_teams its
 * geometry. The output feeds directly into calibrate_teams background
 * and calibrate_teams hole.
 *
 * Why border color, not generic edge/contrast detection: a fixed
 * camera mount means the board's geometry in the frame is stable
 * session to session, but ambient lighting is not -- generic
 * gradient/contrast edge-finding (the earlier border_detect.c
 * approach) was lighting-sensitive (reliable on 2 edges, not the
 * other 2, indoors). Color segmentation against a *known, calibrated*
 * border color is a much more forgiving problem: it only needs the
 * border to be a consistent color, not for edges to produce a strong
 * gradient under whatever light happens to be available that day.
 *
 * Why the hole needs Y (luminance), unlike everything else in this
 * project: bag and background classification deliberately look only
 * at U/V chroma, ignoring Y, specifically so shadows don't affect
 * classification. But a black hole against a white board can have
 * near-identical, close-to-neutral chroma -- brightness is the only
 * thing that actually distinguishes them. This is the one place in
 * the pipeline that intentionally uses Y.
 *
 * Usage:
 *   detect_board <frame.yuv420> <border_sample_x> <border_sample_y>
 *
 * border_sample_x/y is one point clicked on the border tape itself
 * (full-resolution Y-plane coordinates), used to sample its color the
 * same way calibrate_teams samples background/bag color.
 *
 * Output (to stdout):
 *   inner 0 <x> <y>      -- interior-to-tape boundary (4 lines)
 *   inner 1 <x> <y>
 *   inner 2 <x> <y>
 *   inner 3 <x> <y>
 *   outer 0 <x> <y>      -- tape-to-carpet boundary (4 lines)
 *   outer 1 <x> <y>
 *   outer 2 <x> <y>
 *   outer 3 <x> <y>
 *   hole <cx> <cy> <radius>
 *
 * Both boundaries matter because the rules score the tape border the
 * same as the rest of the board surface -- calibrate_teams' updated
 * background step needs the inner boundary to sample the interior's
 * color and the outer boundary to sample the tape's, as two separate
 * "this is board, not a bag" references (see its updated docstring).
 *
 * Corners are in perimeter order (the same walking-the-edge order
 * calibrate_teams' quad_is_simple check requires), full-resolution
 * Y-plane coordinates. If the hole can't be found confidently, the
 * `hole` line is omitted -- run calibrate_teams hole by hand for that
 * session, same as if this tool didn't exist.
 *
 * Diagnostics go to stderr, same convention as the rest of this
 * project.
 *
 * Build:
 *   gcc -std=c17 -O2 -o detect_board detect_board.c -lm
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#define WIDTH   1280
#define HEIGHT  720
#define CHROMA_W (WIDTH  / 2)
#define CHROMA_H (HEIGHT / 2)
#define Y_SIZE   (WIDTH * HEIGHT)
#define UV_SIZE  (CHROMA_W * CHROMA_H)
#define FRAME_SIZE (Y_SIZE + 2 * UV_SIZE)

/* Same tuning family as detect_bags/calibrate_teams. The border check
 * can afford to be a little tighter than bag detection (a known,
 * uniform tape color sampled fresh each session) since it's not
 * fighting shadow/glare variation across a whole board surface. */
#define BORDER_DEVIATION_MAD_MULTIPLIER 4.0
#define BORDER_DEVIATION_FLOOR          6.0
#define BORDER_MIN_PIXEL_FRACTION       0.01  /* of the whole frame */
#define BORDER_MAX_PIXEL_FRACTION       0.30  /* of the whole frame */

#define HOLE_DEVIATION_MAD_MULTIPLIER   4.0
#define HOLE_DEVIATION_FLOOR            10.0
#define HOLE_MIN_AREA_FRACTION          0.001 /* of the quad interior */
#define HOLE_MAX_AREA_FRACTION          0.15  /* of the quad interior */

typedef struct {
    uint8_t *y, *u, *v;
} Frame;

typedef struct {
    double x, y;
} Pt;

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

static int cmp_int(const void *a, const void *b) {
    return (*(const int *)a) - (*(const int *)b);
}

/* Same median/MAD as the rest of the project (duplicated, not shared
 * -- consistent with how roi_to_chroma etc. are already duplicated
 * across these files rather than pulled into a shared header). */
static void median_and_mad(int *vals, int n, double *median, double *mad) {
    qsort(vals, n, sizeof(int), cmp_int);
    *median = (n % 2) ? vals[n / 2]
                       : (vals[n / 2 - 1] + vals[n / 2]) / 2.0;
    int *dev = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        dev[i] = abs((int)(vals[i] - *median));
    }
    qsort(dev, n, sizeof(int), cmp_int);
    *mad = (n % 2) ? dev[n / 2] : (dev[n / 2 - 1] + dev[n / 2]) / 2.0;
    free(dev);
}

/* Point-in-convex-quad test, same as calibrate_teams.c/detect_bags.c. */
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

/* ---- Corner refinement: rough corners -> sub-pixel precise ones ----
 *
 * Ported from border_detect.c's edge-sampling/robust-line-fit/line-
 * intersection approach, which is considerably more precise than a
 * single extremal pixel (the flood-fill corners above, on their own,
 * are only as good as whichever single pixel happened to be most
 * extreme -- one noisy pixel and that's the corner). border_detect.c
 * found its per-sample transition via Y-luma deviation-run detection,
 * tuned for raw pixel noise; here the per-sample signal is instead a
 * lookup into the classification masks this file already computes
 * (is_border / interior / playing_surface), which are a much cleaner
 * signal than raw luma since they're already a color decision, not a
 * brightness one -- so the per-sample detector below is simpler than
 * border_detect's run-detection (nearest mask transition to the
 * expected position) rather than needing the same machinery. The
 * multi-sample line fit + outlier rejection + line-intersection part
 * is unchanged in spirit from border_detect.c. */

#define REFINE_N_SAMPLES     8
#define REFINE_EDGE_INSET    0.12
#define REFINE_SEARCH_RADIUS_CHROMA 15  /* ~30 full-res px, like border_detect's default */
#define REFINE_MAX_POINTS    32

typedef struct { double a, b, c; } Line; /* a*x + b*y = c, full-res coords */

static int fit_line(const double *xs, const double *ys, int n, Line *out) {
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

static int cmp_double_rc(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

/* Fits an initial line, discards points whose residual exceeds 3x the
 * median (floored at 10px), refits from the remainder -- same outlier
 * story as border_detect.c: a single bad sample point shouldn't drag
 * the whole edge (and the corner it's intersected into) off. */
static int fit_line_robust(const double *xs, const double *ys, int n,
                            const char *name, Line *out) {
    Line initial;
    if (!fit_line(xs, ys, n, &initial)) return 0;
    if (n <= 2) { *out = initial; return 1; }

    double residuals[REFINE_MAX_POINTS], sorted[REFINE_MAX_POINTS];
    for (int i = 0; i < n; i++) {
        residuals[i] = fabs(initial.a * xs[i] + initial.b * ys[i] - initial.c);
        sorted[i] = residuals[i];
    }
    qsort(sorted, (size_t)n, sizeof(double), cmp_double_rc);
    double median_res = sorted[n / 2];
    double threshold = median_res * 3.0;
    if (threshold < 10.0) threshold = 10.0;

    double fxs[REFINE_MAX_POINTS], fys[REFINE_MAX_POINTS];
    int fn = 0;
    for (int i = 0; i < n; i++) {
        if (residuals[i] <= threshold) { fxs[fn] = xs[i]; fys[fn] = ys[i]; fn++; }
        else {
            fprintf(stderr,
                "  [%s] rejecting outlier sample (%.0f,%.0f), residual=%.1f "
                "(threshold=%.1f)\n", name, xs[i], ys[i], residuals[i], threshold);
        }
    }
    if (fn >= 2 && fn < n) {
        fit_line(fxs, fys, fn, out);
        return 1;
    }
    *out = initial;
    return 1;
}

static int line_intersect(Line l1, Line l2, double *x, double *y) {
    double det = l1.a * l2.b - l2.a * l1.b;
    if (fabs(det) < 1e-9) return 0;
    *x = (l1.c * l2.b - l2.c * l1.b) / det;
    *y = (l1.a * l2.c - l2.a * l1.c) / det;
    return 1;
}

/* Scans a short perpendicular window (in chroma-plane coordinates)
 * around an expected position and returns the mask-transition point
 * nearest the center of that window, or 0 if no transition is found
 * within it. `mask` is a full CHROMA_W x CHROMA_H classification
 * array (is_border, interior, or playing_surface); the transition
 * sought is anywhere mask flips between consecutive samples along the
 * scan -- direction-agnostic, since either side of the window could
 * be the "inside" depending on which corner's edge this is. */
static int find_mask_transition(const uint8_t *mask, int horizontal_scan,
                                 int fixed_c, double expected_c2,
                                 double *hit_c2) {
    int ec = (int)lround(expected_c2);
    int lo = ec - REFINE_SEARCH_RADIUS_CHROMA, hi = ec + REFINE_SEARCH_RADIUS_CHROMA;
    int best_pos = -1, best_dist = INT32_MAX;
    for (int c2 = lo + 1; c2 <= hi; c2++) {
        int c2_prev = c2 - 1;
        int x0, y0, x1, y1;
        if (horizontal_scan) { x0 = c2_prev; y0 = fixed_c; x1 = c2; y1 = fixed_c; }
        else                 { x0 = fixed_c; y0 = c2_prev; x1 = fixed_c; y1 = c2; }
        if (x0 < 0 || x0 >= CHROMA_W || x1 < 0 || x1 >= CHROMA_W) continue;
        if (y0 < 0 || y0 >= CHROMA_H || y1 < 0 || y1 >= CHROMA_H) continue;
        uint8_t v0 = mask[y0 * CHROMA_W + x0];
        uint8_t v1 = mask[y1 * CHROMA_W + x1];
        if (v0 == v1) continue;
        int dist = abs(c2 - ec);
        if (dist < best_dist) { best_dist = dist; best_pos = c2; }
    }
    if (best_pos < 0) return 0;
    *hit_c2 = best_pos - 0.5; /* midpoint between the last-0 and first-1 sample */
    return 1;
}

/* The refinement pass itself: 4 edges between 4 rough corners, each
 * edge sampled at REFINE_N_SAMPLES points (interpolated, full-res),
 * each sample's precise transition found via find_mask_transition
 * (in chroma space), the resulting points per edge robustly line-
 * fit, and each final corner taken from the intersection of its two
 * adjacent fitted edges -- same corner-from-adjacent-lines pattern as
 * border_detect.c (TL = intersect(top, left), etc.), generalized to
 * 4 edges via wraparound. All-or-nothing per boundary: if any edge
 * can't get 2 valid samples, the rough corners are kept unchanged
 * rather than mixing fitted and unfitted edges, which would make the
 * intersections inconsistent. */
static void refine_quad(const uint8_t *mask, const Pt rough[4],
                         const char *name, Pt refined_out[4]) {
    static const char *edge_names[4] = {"edge0", "edge1", "edge2", "edge3"};
    Line lines[4];
    int all_ok = 1;

    for (int e = 0; e < 4 && all_ok; e++) {
        Pt c0 = rough[e], c1 = rough[(e + 1) % 4];
        double dx = c1.x - c0.x, dy = c1.y - c0.y;
        /* Scan perpendicular to the edge -- same logic as border_detect.c's
         * scan_edge: an edge running mostly left-right needs a vertical
         * scan to cross it, and vice versa. */
        int edge_mostly_horizontal = fabs(dx) >= fabs(dy);

        double xs[REFINE_MAX_POINTS], ys[REFINE_MAX_POINTS];
        int n = 0;
        for (int i = 0; i < REFINE_N_SAMPLES; i++) {
            double t = REFINE_EDGE_INSET +
                       (1.0 - 2.0 * REFINE_EDGE_INSET) * i / (REFINE_N_SAMPLES - 1);
            double ex = c0.x + dx * t, ey = c0.y + dy * t; /* full-res */
            double ex_c = ex / 2.0, ey_c = ey / 2.0;       /* chroma */

            double hit_c2;
            int found;
            int fixed_c;
            if (edge_mostly_horizontal) {
                /* edge runs left-right -> perpendicular scan is vertical at this x */
                fixed_c = (int)lround(ex_c);
                found = find_mask_transition(mask, 0, fixed_c, ey_c, &hit_c2);
                if (found) { xs[n] = fixed_c * 2.0; ys[n] = hit_c2 * 2.0; n++; }
            } else {
                /* edge runs top-bottom -> perpendicular scan is horizontal at this y */
                fixed_c = (int)lround(ey_c);
                found = find_mask_transition(mask, 1, fixed_c, ex_c, &hit_c2);
                if (found) { xs[n] = hit_c2 * 2.0; ys[n] = fixed_c * 2.0; n++; }
            }
        }

        if (n < 2) {
            fprintf(stderr,
                "warning: %s %s refinement found only %d/%d usable sample "
                "points -- keeping the unrefined (flood-fill) corners for "
                "this boundary instead.\n", name, edge_names[e], n, REFINE_N_SAMPLES);
            all_ok = 0;
            break;
        }
        fit_line_robust(xs, ys, n, edge_names[e], &lines[e]);
    }

    if (!all_ok) {
        for (int i = 0; i < 4; i++) refined_out[i] = rough[i];
        return;
    }

    for (int i = 0; i < 4; i++) {
        double x, y;
        Line ending_here = lines[(i + 3) % 4], starting_here = lines[i];
        if (line_intersect(ending_here, starting_here, &x, &y)) {
            refined_out[i] = (Pt){x, y};
        } else {
            fprintf(stderr,
                "warning: %s corner %d's two edges came out parallel -- "
                "keeping the unrefined corner.\n", name, i);
            refined_out[i] = rough[i];
        }
    }
}

/* Zeroes out everything in a binary mask except its single largest
 * 4-connected component. Guards the border mask against an isolated
 * stray pixel (or small speckle) elsewhere in the frame happening to
 * match the border color -- the min/max pixel-fraction checks above
 * only catch gross over/under-matching, not a lone speck skewing the
 * extremal-point corner search, which has no outlier rejection of its
 * own (unlike the refinement pass that follows it). Confirmed by
 * testing: a single matching pixel far enough from the true corner
 * survives the fraction checks, wins the extremal search outright,
 * and -- if far enough that no refinement sample window reaches the
 * true edge either -- comes out the other end as the final corner. */
static void keep_largest_component(uint8_t *mask, int w, int h) {
    int n = w * h;
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
            static const int ddx[4] = {-1, 1, 0, 0};
            static const int ddy[4] = {0, 0, -1, 1};
            for (int k = 0; k < 4; k++) {
                int nx = px + ddx[k], ny = py + ddy[k];
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
        if (sizes[l] > best_size) { best_size = sizes[l]; best_label = l; }
    }
    for (int i = 0; i < n; i++) {
        mask[i] = (label[i] == best_label) ? 1 : 0;
    }
    free(label);
    free(sizes);
    free(stack);
}

/* ---- Step 1: sample the border's own color from a small patch ---- */

static int sample_border_color(const Frame *f, int sx, int sy,
                                double *u_med, double *v_med,
                                double *u_mad, double *v_mad) {
    int cx = sx / 2, cy = sy / 2;
    int patch = 8; /* +/- chroma px around the click -- ~16 full-res px */
    int x0 = cx - patch, x1 = cx + patch;
    int y0 = cy - patch, y1 = cy + patch;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > CHROMA_W) x1 = CHROMA_W;
    if (y1 > CHROMA_H) y1 = CHROMA_H;
    int n = (x1 - x0) * (y1 - y0);
    if (n <= 0) {
        fprintf(stderr, "error: border sample point is outside the frame\n");
        return -1;
    }
    int *uvals = malloc(sizeof(int) * n);
    int *vvals = malloc(sizeof(int) * n);
    int idx = 0;
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            uvals[idx] = f->u[y * CHROMA_W + x];
            vvals[idx] = f->v[y * CHROMA_W + x];
            idx++;
        }
    }
    median_and_mad(uvals, idx, u_med, u_mad);
    median_and_mad(vvals, idx, v_med, v_mad);
    free(uvals);
    free(vvals);
    return 0;
}

/* ---- Step 2: use the border ring as a wall, flood-fill what it
 * encloses, and take THAT region's 4 extremal points ---- */

/* Finding the border-colored pixels' own extremal points would give
 * the ring's OUTER edge (border-to-carpet) -- the opposite of what's
 * needed. calibrate_teams' ROI must sit safely INSIDE the tape, never
 * touching it (see the tape-scanning contamination this project hit
 * earlier), so what's actually wanted is the ring's INNER edge
 * (border-to-interior). Getting that without a separate interior-
 * color reference: treat border-colored pixels as a wall and flood-
 * fill everything the wall encloses; the enclosed region's own
 * extremal points are the true interior corners, regardless of what
 * color the interior itself turns out to be. */
static int find_border_corners(const Frame *f, double u_med, double v_med,
                                double u_mad, double v_mad,
                                Pt inner_corners[4], Pt outer_corners[4]) {
    double combined_mad = (u_mad + v_mad) / 2.0;
    double threshold = combined_mad * BORDER_DEVIATION_MAD_MULTIPLIER;
    if (threshold < BORDER_DEVIATION_FLOOR) threshold = BORDER_DEVIATION_FLOOR;

    int n = CHROMA_W * CHROMA_H;
    uint8_t *is_border = calloc(n, 1);
    long matched = 0;
    long cx_sum = 0, cy_sum = 0;
    for (int y = 0; y < CHROMA_H; y++) {
        for (int x = 0; x < CHROMA_W; x++) {
            double du = (double)f->u[y * CHROMA_W + x] - u_med;
            double dv = (double)f->v[y * CHROMA_W + x] - v_med;
            if (sqrt(du * du + dv * dv) > threshold) continue;
            is_border[y * CHROMA_W + x] = 1;
            matched++;
            cx_sum += x;
            cy_sum += y;
        }
    }
    if (matched < (long)(n * BORDER_MIN_PIXEL_FRACTION)) {
        fprintf(stderr,
            "error: only %ld px matched the border color (need at least "
            "%.0f) -- is the border sample point actually on the border, "
            "and is the border color distinct enough from everything else "
            "in the frame?\n", matched, n * BORDER_MIN_PIXEL_FRACTION);
        free(is_border);
        return -1;
    }
    if (matched > (long)(n * BORDER_MAX_PIXEL_FRACTION)) {
        fprintf(stderr,
            "error: %.0f%% of the frame matched the border color -- a "
            "border is normally a thin frame, not a large fraction of the "
            "image. This usually means the sample point landed on the "
            "board's white surface or the carpet/background instead of "
            "the border itself -- click again, right on the tape.\n",
            100.0 * matched / n);
        free(is_border);
        return -1;
    }

    /* Drop everything except the single largest connected blob of
     * border-colored pixels -- the actual ring, discarding any
     * isolated speck elsewhere that happened to match by chance.
     * Recompute the centroid from the filtered mask too, so a stray
     * pixel can't even nudge the seed position. */
    keep_largest_component(is_border, CHROMA_W, CHROMA_H);
    cx_sum = 0; cy_sum = 0; matched = 0;
    for (int i = 0; i < n; i++) {
        if (!is_border[i]) continue;
        cx_sum += i % CHROMA_W;
        cy_sum += i / CHROMA_W;
        matched++;
    }
    if (matched == 0) {
        fprintf(stderr, "error: border mask was empty after filtering -- "
                         "this shouldn't happen; please report it.\n");
        free(is_border);
        return -1;
    }

    /* Flood-fill seed: the border mask's own centroid, nudged to the
     * nearest non-border pixel if the centroid itself landed on the
     * ring (spiral outward in a small bounded search -- a proper
     * rectangular ring's centroid should already be inside it, this
     * is just insurance against an odd-shaped border). */
    int seed_x = (int)(cx_sum / matched), seed_y = (int)(cy_sum / matched);
    if (is_border[seed_y * CHROMA_W + seed_x]) {
        int found_seed = 0;
        for (int r = 1; r < 200 && !found_seed; r++) {
            for (int dy = -r; dy <= r && !found_seed; dy++) {
                for (int dx = -r; dx <= r && !found_seed; dx++) {
                    int nx = seed_x + dx, ny = seed_y + dy;
                    if (nx < 0 || nx >= CHROMA_W || ny < 0 || ny >= CHROMA_H) continue;
                    if (!is_border[ny * CHROMA_W + nx]) {
                        seed_x = nx; seed_y = ny; found_seed = 1;
                    }
                }
            }
        }
        if (!found_seed) {
            fprintf(stderr,
                "error: couldn't find a non-border pixel near the border "
                "mask's centroid to seed the interior flood-fill\n");
            free(is_border);
            return -1;
        }
    }

    /* Flood-fill everything reachable from the seed without crossing
     * a border-colored pixel -- the enclosed interior, whatever color
     * it turns out to be. */
    uint8_t *interior = calloc(n, 1);
    int *stack = malloc(sizeof(int) * n);
    int sp = 0;
    int seed_idx = seed_y * CHROMA_W + seed_x;
    interior[seed_idx] = 1;
    stack[sp++] = seed_idx;
    long interior_count = 1;
    while (sp > 0) {
        int p = stack[--sp];
        int py = p / CHROMA_W, px = p % CHROMA_W;
        static const int ddx[4] = {-1, 1, 0, 0};
        static const int ddy[4] = {0, 0, -1, 1};
        for (int k = 0; k < 4; k++) {
            int nx = px + ddx[k], ny = py + ddy[k];
            if (nx < 0 || nx >= CHROMA_W || ny < 0 || ny >= CHROMA_H) continue;
            int ni = ny * CHROMA_W + nx;
            if (is_border[ni] || interior[ni]) continue;
            interior[ni] = 1;
            interior_count++;
            stack[sp++] = ni;
        }
    }
    free(stack);

    /* Sanity check: if the border ring has a gap (occlusion, a bad
     * lighting patch breaking the color match), the flood-fill leaks
     * out through it and swallows most of the frame instead of just
     * the board interior. A real interior is a bounded fraction of
     * the whole image, not most of it. */
    if (interior_count > (long)(n * 0.75)) {
        fprintf(stderr,
            "error: the flood-filled interior is implausibly large (%.0f%% "
            "of the frame) -- the border likely has a gap somewhere "
            "(occlusion, a lighting patch breaking the color match) that "
            "let the fill leak out past it. Check the border is fully "
            "visible and continuous, and retry.\n",
            100.0 * interior_count / n);
        free(interior);
        free(is_border);
        return -1;
    }

    /* Rough inner corners: extremal points of the interior alone (tape
     * excluded) -- the interior-to-tape boundary. These seed the
     * precision refinement below, same role border_detect.c's hand-
     * typed corners.txt used to serve, just found automatically. */
    double best_sum_lo = 1e18, best_sum_hi = -1e18;
    double best_diff_lo = 1e18, best_diff_hi = -1e18;
    Pt p_sum_lo = {0}, p_sum_hi = {0}, p_diff_lo = {0}, p_diff_hi = {0};
    for (int y = 0; y < CHROMA_H; y++) {
        for (int x = 0; x < CHROMA_W; x++) {
            if (!interior[y * CHROMA_W + x]) continue;
            double fx = x * 2.0, fy = y * 2.0; /* full-res */
            double s = fx + fy, d = fx - fy;
            if (s < best_sum_lo)  { best_sum_lo = s;  p_sum_lo  = (Pt){fx, fy}; }
            if (s > best_sum_hi)  { best_sum_hi = s;  p_sum_hi  = (Pt){fx, fy}; }
            if (d < best_diff_lo) { best_diff_lo = d; p_diff_lo = (Pt){fx, fy}; }
            if (d > best_diff_hi) { best_diff_hi = d; p_diff_hi = (Pt){fx, fy}; }
        }
    }
    Pt rough_inner[4] = {p_sum_lo, p_diff_hi, p_sum_hi, p_diff_lo};

    /* Rough outer corners: extremal points of interior-OR-border --
     * the tape-to-carpet boundary. The board's playing surface (per
     * the rules, interior and tape are scored identically) is
     * everything inside THIS boundary, not just the interior. */
    uint8_t *playing_surface = malloc(n);
    for (int i = 0; i < n; i++) playing_surface[i] = interior[i] || is_border[i];

    best_sum_lo = 1e18; best_sum_hi = -1e18;
    best_diff_lo = 1e18; best_diff_hi = -1e18;
    for (int y = 0; y < CHROMA_H; y++) {
        for (int x = 0; x < CHROMA_W; x++) {
            if (!playing_surface[y * CHROMA_W + x]) continue;
            double fx = x * 2.0, fy = y * 2.0;
            double s = fx + fy, d = fx - fy;
            if (s < best_sum_lo)  { best_sum_lo = s;  p_sum_lo  = (Pt){fx, fy}; }
            if (s > best_sum_hi)  { best_sum_hi = s;  p_sum_hi  = (Pt){fx, fy}; }
            if (d < best_diff_lo) { best_diff_lo = d; p_diff_lo = (Pt){fx, fy}; }
            if (d > best_diff_hi) { best_diff_hi = d; p_diff_hi = (Pt){fx, fy}; }
        }
    }
    Pt rough_outer[4] = {p_sum_lo, p_diff_hi, p_sum_hi, p_diff_lo};

    /* Precision pass: refine both boundaries via multi-sample line
     * fitting against the classification masks, same architecture as
     * border_detect.c, instead of trusting a single extremal pixel. */
    refine_quad(interior, rough_inner, "inner", inner_corners);
    refine_quad(playing_surface, rough_outer, "outer", outer_corners);

    free(playing_surface);
    free(interior);
    free(is_border);
    return 0;
}

/* ---- Step 3: largest dark blob strictly inside the 4 corners ---- */

static int find_hole(const Frame *f, const Pt corners[4],
                      int *hole_cx, int *hole_cy, int *hole_radius) {
    /* Bounding box of the quad, full-res, just to bound the scan --
     * mirrors detect_bags' interior-only scanning discipline: never
     * scan pixels outside the calibrated board region. */
    double x0 = corners[0].x, x1 = corners[0].x;
    double y0 = corners[0].y, y1 = corners[0].y;
    for (int i = 1; i < 4; i++) {
        if (corners[i].x < x0) x0 = corners[i].x;
        if (corners[i].x > x1) x1 = corners[i].x;
        if (corners[i].y < y0) y0 = corners[i].y;
        if (corners[i].y > y1) y1 = corners[i].y;
    }
    int ix0 = (int)x0 < 0 ? 0 : (int)x0;
    int iy0 = (int)y0 < 0 ? 0 : (int)y0;
    int ix1 = (int)x1 > WIDTH  ? WIDTH  : (int)x1;
    int iy1 = (int)y1 > HEIGHT ? HEIGHT : (int)y1;
    int w = ix1 - ix0, h = iy1 - iy0;
    if (w <= 0 || h <= 0) {
        fprintf(stderr, "error: board quad has no interior area\n");
        return -1;
    }
    long interior_area = (long)w * h; /* upper bound; quad is smaller */

    /* Board-surface Y reference: median Y over the quad interior.
     * Robust to the hole itself (a small fraction of total area)
     * without needing to explicitly exclude it first. */
    int *yvals = malloc(sizeof(int) * w * h);
    long yn = 0;
    for (int y = iy0; y < iy1; y++) {
        for (int x = ix0; x < ix1; x++) {
            if (!point_in_quad((double)x, (double)y, corners)) continue;
            yvals[yn++] = f->y[y * WIDTH + x];
        }
    }
    if (yn == 0) {
        fprintf(stderr, "error: no pixels inside the board quad\n");
        free(yvals);
        return -1;
    }
    double y_med, y_mad;
    median_and_mad(yvals, (int)yn, &y_med, &y_mad);
    free(yvals);

    double threshold = y_mad * HOLE_DEVIATION_MAD_MULTIPLIER;
    if (threshold < HOLE_DEVIATION_FLOOR) threshold = HOLE_DEVIATION_FLOOR;
    double dark_cutoff = y_med - threshold; /* one-sided: darker only */

    /* Connected-component label the dark mask, same 4-connected flood
     * fill as detect_bags' blob finder, just on Y instead of chroma
     * and at full resolution (Y already is full-res). */
    uint8_t *mask = calloc((size_t)w * h, 1);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int fx = ix0 + x, fy = iy0 + y;
            if (!point_in_quad((double)fx, (double)fy, corners)) continue;
            mask[y * w + x] = (f->y[fy * WIDTH + fx] < dark_cutoff) ? 1 : 0;
        }
    }

    int n = w * h;
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

    int best_label = 0, best_size = 0;
    for (int l = 1; l <= next_label; l++) {
        if (sizes[l] > best_size) { best_size = sizes[l]; best_label = l; }
    }

    int ok = 1;
    if (best_label == 0 ||
        best_size < (int)(interior_area * HOLE_MIN_AREA_FRACTION) ||
        best_size > (int)(interior_area * HOLE_MAX_AREA_FRACTION)) {
        fprintf(stderr,
            "warning: no plausible hole-sized dark region found (largest "
            "dark blob: %d px, need roughly %.0f-%.0f for a hole) -- "
            "omitting hole from output. Run calibrate_teams hole by hand "
            "for this session.\n",
            best_size, interior_area * HOLE_MIN_AREA_FRACTION,
            interior_area * HOLE_MAX_AREA_FRACTION);
        ok = 0;
    } else {
        long x_sum = 0, y_sum = 0;
        for (int i = 0; i < n; i++) {
            if (label[i] != best_label) continue;
            int py = i / w, px = i % w;
            x_sum += ix0 + px;
            y_sum += iy0 + py;
        }
        *hole_cx = (int)(x_sum / best_size);
        *hole_cy = (int)(y_sum / best_size);
        *hole_radius = (int)sqrt((double)best_size / M_PI);
    }

    free(mask);
    free(label);
    free(sizes);
    free(stack);
    return ok ? 0 : 1; /* 1 = "ran fine, just didn't find a hole" */
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr,
            "usage: detect_board <frame.yuv420> <border_sample_x> "
            "<border_sample_y>\n");
        return 1;
    }
    const char *frame_path = argv[1];
    int sx = atoi(argv[2]), sy = atoi(argv[3]);

    Frame f;
    if (load_frame(frame_path, &f) != 0) return 1;

    double u_med, v_med, u_mad, v_mad;
    if (sample_border_color(&f, sx, sy, &u_med, &v_med, &u_mad, &v_mad) != 0) {
        free_frame(&f);
        return 1;
    }
    fprintf(stderr, "info: border color U median=%.1f (MAD %.1f)  "
                     "V median=%.1f (MAD %.1f)\n", u_med, u_mad, v_med, v_mad);

    Pt inner_corners[4], outer_corners[4];
    if (find_border_corners(&f, u_med, v_med, u_mad, v_mad,
                             inner_corners, outer_corners) != 0) {
        free_frame(&f);
        return 1;
    }
    for (int i = 0; i < 4; i++) {
        printf("inner %d %d %d\n", i, (int)inner_corners[i].x, (int)inner_corners[i].y);
    }
    for (int i = 0; i < 4; i++) {
        printf("outer %d %d %d\n", i, (int)outer_corners[i].x, (int)outer_corners[i].y);
    }

    int hole_cx, hole_cy, hole_radius;
    int hole_rc = find_hole(&f, inner_corners, &hole_cx, &hole_cy, &hole_radius);
    if (hole_rc == 0) {
        printf("hole %d %d %d\n", hole_cx, hole_cy, hole_radius);
    }

    free_frame(&f);
    return 0;
}