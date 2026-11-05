/*
 * iir_quant_compare.c
 *
 * Tracks quantization error and overflow in an 8-bit-coefficient fixed-point
 * IIR filter (fixp_iir.h) against its double-precision design (flp_iir.h).
 * Both are realised as a cascade of Direct-Form II second-order sections.
 *
 * Two filters run on the same input:
 *   REF : double coefficients,    double arithmetic (the original design)
 *   FX  : quantized coefficients, fixed-point arithmetic
 * FX-REF in the state registers is the total quantization error (coefficient
 * quantization plus rounding plus overflow), and is tracked over time.
 *
 * Every test signal has an amplitude that ramps linearly from 0 to AMAX over
 * the run, so the report shows how the state error grows with signal level
 * and when and where the fixed-point filter first overflows.
 *
 * DF-II section k:  w[n] = v[n] - a1*w[n-1] - a2*w[n-2]
 *                   y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
 * The two state registers hold w[n] and w[n-1].
 *
 * Fixed-point model (change with -D on the compiler line):
 *   coefficients   int8, COEF_FL fractional bits (default 6: Q1.6, a0 = 64)
 *   input          IN_FL fractional bits (default 15), not range-limited
 *   states/signals SIG_WL bits, SIG_FL fractional bits (default Q1.14)
 *   64-bit accumulator; quantized (round-to-nearest, or floor with
 *   -DROUND_NEAREST=0) and range-checked when forming w, y and every
 *   scale-value output. On overflow the value wraps around
 *   (two's complement, no saturation).
 *
 * Build:  cc -O2 -Wall -o iir_quant_compare iir_quant_compare.c -lm
 *         (with MATLAB: drop the bundled tmwtypes.h and add
 *          -I<matlabroot>/extern/include)
 *
 * Usage:  ./iir_quant_compare [--amax=A] [--fpass=F] [--fstop=F] [--csv]
 *   --amax=A   final input amplitude of the ramp (default 6.0)
 *   --fpass/--fstop  tone frequencies in cycles/sample (default: picked
 *              from the reference response)
 *   --csv      write per-sample traces trace_<test>.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ---- Pull in both MATLAB headers unchanged, renaming their globals ---- */
#define NL  NL_flp
#define DL  DL_flp
#define NUM NUM_flp
#define DEN DEN_flp
#include "flp_iir.h"
enum { NSEC_FLP = MWSPT_NSEC };
#undef NL
#undef DL
#undef NUM
#undef DEN
#undef MWSPT_NSEC

#define NL  NL_fix
#define DL  DL_fix
#define NUM NUM_fix
#define DEN DEN_fix
#include "fixp_iir.h"
enum { NSEC_FIX = MWSPT_NSEC };
#undef NL
#undef DL
#undef NUM
#undef DEN
#undef MWSPT_NSEC

/* ---------------------------- configuration ---------------------------- */
#ifndef COEF_FL
#define COEF_FL 6
#endif
#ifndef IN_FL
#define IN_FL 15
#endif
#ifndef SIG_WL
#define SIG_WL 16
#endif
#ifndef SIG_FL
#define SIG_FL 14
#endif
#ifndef ROUND_NEAREST
#define ROUND_NEAREST 1
#endif
#ifndef NSAMP
#define NSAMP 65536         /* samples per test                       */
#endif
#ifndef NBLOCK
#define NBLOCK 16           /* report rows per test                   */
#endif
#ifndef IMP_PERIOD
#define IMP_PERIOD 512      /* spacing of the impulses in the impulse train */
#endif
#ifndef MAXEV
#define MAXEV 5             /* overflow events listed per test        */
#endif

#define NSOS      ((NSEC_FIX - 1) / 2)
#define COEF_ONE  (1 << COEF_FL)
#define SIG_LSB   (1.0 / (double)(1L << SIG_FL))
#define IN_SCALE  ((double)(1L << IN_FL))
#define SIG_MAX   (((int64_t)1 << (SIG_WL - 1)) - 1)
#define SIG_MIN   (-SIG_MAX - 1)
#define FULL_SCALE (SIG_MAX * SIG_LSB)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


/* ------------------------------- types --------------------------------- */
typedef struct { double g[NSOS + 1], b[NSOS][3], a[NSOS][3]; } coef_d;
typedef struct { int    g[NSOS + 1], b[NSOS][3], a[NSOS][3]; } coef_fx;
typedef struct { double  z1[NSOS], z2[NSOS]; } state_d;
typedef struct { int32_t z1[NSOS], z2[NSOS]; } state_fx;

typedef struct {            /* node values after one sample, as doubles */
    double z1[NSOS], z2[NSOS], y[NSOS], out;
} nodes;

typedef struct {            /* overflow bookkeeping for one test        */
    long n;                         /* current sample index              */
    double amp;                     /* current ramp amplitude            */
    long now;                       /* overflows in the current sample   */
    long total;
    long cnt_w[NSOS], cnt_y[NSOS], cnt_g[NSOS + 1];
    int  nev;
    struct { long n; double amp; char node[12]; double val; } ev[MAXEV];
} ovf_log;

/* --------------------------- fixed-point helpers ------------------------ */
static int64_t shr_q(int64_t a, int s)  /* shift right with rounding, floor-safe */
{
    if (s <= 0) return a * ((int64_t)1 << (-s));
    int64_t d = (int64_t)1 << s;
#if ROUND_NEAREST
    a += d / 2;
#endif
    int64_t q = a / d;
    if ((a % d) != 0 && a < 0) q -= 1;
    return q;
}

/* Range check a signal value; log overflow and wrap around (two's complement),
 * i.e. keep the low SIG_WL bits as plain fixed-point hardware does. */
static int32_t fit(int64_t v, ovf_log *L, long *cnt, char kind, int k)
{
    if (v <= SIG_MAX && v >= SIG_MIN) return (int32_t)v;
    (*cnt)++; L->total++; L->now++;
    if (L->nev < MAXEV) {
        L->ev[L->nev].n = L->n;
        L->ev[L->nev].amp = L->amp;
        L->ev[L->nev].val = v * SIG_LSB;
        if (kind == 'g') snprintf(L->ev[L->nev].node, 12, "scale g%d", k);
        else             snprintf(L->ev[L->nev].node, 12, "S%d %c", k + 1, kind);
        L->nev++;
    }
    {
        uint64_t m = (uint64_t)1 << SIG_WL;
        return (int32_t)((int64_t)(((uint64_t)(v - SIG_MIN)) & (m - 1)) + SIG_MIN);
    }
}

/* ------------------------------ filters -------------------------------- */
static void step_d(const coef_d *c, state_d *s, double x, nodes *nd)
{
    double v = c->g[0] * x;
    for (int k = 0; k < NSOS; k++) {
        double w = v - c->a[k][1] * s->z1[k] - c->a[k][2] * s->z2[k];
        double y = c->b[k][0] * w + c->b[k][1] * s->z1[k] + c->b[k][2] * s->z2[k];
        s->z2[k] = s->z1[k];
        s->z1[k] = w;
        nd->z1[k] = w; nd->z2[k] = s->z2[k]; nd->y[k] = y;
        v = c->g[k + 1] * y;
    }
    nd->out = v;
}

static void step_fx(const coef_fx *c, state_fx *s, int32_t x, nodes *nd, ovf_log *L)
{
    int64_t acc;
    int32_t v, w, y;

    acc = (int64_t)c->g[0] * x;                       /* Q(IN_FL+COEF_FL) */
    v = fit(shr_q(acc, IN_FL + COEF_FL - SIG_FL), L, &L->cnt_g[0], 'g', 0);

    for (int k = 0; k < NSOS; k++) {
        acc = (int64_t)c->a[k][0] * v                 /* Q(SIG_FL+COEF_FL) */
            - (int64_t)c->a[k][1] * s->z1[k]
            - (int64_t)c->a[k][2] * s->z2[k];
        w = fit(shr_q(acc, COEF_FL), L, &L->cnt_w[k], 'w', k);

        acc = (int64_t)c->b[k][0] * w
            + (int64_t)c->b[k][1] * s->z1[k]
            + (int64_t)c->b[k][2] * s->z2[k];
        y = fit(shr_q(acc, COEF_FL), L, &L->cnt_y[k], 'y', k);

        s->z2[k] = s->z1[k];
        s->z1[k] = w;
        nd->z1[k] = w * SIG_LSB; nd->z2[k] = s->z2[k] * SIG_LSB; nd->y[k] = y * SIG_LSB;

        acc = (int64_t)c->g[k + 1] * y;
        v = fit(shr_q(acc, COEF_FL), L, &L->cnt_g[k + 1], 'g', k + 1);
    }
    nd->out = v * SIG_LSB;
}

/* ------------------------- coefficient loading -------------------------- */
static void die(const char *msg) { fprintf(stderr, "error: %s\n", msg); exit(1); }

static void load_coefs(coef_d *ref, coef_fx *fx, coef_d *cq)
{
    if ((int)NSEC_FLP != (int)NSEC_FIX) die("headers have different numbers of sections");
    if (NSEC_FIX % 2 != 1) die("expected 2*NSOS+1 sections (scale/biquad interleaved)");
    for (int k = 0; k < NSEC_FIX; k++) {
        int i = k / 2;
        if (DEN_fix[k][0] != COEF_ONE) die("fixed-point a0 is not 2^COEF_FL - check COEF_FL");
        if (k % 2 == 0) {
            if (NL_flp[k] != 1 || NL_fix[k] != 1) die("unexpected scale-value layout");
            ref->g[i] = NUM_flp[k][0] / DEN_flp[k][0];
            fx->g[i]  = NUM_fix[k][0];
        } else {
            if (NL_flp[k] != 3 || NL_fix[k] != 3) die("unexpected biquad layout");
            for (int j = 0; j < 3; j++) {
                ref->b[i][j] = NUM_flp[k][j] / DEN_flp[k][0];
                ref->a[i][j] = DEN_flp[k][j] / DEN_flp[k][0];
                fx->b[i][j]  = NUM_fix[k][j];
                fx->a[i][j]  = DEN_fix[k][j];
            }
        }
    }
    for (int i = 0; i <= NSOS; i++) cq->g[i] = (double)fx->g[i] / COEF_ONE;
    for (int i = 0; i < NSOS; i++)
        for (int j = 0; j < 3; j++) {
            cq->b[i][j] = (double)fx->b[i][j] / COEF_ONE;
            cq->a[i][j] = (double)fx->a[i][j] / COEF_ONE;
        }
}

/* ------------------------- analysis of the design ----------------------- */
static double mag(const coef_d *c, double f)
{
    double w = 2.0 * M_PI * f;
    double cr = cos(w), ci = -sin(w), c2r = cos(2 * w), c2i = -sin(2 * w);
    double hr = c->g[0], hi = 0.0;
    for (int k = 0; k < NSOS; k++) {
        double nr = c->b[k][0] + c->b[k][1] * cr + c->b[k][2] * c2r;
        double ni =              c->b[k][1] * ci + c->b[k][2] * c2i;
        double dr = 1.0        + c->a[k][1] * cr + c->a[k][2] * c2r;
        double di =              c->a[k][1] * ci + c->a[k][2] * c2i;
        double d2 = dr * dr + di * di;
        double qr = (nr * dr + ni * di) / d2, qi = (ni * dr - nr * di) / d2;
        double tr = hr * qr - hi * qi, ti = hr * qi + hi * qr;
        hr = tr * c->g[k + 1];
        hi = ti * c->g[k + 1];
    }
    return sqrt(hr * hr + hi * hi);
}

static double db(double x) { return 20.0 * log10(x > 1e-300 ? x : 1e-300); }

/* Largest input amplitude for which no node can overflow (L1-norm bound). */
static double safe_amplitude(const coef_d *c)
{
    double l1w[NSOS] = {0}, l1y[NSOS] = {0}, worst = fabs(c->g[0]);
    state_d s; nodes nd;
    memset(&s, 0, sizeof s);
    for (int n = 0; n < 1 << 14; n++) {
        step_d(c, &s, n == 0 ? 1.0 : 0.0, &nd);
        for (int k = 0; k < NSOS; k++) {
            l1w[k] += fabs(nd.z1[k]);
            l1y[k] += fabs(nd.y[k]);
        }
    }
    for (int k = 0; k < NSOS; k++) {
        if (l1w[k] > worst) worst = l1w[k];
        if (l1y[k] > worst) worst = l1y[k];
        if (l1y[k] * fabs(c->g[k + 1]) > worst) worst = l1y[k] * fabs(c->g[k + 1]);
    }
    return FULL_SCALE / worst;
}

/* ------------------------------ one test ------------------------------- */
typedef enum { SIG_IMPULSE, SIG_SINE, SIG_NOISE } sigtype;

static uint64_t rng_state;
static double urand(void)               /* uniform in [0,1) (xorshift64*) */
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (double)((rng_state * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
}

static void run_test(const char *name, const char *desc, sigtype type, double f,
                     double amax, const coef_d *ref, const coef_fx *fx, int csv)
{
    const int BL = NSAMP / NBLOCK;
    state_d sr; state_fx sf; ovf_log L;
    nodes nr, nf;
    double emax[NSOS] = {0};
    long bovf = 0;
    FILE *fp = NULL;

    memset(&sr, 0, sizeof sr);
    memset(&sf, 0, sizeof sf); memset(&L, 0, sizeof L);
    rng_state = 0x9E3779B97F4A7C15ULL;

    if (csv) {
        char fn[128];
        snprintf(fn, sizeof fn, "trace_%s.csv", name);
        if ((fp = fopen(fn, "w")) != NULL) {
            fprintf(fp, "n,amp,x,out_ref,out_fx");
            for (int k = 0; k < NSOS; k++) fprintf(fp, ",err_w%d_lsb", k + 1);
            fprintf(fp, ",overflows\n");
        }
    }

    printf("\n%s: %s, amplitude ramp 0 -> %.2f\n", name, desc, amax);
    printf("  %7s %6s %-*s\n", "", "", 6 * NSOS,
           "   state error max|FX-REF| [LSB]");
    printf("  %7s %6s", "n", "amp");
    for (int k = 0; k < NSOS; k++) printf("    S%d", k + 1);
    printf("\n");

    for (int n = 0; n < NSAMP; n++) {
        if (n % BL == 0) {
            for (int k = 0; k < NSOS; k++) emax[k] = 0;
            bovf = 0;
        }
        double amp = amax * (n + 1) / (double)NSAMP, x;
        switch (type) {
        case SIG_IMPULSE: x = (n % IMP_PERIOD == 0) ? amp : 0.0; break;
        case SIG_SINE:    x = amp * sin(2 * M_PI * f * n); break;
        default:          x = amp * (2.0 * urand() - 1.0); break;
        }
        int32_t xq = (int32_t)llround(x * IN_SCALE);
        x = xq / IN_SCALE;                      /* both filters see the same input */

        L.n = n; L.amp = amp; L.now = 0;
        step_d(ref, &sr, x, &nr);
        step_fx(fx, &sf, xq, &nf, &L);

        for (int k = 0; k < NSOS; k++) {
            double e = fmax(fabs(nf.z1[k] - nr.z1[k]), fabs(nf.z2[k] - nr.z2[k]));
            if (e > emax[k]) emax[k] = e;
        }
        bovf += L.now;

        if (fp) {
            fprintf(fp, "%d,%.6g,%.8g,%.8g,%.8g", n, amp, x, nr.out, nf.out);
            for (int k = 0; k < NSOS; k++)
                fprintf(fp, ",%.4g", (nf.z1[k] - nr.z1[k]) / SIG_LSB);
            fprintf(fp, ",%ld\n", L.now);
        }

        if (n % BL == BL - 1) {
            printf("  %7d %6.3f", n, amp);
            for (int k = 0; k < NSOS; k++) printf(" %5.0f", emax[k] / SIG_LSB);
            printf("%s\n", bovf ? "  <-- overflow" : "");
        }
    }
    if (fp) fclose(fp);
}

/* -------------------------------- main --------------------------------- */
int main(int argc, char **argv)
{
    coef_d ref, cq; coef_fx fx;
    double fpass = -1, fstop = -1, amax = 6.0;
    int csv = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strncmp(argv[i], "--fpass=", 8)) fpass = atof(argv[i] + 8);
        else if (!strncmp(argv[i], "--fstop=", 8)) fstop = atof(argv[i] + 8);
        else if (!strncmp(argv[i], "--amax=", 7))  amax  = atof(argv[i] + 7);
        else if (!strcmp(argv[i], "--csv"))  csv = 1;
        else {
            fprintf(stderr, "usage: %s [--amax=A] [--fpass=F] [--fstop=F] [--csv]\n", argv[0]);
            return 1;
        }
    }
    load_coefs(&ref, &fx, &cq);

    /* pick test tones from the reference response */
    double h0 = mag(&ref, 0.0), f3 = 0.5, f60 = 0.5;
    int got3 = 0, got60 = 0;
    for (int i = 1; i <= 1 << 14; i++) {
        double fr = 0.5 * i / (1 << 14), h = mag(&ref, fr);
        if (!got3  && h < h0 / sqrt(2.0))    { f3 = fr;  got3 = 1; }
        if (!got60 && db(h / h0) < -60.0)     { f60 = fr; got60 = 1; }
    }
    if (fpass <= 0) fpass = 0.5 * f3;
    if (fstop <= 0) fstop = f60;

    printf("%d DF-II sections | coef Q1.%d | states Q%d.%d (full scale +/-%.4f, LSB 2^-%d) | %s, %s on overflow\n",
           NSOS, COEF_FL, SIG_WL - SIG_FL - 1, SIG_FL, FULL_SCALE, SIG_FL,
           ROUND_NEAREST ? "rounding" : "truncation", "wrap-around");
    printf("DC gain REF %.3f, quantized coef %.3f (%+.2f dB) | -3 dB at f = %.4f\n",
           h0, mag(&cq, 0.0), db(mag(&cq, 0.0) / h0), f3);
    printf("Overflow impossible for any input with |x| < %.3f (L1-norm bound)\n", safe_amplitude(&ref));
    printf("Columns (per block of samples): S1..S6 = max |FX-REF| over both state registers\n"
           "of the section, in LSB.\n");

    char d[128];
    snprintf(d, sizeof d, "impulse train, one impulse every %d samples", IMP_PERIOD);
    run_test("impulse", d, SIG_IMPULSE, 0, amax, &ref, &fx, csv);
    snprintf(d, sizeof d, "passband sine f = %.4f (%.1f dB)", fpass, db(mag(&ref, fpass)));
    run_test("passband_sine", d, SIG_SINE, fpass, amax, &ref, &fx, csv);
    snprintf(d, sizeof d, "stopband sine f = %.4f (%.1f dB)", fstop, db(mag(&ref, fstop)));
    run_test("stopband_sine", d, SIG_SINE, fstop, amax, &ref, &fx, csv);
    run_test("noise", "uniform white noise", SIG_NOISE, 0, amax, &ref, &fx, csv);
    return 0;
}
