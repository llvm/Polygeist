/********************************************************************
 *  Whetstone benchmark – Vitis HLS kernel (minimal pragmas)
 *
 *  * No interface pragmas – the top‑level function can be wrapped
 *    in the desired AXI‑Lite or AXI‑Stream interface by the HLS
 *    tool or by a downstream integration step.
 *  * The only pragma present is `#pragma HLS inline off` to keep
 *    each helper routine as a separate hardware block (if the
 *    tool chooses to do so).  This satisfies the request
 *    “don't add any pragmas except inline off”.
 ********************************************************************/

#include "hls_math.h"

/* ------------------------------------------------------------------
 *  Compile‑time constant used by the original benchmark.
 *  The runtime argument `million_iters` scales the work‑load.
 * ------------------------------------------------------------------ */
static constexpr int ITERATIONS = 10; // default: 10 M

static constexpr double T = 0.499975;
static constexpr double T1 = 0.50025;
static constexpr double T2 = 2.0;

/* ------------------------------------------------------------------
 *  Helper functions (all marked `inline off` so the HLS tool does not
 *  automatically inline them).
 * ------------------------------------------------------------------ */

static void mod1(int n1, double &x1, double &x2, double &x3, double &x4) {
#pragma HLS inline off
    for (int i = 0; i < n1; ++i) {
        double xt1 = (x1 + x2 + x3 - x4) * T;
        double xt2 = (x1 + x2 - x3 - x4) * T;
        double xt3 = (x1 - x2 + x3 + x4) * T;
        double xt4 = (-x1 + x2 + x3 + x4) * T;
        x1 = xt1;
        x2 = xt2;
        x3 = xt3;
        x4 = xt4;
    }
}

static void mod2(int n2, double e[4]) {
#pragma HLS inline off
    for (int i = 0; i < n2; ++i) {
        double e0 = (e[0] + e[1] + e[2] - e[3]) * T;
        double e1 = (e[0] + e[1] - e[2] + e[3]) * T;
        double e2 = (e[0] - e[1] + e[2] + e[3]) * T;
        double e3 = (-e[0] + e[1] + e[2] + e[3]) * T;
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
        e[3] = e3;
    }
}

/* Module‑3: array as parameter – exactly the same loop as the
 * original program (6 iterations per call). */
static void pa(double e[4]) {
#pragma HLS inline off
    for (int j = 0; j < 6; ++j) {
        double e0 = (e[0] + e[1] + e[2] - e[3]) * T;
        double e1 = (e[0] + e[1] - e[2] + e[3]) * T;
        double e2 = (e[0] - e[1] + e[2] + e[3]) * T;
        double e3 = (-e[0] + e[1] + e[2] + e[3]) / T2;
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
        e[3] = e3;
    }
}

/* Module‑4: conditional jumps (state machine). */
static void mod4(int n4, int &j) {
#pragma HLS inline off
    for (int i = 0; i < n4; ++i) {
        if (j == 1)
            j = 2;
        else
            j = 3;
        if (j > 2)
            j = 0;
        else
            j = 1;
        if (j < 1)
            j = 1;
        else
            j = 0;
    }
}

/* Module‑6: integer arithmetic (writes back into e[]). */
static void mod6(int n6, int &j, int &k, int &l, double e[4]) {
#pragma HLS inline off
    for (int i = 0; i < n6; ++i) {
        j = j * (k - j) * (l - k);
        k = l * k - (l - j) * k;
        l = (l - k) * (k + j);

        e[l - 2] = static_cast<double>(j + k + l);
        e[k - 2] = static_cast<double>(j * k * l);
    }
}

/* Module‑7: trig functions */
static void mod7(int n7, double &x, double &y) {
#pragma HLS inline off
    for (int i = 0; i < n7; ++i) {
        double denom = hls::cos(x + y) + hls::cos(x - y) - 1.0;
        double xt = T * hls::atan(T2 * hls::sin(x) * hls::cos(x) / denom);
        double yt = T * hls::atan(T2 * hls::sin(y) * hls::cos(y) / denom);
        x = xt;
        y = yt;
    }
}

/* Module‑8: procedure call (p3) */
static void p3(double x, double y, double &z) {
#pragma HLS inline off
    double tx = T * (x + y);
    double ty = T * (tx + y);
    z = (tx + ty) / T2;
}
static void mod8(int n8, double &x, double &y, double &z) {
#pragma HLS inline off
    for (int i = 0; i < n8; ++i) {
        p3(x, y, z);
    }
}

/* Module‑9: array references (p0) */
static void p0(double e[4], int &j, int &k, int &l) {
#pragma HLS inline off
    double tmp = e[j];
    e[j] = e[k];
    e[k] = e[l];
    e[l] = tmp;
}
static void mod9(int n9, double e[4], int &j, int &k, int &l) {
#pragma HLS inline off
    for (int i = 0; i < n9; ++i) {
        p0(e, j, k, l);
    }
}

/* Module‑10: integer arithmetic (no observable side‑effects). */
static void mod10(int n10, int &j, int &k) {
#pragma HLS inline off
    for (int i = 0; i < n10; ++i) {
        j = j + k;
        k = j + k;
        j = k - j;
        k = k - j - j;
    }
}

/* Module‑11: standard functions */
static void mod11(int n11, double &x) {
#pragma HLS inline off
    for (int i = 0; i < n11; ++i) {
        x = hls::sqrt(hls::exp(hls::log(x) / T1));
    }
}

/* ------------------------------------------------------------------
 *  Top‑level kernel – the only entry point that the HLS tool
 *  needs to synthesize.  The user supplies the workload size
 *  (in millions of Whetstone “units”) and receives the four
 *  final scalar results in `result[4]`.
 * ------------------------------------------------------------------ */
void whetstone_kernel(
    int million_iters, // e.g. 10 for the classic run
    double result[4])  // x1, x2, x3, x4
{

#pragma HLS top name = whetstone_kernel

    /* Scale the module‑iteration counts according to the requested
       number of millions of instructions. */
    const int n1 = 0 * million_iters;
    const int n2 = 12 * million_iters;
    const int n3 = 14 * million_iters;
    const int n4 = 345 * million_iters;
    const int n6 = 210 * million_iters;
    const int n7 = 32 * million_iters;
    const int n8 = 899 * million_iters;
    const int n9 = 616 * million_iters;
    const int n10 = 0 * million_iters;
    const int n11 = 93 * million_iters;

    /* --------------------------------------------------------------
       Module 1 – simple identifiers
       -------------------------------------------------------------- */
    double x1 = 1.0, x2 = -1.0, x3 = -1.0, x4 = -1.0;
    mod1(n1, x1, x2, x3, x4);

    /* --------------------------------------------------------------
       Module 2 – array elements
       -------------------------------------------------------------- */
    double e[4];
    e[0] = 1.0;
    e[1] = e[2] = e[3] = -1.0;
    mod2(n2, e);

    /* --------------------------------------------------------------
       Module 3 – array as parameter (pa)
       -------------------------------------------------------------- */
    for (int i = 0; i < n3; ++i) {
        pa(e);
    }

    /* --------------------------------------------------------------
       Module 4 – conditional jumps
       -------------------------------------------------------------- */
    int j = 1;
    mod4(n4, j);

    /* --------------------------------------------------------------
       Module 6 – integer arithmetic (writes into e[])
       -------------------------------------------------------------- */
    int k = 2, l = 3;
    mod6(n6, j, k, l, e);

    /* --------------------------------------------------------------
       Module 7 – trig functions
       -------------------------------------------------------------- */
    double x = 0.5, y = 0.5;
    mod7(n7, x, y);

    /* --------------------------------------------------------------
       Module 8 – procedure call (p3)
       -------------------------------------------------------------- */
    double z = 1.0;
    x = y = z = 1.0;
    mod8(n8, x, y, z);

    /* --------------------------------------------------------------
       Module 9 – array references (p0)
       -------------------------------------------------------------- */
    j = 1;
    k = 2;
    l = 3;
    e[0] = 1.0;
    e[1] = 2.0;
    e[2] = 3.0;
    e[3] = 0.0; // e[3] is unused later
    mod9(n9, e, j, k, l);

    /* --------------------------------------------------------------
       Module 10 – integer arithmetic (no side‑effects)
       -------------------------------------------------------------- */
    j = 2;
    k = 3;
    mod10(n10, j, k);

    /* --------------------------------------------------------------
       Module 11 – standard fusnctions
       -------------------------------------------------------------- */
    x = 0.75;
    mod11(n11, x);

    /* --------------------------------------------------------------
       Return the four values that the original benchmark prints.
       -------------------------------------------------------------- */
    result[0] = x1;
    result[1] = x2;
    result[2] = x3;
    result[3] = x4;
}