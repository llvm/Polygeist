/* polygeist_tensornet_cpu.c -- CPU execution of LoopLibrarian tensor-network
 * and pairwise-contraction calls through optimised libraries.
 *
 * Provides strong definitions of
 *   polygeist_cutensornet_network_f64(pointers, metadata)
 *   polygeist_cutensornet_contraction2_f64(A, B, C, metadata)
 * with the same ABI as the reference interpreters in polygeist_cublas_rt_cpu.c.
 * Link this object and weaken those two symbols in the reference runtime
 * object (objcopy --weaken-symbol=...), so the reference file stays unchanged.
 *
 * Execution model (the CPU analogue of cuTensorNet): a network is contracted
 * pairwise in a greedily chosen order (cheapest pair first), which recovers
 * sum factorisation.  Every pairwise step is mapped to a batched GEMM
 *     C[b][m][n] = sum_k A[b][m][k] * B[b][k][n]
 * over packed operands and executed by the backend selected at run time:
 *     POLYGEIST_TENSORNET_BACKEND = loops | openblas | xsmm
 * (default: xsmm if compiled in, else openblas, else loops).
 * Compile-time: -DPOLYGEIST_TN_HAVE_OPENBLAS (link -lopenblas),
 *               -DPOLYGEIST_TN_HAVE_XSMM    (link libxsmm).
 * POLYGEIST_TENSORNET_VERBOSE=1 prints the chosen contraction order once.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef POLYGEIST_TN_HAVE_XSMM
#include <libxsmm.h>
#endif

#define TN_MAX_MODES 64
#define TN_MAX_INPUTS 32

/* ------------------------------------------------------------------------ */
/* Tensors                                                                   */
/* ------------------------------------------------------------------------ */

typedef struct {
  int rank;
  int mode[TN_MAX_MODES];
  int64_t stride[TN_MAX_MODES];
  double *data;   /* base pointer (inputs are only read) */
  double *owned;  /* non-NULL for intermediates allocated here */
} tn_tensor;

static int64_t g_ext[TN_MAX_MODES]; /* extent of every mode in the current call */

/* Scratch arena for packed operands and intermediates.  Chunks persist across
 * calls and are reset (not freed) per call or per slice, so steady-state
 * execution performs no malloc/free and no page faulting. */
typedef struct { double *p; size_t cap; } tn_chunk;
static tn_chunk g_chunks[64];
static int g_nchunks = 0, g_cur = 0;
static size_t g_top = 0;
static void tn_arena_reset(void) { g_cur = 0; g_top = 0; }
static double *tn_alloc(int64_t n) {
  size_t need = (size_t)(n > 0 ? n : 1);
  need = (need + 7) & ~(size_t)7; /* keep 64-byte granularity */
  while (g_cur < g_nchunks && g_top + need > g_chunks[g_cur].cap) { ++g_cur; g_top = 0; }
  if (g_cur == g_nchunks) {
    if (g_nchunks == 64) { fprintf(stderr, "polygeist tensornet: arena exhausted\n"); abort(); }
    const size_t cap = need > ((size_t)1 << 17) ? need : ((size_t)1 << 17);
    void *p = NULL;
    if (posix_memalign(&p, 64, cap * sizeof(double))) abort();
    g_chunks[g_nchunks].p = (double *)p;
    g_chunks[g_nchunks].cap = cap;
    ++g_nchunks;
    g_top = 0;
  }
  double *r = g_chunks[g_cur].p + g_top;
  g_top += need;
  return r;
}

static int tn_has(const tn_tensor *t, int mode) {
  for (int d = 0; d < t->rank; ++d)
    if (t->mode[d] == mode) return 1;
  return 0;
}

static int64_t tn_stride(const tn_tensor *t, int mode) {
  for (int d = 0; d < t->rank; ++d)
    if (t->mode[d] == mode) return t->stride[d];
  return 0; /* absent mode: broadcast */
}

static int64_t tn_size(int n, const int *modes) {
  int64_t s = 1;
  for (int i = 0; i < n; ++i) s *= g_ext[modes[i]];
  return s;
}

/* Copy (or accumulate) `src` into a contiguous buffer laid out in `modes`
 * order (last mode fastest), reading src with its own strides.  Modes absent
 * from src are broadcast.  Odometer iteration: no division per element. */
static void tn_pack(const tn_tensor *src, int n, const int *modes, double *dst) {
  if (n == 0) { dst[0] = src->data[0]; return; }
  int64_t ext[TN_MAX_MODES], str[TN_MAX_MODES], idx[TN_MAX_MODES];
  int64_t total = 1;
  for (int i = 0; i < n; ++i) {
    ext[i] = g_ext[modes[i]];
    str[i] = tn_stride(src, modes[i]);
    idx[i] = 0;
    total *= ext[i];
  }
  const int64_t ie = ext[n - 1], is = str[n - 1];
  int64_t off = 0;
  for (int64_t p = 0; p < total; p += ie) {
    const double *s = src->data + off;
    if (is == 1) memcpy(dst + p, s, (size_t)ie * sizeof(double));
    else for (int64_t j = 0; j < ie; ++j) dst[p + j] = s[j * is];
    for (int d = n - 2; d >= 0; --d) {
      off += str[d];
      if (++idx[d] < ext[d]) break;
      off -= str[d] * ext[d];
      idx[d] = 0;
    }
  }
}

/* Is `t` already contiguous in `modes` order (so packing can be skipped)? */
static int tn_is_packed(const tn_tensor *t, int n, const int *modes) {
  if (t->rank != n) return 0;
  int64_t expect = 1;
  for (int i = n - 1; i >= 0; --i) {
    if (!tn_has(t, modes[i]) || tn_stride(t, modes[i]) != expect) return 0;
    expect *= g_ext[modes[i]];
  }
  return 1;
}

/* Write contiguous `src` (in `modes` order) into strided `dst`; add if acc. */
static void tn_scatter(const double *src, int n, const int *modes,
                       const tn_tensor *dst, int acc) {
  int64_t ext[TN_MAX_MODES], str[TN_MAX_MODES], idx[TN_MAX_MODES];
  int64_t total = 1;
  for (int i = 0; i < n; ++i) {
    ext[i] = g_ext[modes[i]];
    str[i] = tn_stride(dst, modes[i]);
    idx[i] = 0;
    total *= ext[i];
  }
  if (n == 0) { if (acc) dst->data[0] += src[0]; else dst->data[0] = src[0]; return; }
  const int64_t ie = ext[n - 1], is = str[n - 1];
  int64_t off = 0;
  for (int64_t p = 0; p < total; p += ie) {
    double *d = dst->data + off;
    if (acc) for (int64_t j = 0; j < ie; ++j) d[j * is] += src[p + j];
    else for (int64_t j = 0; j < ie; ++j) d[j * is] = src[p + j];
    for (int k = n - 2; k >= 0; --k) {
      off += str[k];
      if (++idx[k] < ext[k]) break;
      off -= str[k] * ext[k];
      idx[k] = 0;
    }
  }
}

/* ------------------------------------------------------------------------ */
/* Backends: C[b][m][n] = sum_k A[b][m][k] * B[b][k][n] (row-major, packed)  */
/* ------------------------------------------------------------------------ */

enum { TN_LOOPS = 0, TN_OPENBLAS = 1, TN_XSMM = 2 };

static int tn_backend(void) {
  static int backend = -1;
  if (backend >= 0) return backend;
#if defined(POLYGEIST_TN_HAVE_XSMM)
  backend = TN_XSMM;
#elif defined(POLYGEIST_TN_HAVE_OPENBLAS)
  backend = TN_OPENBLAS;
#else
  backend = TN_LOOPS;
#endif
  const char *e = getenv("POLYGEIST_TENSORNET_BACKEND");
  if (e) {
    if (!strcmp(e, "loops")) backend = TN_LOOPS;
#ifdef POLYGEIST_TN_HAVE_OPENBLAS
    else if (!strcmp(e, "openblas")) backend = TN_OPENBLAS;
#endif
#ifdef POLYGEIST_TN_HAVE_XSMM
    else if (!strcmp(e, "xsmm")) backend = TN_XSMM;
#endif
    else if (strcmp(e, "loops"))
      fprintf(stderr, "polygeist tensornet: backend '%s' not compiled in; using default\n", e);
  }
  return backend;
}

static void tn_gemm_loops(int64_t B, int64_t M, int64_t N, int64_t K,
                          const double *A, const double *Bm, double *C) {
  for (int64_t b = 0; b < B; ++b)
    for (int64_t m = 0; m < M; ++m) {
      double *c = C + (b * M + m) * N;
      const double *a = A + (b * M + m) * K;
      for (int64_t n = 0; n < N; ++n) c[n] = 0.0;
      for (int64_t k = 0; k < K; ++k) {
        const double av = a[k];
        const double *bp = Bm + (b * K + k) * N;
        for (int64_t n = 0; n < N; ++n) c[n] += av * bp[n];
      }
    }
}

#ifdef POLYGEIST_TN_HAVE_OPENBLAS
/* CBLAS prototype declared here to avoid depending on a particular header. */
void cblas_dgemm(int order, int transa, int transb, int m, int n, int k,
                 double alpha, const double *a, int lda, const double *b, int ldb,
                 double beta, double *c, int ldc);
static void tn_gemm_openblas(int64_t B, int64_t M, int64_t N, int64_t K,
                             const double *A, const double *Bm, double *C) {
  for (int64_t b = 0; b < B; ++b)
    cblas_dgemm(101 /*RowMajor*/, 111 /*NoTrans*/, 111, (int)M, (int)N, (int)K,
                1.0, A + b * M * K, (int)K, Bm + b * K * N, (int)N, 0.0,
                C + b * M * N, (int)N);
}
#endif

#ifdef POLYGEIST_TN_HAVE_XSMM
/* Row-major C[M][N] = A[M][K] B[K][N] is column-major C' = B' A' with
 * m' = N, n' = M, k' = K, lda' = N, ldb' = K, ldc' = N: no transposes. */
static libxsmm_gemmfunction tn_xsmm_kernel(int64_t mp, int64_t np, int64_t kp) {
  enum { CACHE = 32 };
  static struct { int64_t m, n, k; libxsmm_gemmfunction f; } cache[CACHE];
  static int used = 0;
  for (int i = 0; i < used; ++i)
    if (cache[i].m == mp && cache[i].n == np && cache[i].k == kp) return cache[i].f;
  libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(
      (libxsmm_blasint)mp, (libxsmm_blasint)np, (libxsmm_blasint)kp,
      (libxsmm_blasint)mp, (libxsmm_blasint)kp, (libxsmm_blasint)mp,
      LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64,
      LIBXSMM_DATATYPE_F64);
  libxsmm_gemmfunction f = libxsmm_dispatch_gemm(
      shape, (libxsmm_bitfield)LIBXSMM_GEMM_FLAG_BETA_0,
      (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE);
  if (used < CACHE) { cache[used].m = mp; cache[used].n = np; cache[used].k = kp; cache[used].f = f; ++used; }
  return f;
}

static void tn_gemm_xsmm(int64_t B, int64_t M, int64_t N, int64_t K,
                         const double *A, const double *Bm, double *C) {
  enum { CHUNK = 64 }; /* rows of C per JIT call: keeps kernels small */
  for (int64_t b = 0; b < B; ++b) {
    const double *Ab = A + b * M * K, *Bb = Bm + b * K * N;
    double *Cb = C + b * M * N;
    for (int64_t m0 = 0; m0 < M; m0 += CHUNK) {
      const int64_t rows = (M - m0 < CHUNK) ? (M - m0) : CHUNK;
      libxsmm_gemmfunction f = tn_xsmm_kernel(N, rows, K);
      if (!f) { tn_gemm_loops(1, rows, N, K, Ab + m0 * K, Bb, Cb + m0 * N); continue; }
      libxsmm_gemm_param p;
      memset(&p, 0, sizeof(p));
      p.a.primary = (void *)Bb;
      p.b.primary = (void *)(Ab + m0 * K);
      p.c.primary = (void *)(Cb + m0 * N);
      f(&p);
    }
  }
}
#endif

static void tn_gemm(int64_t B, int64_t M, int64_t N, int64_t K,
                    const double *A, const double *Bm, double *C) {
  if (M == 1 && N == 1 && K == 1) { /* element-wise product: no GEMM to call */
    for (int64_t b = 0; b < B; ++b) C[b] = A[b] * Bm[b];
    return;
  }
  switch (tn_backend()) {
#ifdef POLYGEIST_TN_HAVE_XSMM
  case TN_XSMM: tn_gemm_xsmm(B, M, N, K, A, Bm, C); return;
#endif
#ifdef POLYGEIST_TN_HAVE_OPENBLAS
  case TN_OPENBLAS: tn_gemm_openblas(B, M, N, K, A, Bm, C); return;
#endif
  default: tn_gemm_loops(B, M, N, K, A, Bm, C); return;
  }
}

/* ------------------------------------------------------------------------ */
/* Pairwise contraction                                                      */
/* ------------------------------------------------------------------------ */

/* Sum `t` over the modes in `red` (modes that appear only in t and are not
 * needed later), producing an owned contiguous tensor. */
static tn_tensor tn_reduce(const tn_tensor *t, int nr, const int *red) {
  int keep[TN_MAX_MODES], nk = 0, order[TN_MAX_MODES];
  for (int d = 0; d < t->rank; ++d) {
    int r = 0;
    for (int i = 0; i < nr; ++i) r |= (red[i] == t->mode[d]);
    if (!r) keep[nk++] = t->mode[d];
  }
  for (int i = 0; i < nk; ++i) order[i] = keep[i];
  for (int i = 0; i < nr; ++i) order[nk + i] = red[i];
  const int64_t outer = tn_size(nk, keep), inner = tn_size(nr, red);
  double *tmp = tn_alloc(outer * inner);
  tn_pack(t, nk + nr, order, tmp);
  tn_tensor r;
  memset(&r, 0, sizeof(r));
  r.rank = nk;
  r.data = tn_alloc(outer);
  for (int64_t o = 0; o < outer; ++o) {
    double s = 0.0;
    for (int64_t i = 0; i < inner; ++i) s += tmp[o * inner + i];
    r.data[o] = s;
  }
  int64_t st = 1;
  for (int i = nk - 1; i >= 0; --i) { r.mode[i] = keep[i]; r.stride[i] = st; st *= g_ext[keep[i]]; }
  return r;
}

/* res = sum over (modes of a ∩ b not in keep) of a*b; result modes are
 * batch ++ m ++ n (contiguous).  keep[mode] != 0 if the mode is still needed
 * (output or another live tensor). */
static tn_tensor tn_contract(const tn_tensor *a0, const tn_tensor *b0, const int *keep) {
  tn_tensor ra, rb;
  const tn_tensor *a = a0, *b = b0;
  int bat[TN_MAX_MODES], km[TN_MAX_MODES], mm[TN_MAX_MODES], nm[TN_MAX_MODES];
  int ra_m[TN_MAX_MODES], rb_m[TN_MAX_MODES];
  int nb = 0, nk = 0, nmm = 0, nn = 0, nra = 0, nrb = 0;
  for (int d = 0; d < a->rank; ++d) {
    const int m = a->mode[d];
    if (tn_has(b, m)) { if (keep[m]) bat[nb++] = m; else km[nk++] = m; }
    else if (keep[m]) mm[nmm++] = m;
    else ra_m[nra++] = m;
  }
  for (int d = 0; d < b->rank; ++d) {
    const int m = b->mode[d];
    if (tn_has(a, m)) continue;
    if (keep[m]) nm[nn++] = m; else rb_m[nrb++] = m;
  }
  if (nra) { ra = tn_reduce(a, nra, ra_m); a = &ra; }
  if (nrb) { rb = tn_reduce(b, nrb, rb_m); b = &rb; }

  const int64_t B = tn_size(nb, bat), M = tn_size(nmm, mm),
                N = tn_size(nn, nm), K = tn_size(nk, km);
  int oa[TN_MAX_MODES], ob[TN_MAX_MODES], na = 0, nbb = 0;
  for (int i = 0; i < nb; ++i) oa[na++] = bat[i];
  for (int i = 0; i < nmm; ++i) oa[na++] = mm[i];
  for (int i = 0; i < nk; ++i) oa[na++] = km[i];
  for (int i = 0; i < nb; ++i) ob[nbb++] = bat[i];
  for (int i = 0; i < nk; ++i) ob[nbb++] = km[i];
  for (int i = 0; i < nn; ++i) ob[nbb++] = nm[i];

  const double *A = a->data, *Bm = b->data;
  if (!tn_is_packed(a, na, oa)) { double *pa = tn_alloc(B * M * K); tn_pack(a, na, oa, pa); A = pa; }
  if (!tn_is_packed(b, nbb, ob)) { double *pb = tn_alloc(B * K * N); tn_pack(b, nbb, ob, pb); Bm = pb; }

  tn_tensor r;
  memset(&r, 0, sizeof(r));
  r.data = tn_alloc(B * M * N);
  tn_gemm(B, M, N, K, A, Bm, r.data);

  r.rank = nb + nmm + nn;
  int pos = 0;
  for (int i = 0; i < nb; ++i) r.mode[pos++] = bat[i];
  for (int i = 0; i < nmm; ++i) r.mode[pos++] = mm[i];
  for (int i = 0; i < nn; ++i) r.mode[pos++] = nm[i];
  int64_t st = 1;
  for (int i = r.rank - 1; i >= 0; --i) { r.stride[i] = st; st *= g_ext[r.mode[i]]; }
  return r;
}

/* ------------------------------------------------------------------------ */
/* Network execution (greedy pairwise order)                                 */
/* ------------------------------------------------------------------------ */

/* Contraction order.  Tensors are mode bitmasks; contracting live[i] and
 * live[j] (i < j) costs prod(extents of their union) multiply-adds and leaves
 * the union restricted to modes still needed (output or another live tensor)
 * in slot i, with the last tensor moved into slot j.  For up to 8 inputs the
 * order minimising total multiply-adds is found exhaustively (branch and
 * bound); larger networks fall back to greedy cheapest-pair-first.  Plans are
 * cached per network signature, so the search runs once per call site. */
static double tn_mask_size(uint64_t mask) {
  double s = 1.0;
  for (int m = 0; m < TN_MAX_MODES; ++m)
    if (mask >> m & 1u) s *= (double)g_ext[m];
  return s;
}

static uint64_t tn_needed(const uint64_t *mask, int n, int i, int j, uint64_t outmask) {
  uint64_t need = outmask;
  for (int o = 0; o < n; ++o)
    if (o != i && o != j) need |= mask[o];
  return need;
}

static void tn_search(uint64_t *mask, int n, uint64_t outmask, double acc, int depth,
                      int *pi, int *pj, double *best, int *bpi, int *bpj) {
  if (acc >= *best) return;
  if (n == 1) {
    *best = acc;
    for (int d = 0; d < depth; ++d) { bpi[d] = pi[d]; bpj[d] = pj[d]; }
    return;
  }
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      const uint64_t mi = mask[i], mj = mask[j], ml = mask[n - 1];
      const double cost = tn_mask_size(mi | mj);
      mask[i] = (mi | mj) & tn_needed(mask, n, i, j, outmask);
      mask[j] = ml;
      pi[depth] = i;
      pj[depth] = j;
      tn_search(mask, n - 1, outmask, acc + cost, depth + 1, pi, pj, best, bpi, bpj);
      mask[n - 1] = ml;
      mask[j] = mj;
      mask[i] = mi;
    }
}

typedef struct {
  int valid, n;
  int64_t sig[TN_MAX_INPUTS + 1 + 2 * TN_MAX_MODES];
  int pi[TN_MAX_INPUTS], pj[TN_MAX_INPUTS];
} tn_plan;

static void tn_make_plan(const tn_tensor *in, int n, const tn_tensor *out, int *pi, int *pj) {
  enum { PLANS = 32 };
  static tn_plan cache[PLANS];
  static int used = 0;
  uint64_t mask[TN_MAX_INPUTS], outmask = 0;
  for (int i = 0; i < n; ++i) {
    mask[i] = 0;
    for (int d = 0; d < in[i].rank; ++d) mask[i] |= 1ull << in[i].mode[d];
  }
  for (int d = 0; d < out->rank; ++d) outmask |= 1ull << out->mode[d];
  /* signature: tensor masks, output mask and every mode extent */
  int64_t sig[TN_MAX_INPUTS + 1 + 2 * TN_MAX_MODES];
  int ns = 0;
  for (int i = 0; i < n; ++i) sig[ns++] = (int64_t)mask[i];
  sig[ns++] = (int64_t)outmask;
  for (int m = 0; m < TN_MAX_MODES; ++m) sig[ns++] = g_ext[m];
  for (int c = 0; c < used; ++c)
    if (cache[c].n == n && !memcmp(cache[c].sig, sig, (size_t)ns * sizeof(int64_t))) {
      memcpy(pi, cache[c].pi, sizeof(int) * (size_t)(n - 1));
      memcpy(pj, cache[c].pj, sizeof(int) * (size_t)(n - 1));
      return;
    }
  if (n <= 8) {
    int cpi[TN_MAX_INPUTS], cpj[TN_MAX_INPUTS];
    double best = 1e300;
    tn_search(mask, n, outmask, 0.0, 0, cpi, cpj, &best, pi, pj);
  } else {
    int k = n;
    for (int step = 0; k > 1; ++step) {
      int bi = 0, bj = 1;
      double best = -1.0;
      for (int i = 0; i < k; ++i)
        for (int j = i + 1; j < k; ++j) {
          const double c = tn_mask_size(mask[i] | mask[j]);
          if (best < 0 || c < best) { best = c; bi = i; bj = j; }
        }
      pi[step] = bi;
      pj[step] = bj;
      mask[bi] = (mask[bi] | mask[bj]) & tn_needed(mask, k, bi, bj, outmask);
      mask[bj] = mask[k - 1];
      --k;
    }
  }
  if (used < PLANS) {
    cache[used].valid = 1;
    cache[used].n = n;
    memcpy(cache[used].sig, sig, (size_t)ns * sizeof(int64_t));
    memcpy(cache[used].pi, pi, sizeof(int) * (size_t)(n - 1));
    memcpy(cache[used].pj, pj, sizeof(int) * (size_t)(n - 1));
    ++used;
  }
}

static int g_verbose = -1;

/* Execute a planned pairwise order on (possibly sliced) tensors; all scratch
 * comes from the arena. */
static void tn_execute(int nin, const tn_tensor *in, const tn_tensor *out, int acc,
                       const int *plan_i, const int *plan_j) {
  tn_tensor live[TN_MAX_INPUTS];
  int n = nin;
  for (int i = 0; i < n; ++i) live[i] = in[i];
  int in_out[TN_MAX_MODES] = {0};
  for (int d = 0; d < out->rank; ++d) in_out[out->mode[d]] = 1;
  const int verbose = g_verbose;

  for (int step = 0; n > 1; ++step) {
    const int bi = plan_i[step], bj = plan_j[step];
    const double best = 0.0;
    int keep[TN_MAX_MODES];
    for (int m = 0; m < TN_MAX_MODES; ++m) {
      keep[m] = in_out[m];
      for (int o = 0; o < n && !keep[m]; ++o)
        if (o != bi && o != bj && tn_has(&live[o], m)) keep[m] = 1;
    }
    if (verbose == 1) {
      uint64_t u = 0;
      for (int d = 0; d < live[bi].rank; ++d) u |= 1ull << live[bi].mode[d];
      for (int d = 0; d < live[bj].rank; ++d) u |= 1ull << live[bj].mode[d];
      fprintf(stderr, "polygeist tensornet: step %d contracts (%d,%d) of %d live, %.0f mult-adds\n",
              step, bi, bj, n, tn_mask_size(u) + best);
    }
    tn_tensor r = tn_contract(&live[bi], &live[bj], keep);
    live[bi] = r;
    live[bj] = live[n - 1];
    --n;
  }
  if (g_verbose == 1) g_verbose = 2; /* print the order for the first slice only */

  /* live[0] has exactly the output modes (any extra mode would be a sum over
   * a single tensor, handled by tn_contract's reduction). */
  tn_tensor f = live[0];
  int order[TN_MAX_MODES];
  for (int d = 0; d < out->rank; ++d) order[d] = out->mode[d];
  double *tmp = tn_alloc(tn_size(out->rank, order));
  tn_pack(&f, out->rank, order, tmp); /* output modes absent in f broadcast */
  tn_scatter(tmp, out->rank, order, out, acc);
}

/* ------------------------------------------------------------------------ */
/* Compiled per-slice plans                                                  */
/* ------------------------------------------------------------------------ */

/* Element offsets (relative to t->data) of t laid out in `modes` order. */
static void tn_pack_offsets(const tn_tensor *t, int n, const int *modes, int32_t *dst) {
  int64_t ext[TN_MAX_MODES], str[TN_MAX_MODES], idx[TN_MAX_MODES], total = 1;
  for (int i = 0; i < n; ++i) {
    ext[i] = g_ext[modes[i]];
    str[i] = tn_stride(t, modes[i]);
    idx[i] = 0;
    total *= ext[i];
  }
  int64_t off = 0;
  for (int64_t p = 0; p < total; ++p) {
    dst[p] = (int32_t)off;
    for (int d = n - 1; d >= 0; --d) {
      off += str[d];
      if (++idx[d] < ext[d]) break;
      off -= str[d] * ext[d];
      idx[d] = 0;
    }
  }
}

typedef struct {
  int src_a, src_b;       /* >= 0: input index; < 0: result of step -(src+1) */
  int64_t B, M, N, K;
  int32_t *ga, *gb;       /* gather tables, NULL if the operand is already packed */
  int64_t na, nb;
  double *pa, *pb, *res;  /* scratch reused by every slice */
} tn_cstep;

typedef struct {
  int nsteps;
  tn_cstep step[TN_MAX_INPUTS];
  int final_src;          /* input index or -(step+1) */
  int64_t nout;
  int32_t *gf, *so;       /* final gather (from result) and scatter (into output) */
  double *fbuf;
} tn_cplan;

/* Build a compiled plan from slice-0 views.  Returns 0 if some step needs a
 * single-tensor reduction (handled only by the generic executor). */
static int tn_compile(int nin, const tn_tensor *in, const tn_tensor *out,
                      const int *plan_i, const int *plan_j, tn_cplan *cp) {
  tn_tensor live[TN_MAX_INPUTS];
  int src[TN_MAX_INPUTS], n = nin;
  for (int i = 0; i < n; ++i) { live[i] = in[i]; src[i] = i; }
  int in_out[TN_MAX_MODES] = {0};
  for (int d = 0; d < out->rank; ++d) in_out[out->mode[d]] = 1;
  cp->nsteps = 0;
  for (int s = 0; n > 1; ++s) {
    const int bi = plan_i[s], bj = plan_j[s];
    int keep[TN_MAX_MODES];
    for (int m = 0; m < TN_MAX_MODES; ++m) {
      keep[m] = in_out[m];
      for (int o = 0; o < n && !keep[m]; ++o)
        if (o != bi && o != bj && tn_has(&live[o], m)) keep[m] = 1;
    }
    const tn_tensor *a = &live[bi], *b = &live[bj];
    int bat[TN_MAX_MODES], km[TN_MAX_MODES], mm[TN_MAX_MODES], nm[TN_MAX_MODES];
    int nb = 0, nk = 0, nmm = 0, nn = 0;
    for (int d = 0; d < a->rank; ++d) {
      const int m = a->mode[d];
      if (tn_has(b, m)) { if (keep[m]) bat[nb++] = m; else km[nk++] = m; }
      else if (keep[m]) mm[nmm++] = m;
      else return 0;
    }
    for (int d = 0; d < b->rank; ++d) {
      const int m = b->mode[d];
      if (tn_has(a, m)) continue;
      if (keep[m]) nm[nn++] = m; else return 0;
    }
    tn_cstep *st = &cp->step[cp->nsteps++];
    st->src_a = src[bi];
    st->src_b = src[bj];
    st->B = tn_size(nb, bat); st->M = tn_size(nmm, mm);
    st->N = tn_size(nn, nm); st->K = tn_size(nk, km);
    int oa[TN_MAX_MODES], ob[TN_MAX_MODES], na = 0, nbb = 0;
    for (int i = 0; i < nb; ++i) oa[na++] = bat[i];
    for (int i = 0; i < nmm; ++i) oa[na++] = mm[i];
    for (int i = 0; i < nk; ++i) oa[na++] = km[i];
    for (int i = 0; i < nb; ++i) ob[nbb++] = bat[i];
    for (int i = 0; i < nk; ++i) ob[nbb++] = km[i];
    for (int i = 0; i < nn; ++i) ob[nbb++] = nm[i];
    st->na = st->B * st->M * st->K;
    st->nb = st->B * st->K * st->N;
    st->ga = NULL; st->gb = NULL; st->pa = NULL; st->pb = NULL;
    if (!tn_is_packed(a, na, oa)) {
      st->ga = malloc((size_t)st->na * sizeof(int32_t));
      tn_pack_offsets(a, na, oa, st->ga);
      st->pa = malloc((size_t)st->na * sizeof(double));
    }
    if (!tn_is_packed(b, nbb, ob)) {
      st->gb = malloc((size_t)st->nb * sizeof(int32_t));
      tn_pack_offsets(b, nbb, ob, st->gb);
      st->pb = malloc((size_t)st->nb * sizeof(double));
    }
    st->res = malloc((size_t)(st->B * st->M * st->N) * sizeof(double));
    tn_tensor r;
    memset(&r, 0, sizeof(r));
    r.rank = nb + nmm + nn;
    int pos = 0;
    for (int i = 0; i < nb; ++i) r.mode[pos++] = bat[i];
    for (int i = 0; i < nmm; ++i) r.mode[pos++] = mm[i];
    for (int i = 0; i < nn; ++i) r.mode[pos++] = nm[i];
    int64_t stv = 1;
    for (int i = r.rank - 1; i >= 0; --i) { r.stride[i] = stv; stv *= g_ext[r.mode[i]]; }
    live[bi] = r;
    src[bi] = -(cp->nsteps);
    live[bj] = live[n - 1];
    src[bj] = src[n - 1];
    --n;
  }
  int order[TN_MAX_MODES];
  for (int d = 0; d < out->rank; ++d) order[d] = out->mode[d];
  cp->final_src = src[0];
  cp->nout = tn_size(out->rank, order);
  cp->gf = malloc((size_t)cp->nout * sizeof(int32_t));
  cp->so = malloc((size_t)cp->nout * sizeof(int32_t));
  cp->fbuf = NULL;
  tn_pack_offsets(&live[0], out->rank, order, cp->gf);
  tn_pack_offsets(out, out->rank, order, cp->so);
  return 1;
}

static void tn_free_cplan(tn_cplan *cp) {
  for (int s = 0; s < cp->nsteps; ++s) {
    free(cp->step[s].ga); free(cp->step[s].gb);
    free(cp->step[s].pa); free(cp->step[s].pb); free(cp->step[s].res);
  }
  free(cp->gf);
  free(cp->so);
}

/* Run a compiled plan for one slice: `base[i]` are the sliced input bases. */
static void tn_run_compiled(const tn_cplan *cp, const double *const *base, double *outb, int acc) {
  for (int s = 0; s < cp->nsteps; ++s) {
    const tn_cstep *st = &cp->step[s];
    const double *A = st->src_a >= 0 ? base[st->src_a] : cp->step[-st->src_a - 1].res;
    const double *Bm = st->src_b >= 0 ? base[st->src_b] : cp->step[-st->src_b - 1].res;
    if (st->ga) { for (int64_t t = 0; t < st->na; ++t) st->pa[t] = A[st->ga[t]]; A = st->pa; }
    if (st->gb) { for (int64_t t = 0; t < st->nb; ++t) st->pb[t] = Bm[st->gb[t]]; Bm = st->pb; }
    tn_gemm(st->B, st->M, st->N, st->K, A, Bm, st->res);
  }
  const double *f = cp->final_src >= 0 ? base[cp->final_src] : cp->step[-cp->final_src - 1].res;
  if (acc) for (int64_t t = 0; t < cp->nout; ++t) outb[cp->so[t]] += f[cp->gf[t]];
  else for (int64_t t = 0; t < cp->nout; ++t) outb[cp->so[t]] = f[cp->gf[t]];
}

/* Remove `mode` from a tensor view, offsetting its data to slice `idx`. */
static tn_tensor tn_slice(const tn_tensor *t, int mode, int64_t idx) {
  tn_tensor s = *t;
  s.rank = 0;
  for (int d = 0; d < t->rank; ++d) {
    if (t->mode[d] == mode) { s.data = t->data + idx * t->stride[d]; continue; }
    s.mode[s.rank] = t->mode[d];
    s.stride[s.rank] = t->stride[d];
    ++s.rank;
  }
  return s;
}

/* Slicing (as in cuTensorNet): when an output mode is large, contract the
 * network once per index of that mode.  Every slice is a small network whose
 * operands and intermediates stay in cache; for MFEM this is the per-element
 * sum-factorised operator.  POLYGEIST_TENSORNET_SLICE=0 disables slicing. */
static void tn_run(int nin, tn_tensor *in, const tn_tensor *out, int acc) {
  if (g_verbose < 0) { const char *v = getenv("POLYGEIST_TENSORNET_VERBOSE"); g_verbose = v && *v == '1'; }
  static int slice_on = -1;
  if (slice_on < 0) { const char *s = getenv("POLYGEIST_TENSORNET_SLICE"); slice_on = !(s && *s == '0'); }
  int smode = -1;
  if (slice_on)
    for (int d = 0; d < out->rank; ++d) {
      const int m = out->mode[d];
      int in_any = 0;
      for (int i = 0; i < nin; ++i) in_any |= tn_has(&in[i], m);
      if (in_any && g_ext[m] >= 16 && (smode < 0 || g_ext[m] > g_ext[smode])) smode = m;
    }
  int plan_i[TN_MAX_INPUTS], plan_j[TN_MAX_INPUTS];
  if (smode < 0) {
    tn_arena_reset();
    tn_make_plan(in, nin, out, plan_i, plan_j);
    tn_execute(nin, in, out, acc, plan_i, plan_j);
    return;
  }
  const int64_t E = g_ext[smode];
  tn_tensor sin[TN_MAX_INPUTS], sout;
  for (int i = 0; i < nin; ++i) sin[i] = tn_slice(&in[i], smode, 0);
  sout = tn_slice(out, smode, 0);
  g_ext[smode] = 1; /* plan the per-slice network */
  tn_make_plan(sin, nin, &sout, plan_i, plan_j);
  if (g_verbose == 1)
    fprintf(stderr, "polygeist tensornet: slicing mode %d into %lld slices\n", smode, (long long)E);
  tn_cplan cp;
  static int phase = -1;
  static double ph_compile = 0, ph_loop = 0, ph_free = 0;
  static long ph_calls = 0;
  if (phase < 0) {
    const char *p = getenv("POLYGEIST_TENSORNET_PROFILE");
    phase = p && *p == '2';
  }
  struct timespec ts0, ts1, ts2, ts3;
  if (phase) clock_gettime(CLOCK_MONOTONIC, &ts0);
  if (tn_compile(nin, sin, &sout, plan_i, plan_j, &cp)) {
    if (phase) clock_gettime(CLOCK_MONOTONIC, &ts1);
    if (g_verbose == 1) {
      for (int s = 0; s < cp.nsteps; ++s)
        fprintf(stderr, "polygeist tensornet: compiled step %d: batch %lld, gemm %lldx%lldx%lld%s%s\n", s,
                (long long)cp.step[s].B, (long long)cp.step[s].M, (long long)cp.step[s].N,
                (long long)cp.step[s].K, cp.step[s].ga ? ", gather A" : "", cp.step[s].gb ? ", gather B" : "");
      g_verbose = 2;
    }
    int64_t sstride[TN_MAX_INPUTS] = {0}, ostride = 0;
    for (int i = 0; i < nin; ++i) sstride[i] = tn_stride(&in[i], smode);
    ostride = tn_stride(out, smode);
    const double *base[TN_MAX_INPUTS];
    for (int64_t e = 0; e < E; ++e) {
      for (int i = 0; i < nin; ++i) base[i] = in[i].data + e * sstride[i];
      tn_run_compiled(&cp, base, out->data + e * ostride, acc);
    }
    if (phase) clock_gettime(CLOCK_MONOTONIC, &ts2);
    tn_free_cplan(&cp);
    if (phase) {
      clock_gettime(CLOCK_MONOTONIC, &ts3);
#define TN_NS(a, b) ((double)((b).tv_sec - (a).tv_sec) * 1e9 + (double)((b).tv_nsec - (a).tv_nsec))
      ph_compile += TN_NS(ts0, ts1); ph_loop += TN_NS(ts1, ts2); ph_free += TN_NS(ts2, ts3);
      if (++ph_calls % 100 == 0)
        fprintf(stderr, "polygeist tensornet: per call compile %.1f us, slice loop %.1f us, free %.1f us\n",
                ph_compile * 1e-3 / ph_calls, ph_loop * 1e-3 / ph_calls, ph_free * 1e-3 / ph_calls);
    }
    g_ext[smode] = E;
    return;
  }
  for (int64_t e = 0; e < E; ++e) {
    tn_arena_reset();
    for (int i = 0; i < nin; ++i) sin[i] = tn_slice(&in[i], smode, e);
    sout = tn_slice(out, smode, e);
    tn_execute(nin, sin, &sout, acc, plan_i, plan_j);
  }
  g_ext[smode] = E;
}

static int tn_add_dim(tn_tensor *t, int64_t extent, int64_t stride, int64_t mode) {
  if (mode < 0 || mode >= TN_MAX_MODES || extent <= 0 || tn_has(t, (int)mode)) return 0;
  if (g_ext[mode] != 1 && g_ext[mode] != extent) return 0;
  g_ext[mode] = extent;
  t->mode[t->rank] = (int)mode;
  t->stride[t->rank] = stride;
  t->rank++;
  return 1;
}

/* ------------------------------------------------------------------------ */
/* ABI entry points                                                          */
/* ------------------------------------------------------------------------ */

/* POLYGEIST_TENSORNET_PROFILE=1: report total time spent inside this runtime
 * at exit, to separate library time from the surrounding compiled code. */
#include <time.h>
static double g_prof_ns = 0.0;
static long g_prof_calls = 0;
static int g_prof = -1;
static double tn_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}
static void tn_prof_report(void) {
  fprintf(stderr, "polygeist tensornet: %ld calls, %.3f ms total in runtime, %.3f us/call\n",
          g_prof_calls, g_prof_ns * 1e-6, g_prof_calls ? g_prof_ns * 1e-3 / g_prof_calls : 0.0);
}
static int tn_prof_on(void) {
  if (g_prof < 0) {
    const char *p = getenv("POLYGEIST_TENSORNET_PROFILE");
    g_prof = p && *p == '1';
    if (g_prof) atexit(tn_prof_report);
  }
  return g_prof;
}

static void tn_network_impl(const int64_t *pointers, const int64_t *metadata);

void polygeist_cutensornet_network_f64(const int64_t *pointers, const int64_t *metadata) {
  if (!tn_prof_on()) { tn_network_impl(pointers, metadata); return; }
  const double t0 = tn_now_ns();
  tn_network_impl(pointers, metadata);
  g_prof_ns += tn_now_ns() - t0;
  ++g_prof_calls;
}

static void tn_network_impl(const int64_t *pointers, const int64_t *metadata) {
  if (!pointers || !metadata || metadata[0] != 1) {
    fprintf(stderr, "polygeist tensornet: invalid tensor-network ABI\n");
    return;
  }
  const int64_t nin = metadata[1];
  const int acc = metadata[2] != 0;
  if (nin < 2 || nin > TN_MAX_INPUTS) {
    fprintf(stderr, "polygeist tensornet: invalid input count\n");
    return;
  }
  for (int m = 0; m < TN_MAX_MODES; ++m) g_ext[m] = 1;
  tn_tensor t[TN_MAX_INPUTS + 1];
  memset(t, 0, sizeof(t));
  int64_t cursor = 3 + (nin + 1);
  for (int64_t i = 0; i <= nin; ++i) {
    const int64_t rank = metadata[3 + i];
    t[i].data = (double *)(uintptr_t)pointers[i];
    for (int64_t d = 0; d < rank; ++d, cursor += 3)
      if (!tn_add_dim(&t[i], metadata[cursor], metadata[cursor + 1], metadata[cursor + 2])) {
        fprintf(stderr, "polygeist tensornet: unsupported tensor-network metadata\n");
        return;
      }
  }
  static int dumped = 0;
  const char *dump = getenv("POLYGEIST_TENSORNET_DUMP");
  if (dump && *dump == '1' && !dumped) {
    dumped = 1;
    for (int64_t i = 0; i <= nin; ++i) {
      fprintf(stderr, "polygeist tensornet: %s %lld ptr=%p:", i == nin ? "output" : "input",
              (long long)i, (void *)t[i].data);
      for (int d = 0; d < t[i].rank; ++d)
        fprintf(stderr, " (mode %d ext %lld stride %lld)", t[i].mode[d],
                (long long)g_ext[t[i].mode[d]], (long long)t[i].stride[d]);
      fprintf(stderr, "\n");
    }
  }
  tn_run((int)nin, t, &t[nin], acc);
}

void polygeist_cutensornet_network_f64_device(const int64_t *pointers, const int64_t *metadata) {
  polygeist_cutensornet_network_f64(pointers, metadata);
}

static void tn_contraction2_impl(const double *A, const double *B, double *C,
                                 const int64_t *metadata);

void polygeist_cutensornet_contraction2_f64(const double *A, const double *B, double *C,
                                            const int64_t *metadata) {
  if (!tn_prof_on()) { tn_contraction2_impl(A, B, C, metadata); return; }
  const double t0 = tn_now_ns();
  tn_contraction2_impl(A, B, C, metadata);
  g_prof_ns += tn_now_ns() - t0;
  ++g_prof_calls;
}

static void tn_contraction2_impl(const double *A, const double *B, double *C,
                                 const int64_t *metadata) {
  enum { MAX_RANK = 64, FIELDS = 3 * MAX_RANK };
  for (int m = 0; m < TN_MAX_MODES; ++m) g_ext[m] = 1;
  tn_tensor t[3];
  memset(t, 0, sizeof(t));
  t[0].data = (double *)A;
  t[1].data = (double *)B;
  t[2].data = C;
  for (int i = 0; i < 3; ++i) {
    const int64_t rank = metadata[i], base = 3 + (int64_t)i * FIELDS;
    for (int64_t d = 0; d < rank; ++d)
      if (!tn_add_dim(&t[i], metadata[base + d], metadata[base + MAX_RANK + d],
                      metadata[base + 2 * MAX_RANK + d])) {
        fprintf(stderr, "polygeist tensornet: unsupported contraction metadata\n");
        return;
      }
  }
  tn_run(2, t, &t[2], 0);
}

void polygeist_cutensornet_contraction2_f64_device(const double *A, const double *B, double *C,
                                                   const int64_t *metadata) {
  polygeist_cutensornet_contraction2_f64(A, B, C, metadata);
}
