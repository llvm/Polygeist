/* Complete-buffer runtime regression at the existing ATen fixture size.
 * Covers pointer rebinding/reallocation, aliasing, strides, mapped host memory,
 * CUDA Graph capture/replay and cuTENSOR plan eviction. This is correctness
 * validation, not a native ATen implementation or a performance benchmark.
 */
/* Isolated runtime contract tests; no benchmark/native replacement kernels. */
#include "polygeist_cublas_rt.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N 4096
#define M (2 * N + 128)
#define BYTES (M * sizeof(float))
#define CU(call)                                                               \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA %s\n", cudaGetErrorString(e));                     \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)
#ifdef POLYGEIST_TEST_PROFILE
void profile_begin(const char *);
void profile_end(void);
#else
static void profile_begin(const char *label) { (void)label; }
static void profile_end(void) {}
#endif
static int failures;
static size_t checked;
static void *kept_host[4], *kept_device[12];
static int nh, nd;
static float value(int i, int edge) {
  static const float e[] = {0.0f,      -0.0f, 1,      -1,      2,  -2, INFINITY,
                            -INFINITY, NAN,   1e-20f, -1e-20f, 80, -80};
  return edge ? e[i % 13] : (float)((i * 104729LL) % 8192) / 1024.0f - 4;
}
static int equal(float a, float b) {
  if (isnan(a) || isnan(b))
    return isnan(a) && isnan(b);
  if (a == 0 && b == 0)
    return signbit(a) == signbit(b);
  if (!isfinite(a) || !isfinite(b))
    return a == b;
  return fabsf(a - b) <= 1e-5f + 1e-4f * fabsf(b);
}
static void pointwise(float *x, float *y, int si, int so) {
  polygeist_cudnn_pointwise_graph_f32(N, (int64_t)9 << 24, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, si,
                                      1, 1, 1, so, x, NULL, NULL, NULL, y);
}
static void validate(const char *name, const float *got, const float *want) {
  int bad = 0;
  for (int i = 0; i < M; i++)
    if (!equal(got[i], want[i])) {
      if (bad < 2)
        printf("DIFF %s i=%d got=%a want=%a\n", name, i, got[i], want[i]);
      bad++;
    }
  checked += M;
  failures += bad;
  printf("SAFETY %s count=%d bad=%d\n", name, M, bad);
}
static void contract(const char *name, int host, int alias, int si, int so,
                     int ix, int iy, int edge) {
  float *a, *b, *expected_a = malloc(BYTES), *expected_b = malloc(BYTES),
                *got = malloc(BYTES);
  if (posix_memalign((void **)&a, 4096, BYTES) ||
      posix_memalign((void **)&b, 4096, BYTES)) {
    fprintf(stderr, "host test allocation failed\n");
    exit(2);
  }
  for (int i = 0; i < M; i++)
    a[i] = b[i] = 913.25f;
  for (int i = 0; i < N; i++)
    a[ix + i * si] = value(i, edge);
  memcpy(expected_a, a, BYTES);
  memcpy(expected_b, b, BYTES);
  for (int i = 0; i < N; i++)
    (alias ? expected_a : expected_b)[iy + i * so] = fabsf(a[ix + i * si]);
  float *da = a, *db = b;
  if (!host) {
    CU(cudaMalloc((void **)&da, BYTES));
    CU(cudaMalloc((void **)&db, BYTES));
    CU(cudaMemcpy(da, a, BYTES, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(db, b, BYTES, cudaMemcpyHostToDevice));
  }
  printf("ALLOC %s input=%p output=%p host=%d\n", name, (void *)(da + ix),
         (void *)((alias ? da : db) + iy), host);
  profile_begin(name);
  pointwise(da + ix, (alias ? da : db) + iy, si, so);
  profile_end();
  CU(cudaDeviceSynchronize());
  if (host)
    memcpy(got, a, BYTES);
  else
    CU(cudaMemcpy(got, da, BYTES, cudaMemcpyDeviceToHost));
  validate(name, got, expected_a);
  if (host)
    memcpy(got, b, BYTES);
  else
    CU(cudaMemcpy(got, db, BYTES, cudaMemcpyDeviceToHost));
  validate(name, got, expected_b);
  if (!host) {
    CU(cudaFree(da));
    CU(cudaFree(db));
  }
  /* Keep mapped host allocations alive for the runtime registration lifetime.
   */
  if (host) {
    kept_host[nh++] = a;
    kept_host[nh++] = b;
  } else {
    free(a);
    free(b);
  }
  free(expected_a);
  free(expected_b);
  free(got);
}
static void graph_case(const char *name, int change, int cutensor) {
  float *d[4], *h = malloc(BYTES), *want = malloc(BYTES), *got = malloc(BYTES);
  for (int i = 0; i < M; i++)
    h[i] = want[i] = 913.25f;
  for (int i = 0; i < N; i++)
    h[i + 4] = value(i, 1);
  for (int i = 0; i < N; i++)
    want[i + 4] = cutensor ? acosf(h[i + 4]) : fabsf(h[i + 4]);
  for (int a = 0; a < 4; a++) {
    CU(cudaMalloc((void **)&d[a], BYTES));
    CU(cudaMemcpy(d[a], a % 2 ? want : h, BYTES, cudaMemcpyHostToDevice));
  }
  for (int iteration = 0; iteration < 3; iteration++) {
    int index = change && iteration ? 2 : 0;
    CU(cudaMemset(d[index + 1] + 4, 0, N * sizeof(float)));
    profile_begin(name);
    if (polygeist_cuda_graph_begin(100 + change + 2 * cutensor)) {
      if (cutensor)
        polygeist_cutensor_unary_f32(POLYGEIST_CUTENSOR_UNARY_ACOS, N,
                                     d[index] + 4, d[index + 1] + 4);
      else
        pointwise(d[index] + 4, d[index + 1] + 4, 1, 1);
      polygeist_cuda_graph_end(100 + change + 2 * cutensor);
    }
    profile_end();
    CU(cudaMemcpy(got, d[index + 1], BYTES, cudaMemcpyDeviceToHost));
    validate(name, got, want);
  }
  for (int a = 0; a < 4; a++)
    kept_device[nd++] = d[a];
  free(h);
  free(want);
  free(got);
}
static void cutensor_cache(void) {
  float *a = malloc(BYTES), *b = malloc(BYTES), *want = malloc(BYTES), *da, *db;
  CU(cudaMalloc((void **)&da, BYTES));
  CU(cudaMalloc((void **)&db, BYTES));
  const int offsets[] = {1, 2, 4, 8, 16, 32};
  for (int pass = 0; pass < 19; pass++) {
    int ix = offsets[pass == 18 ? 0 : (pass / 6) % 6],
        iy = offsets[pass == 18 ? 0 : pass % 6];
    for (int i = 0; i < M; i++)
      a[i] = b[i] = want[i] = 913.25f;
    for (int i = 0; i < N; i++) {
      a[ix + i] = sinf((float)i) * 0.9f;
      want[iy + i] = acosf(a[ix + i]);
    }
    CU(cudaMemcpy(da, a, BYTES, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(db, b, BYTES, cudaMemcpyHostToDevice));
    profile_begin(pass == 18 ? "cutensor-evicted-key"
                             : "cutensor-new-alignment");
    polygeist_cutensor_unary_f32(POLYGEIST_CUTENSOR_UNARY_ACOS, N, da + ix,
                                 db + iy);
    profile_end();
    CU(cudaMemcpy(b, db, BYTES, cudaMemcpyDeviceToHost));
    validate("cutensor-cache-alignment-eviction", b, want);
  }
  CU(cudaFree(da));
  CU(cudaFree(db));
  free(a);
  free(b);
  free(want);
}
static void replay_after_rebinding_and_eviction(void) {
  float *got = malloc(BYTES), *want = malloc(BYTES);
  const int slots[] = {0, 6, 8};
  for (int graph = 0; graph < 3; graph++) {
    for (int i = 0; i < M; i++)
      want[i] = 913.25f;
    for (int i = 0; i < N; i++)
      want[i + 4] = graph == 2 ? acosf(value(i, 1)) : fabsf(value(i, 1));
    float *out = (float *)kept_device[slots[graph] + 1];
    CU(cudaMemset(out + 4, 0, N * sizeof(float)));
    if (polygeist_cuda_graph_begin(100 + graph)) {
      fprintf(stderr, "expected captured graph replay\n");
      exit(4);
    }
    CU(cudaMemcpy(got, out, BYTES, cudaMemcpyDeviceToHost));
    validate("replay-after-rebinding-and-eviction", got, want);
  }
  free(got);
  free(want);
}
int main(void) {
  setenv("POLYGEIST_CUDA_GRAPH", "1", 1);
  polygeist_require_gpu_libraries_begin();
  contract("direct-offset4", 0, 0, 1, 1, 1, 3, 0);
  contract("changed-pointers-edge", 0, 0, 1, 1, 4, 4, 1);
  contract("reallocate-direct", 0, 0, 1, 1, 1, 3, 1);
  contract("reallocate-same-tuple", 0, 0, 1, 1, 1, 3, 1);
  contract("exact-alias-fallback", 0, 1, 1, 1, 4, 4, 1);
  contract("partial-overlap-fallback", 0, 1, 1, 1, 4, 5, 1);
  contract("strided-fallback", 0, 0, 2, 2, 4, 4, 1);
  contract("mapped-host-direct", 1, 0, 1, 1, 4, 4, 1);
  contract("mapped-host-alias", 1, 1, 1, 1, 4, 4, 1);
  graph_case("capture-stable-pointer", 0, 0);
  graph_case("capture-changed-pointer-fallback", 1, 0);
  graph_case("capture-cutensor-stable", 0, 1);
  cutensor_cache();
  replay_after_rebinding_and_eviction();
  polygeist_cublas_destroy();
  for (int a = 0; a < nd; a++)
    CU(cudaFree(kept_device[a]));
  /* Host mappings are persistent in this runtime; process exit releases them.
   */
  polygeist_require_gpu_libraries_end();
  printf("TOTAL checked=%zu failures=%d\n", checked, failures);
  return failures ? 3 : 0;
}
