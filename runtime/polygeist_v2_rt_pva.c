/* Typed host-buffer wrappers forwarding to the PVA runtime. Generated bodies
 * only validate and marshal parameters; vendor execution is in the SDK shim. */
#define _POSIX_C_SOURCE 199309L
#include "polygeist_v2_rt.h"
#include "polygeist_cublas_rt.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- call accounting (POLYGEIST_V2_TRACE) ------------------------------- */

#define POLYGEIST_V2_MAX_CALLS 64

typedef struct {
  const char *call;
  const char *entry;
  uint64_t count;
  double ms;
} polygeist_v2_stat_t;

static polygeist_v2_stat_t polygeist_v2_stats[POLYGEIST_V2_MAX_CALLS];
static int polygeist_v2_nstats;
static int polygeist_v2_trace = -1;

static void polygeist_v2_report(void) {
  for (int i = 0; i < polygeist_v2_nstats; ++i)
    fprintf(stderr,
            "POLYGEIST_V2_STATS backend=pva call=%s entry=%s count=%llu "
            "lib_ms=%.6f\n",
            polygeist_v2_stats[i].call, polygeist_v2_stats[i].entry,
            (unsigned long long)polygeist_v2_stats[i].count,
            polygeist_v2_stats[i].ms);
}

static int polygeist_v2_tracing(void) {
  if (polygeist_v2_trace < 0) {
    const char *e = getenv("POLYGEIST_V2_TRACE");
    polygeist_v2_trace = e && *e && strcmp(e, "0") != 0;
    if (polygeist_v2_trace)
      atexit(polygeist_v2_report);
  }
  return polygeist_v2_trace;
}

static polygeist_v2_stat_t *polygeist_v2_stat(const char *call,
                                              const char *entry) {
  for (int i = 0; i < polygeist_v2_nstats; ++i)
    if (polygeist_v2_stats[i].call == call ||
        strcmp(polygeist_v2_stats[i].call, call) == 0)
      return &polygeist_v2_stats[i];
  if (polygeist_v2_nstats == POLYGEIST_V2_MAX_CALLS) {
    fprintf(stderr, "polygeist_v2: too many distinct calls\n");
    abort();
  }
  polygeist_v2_stat_t *s = &polygeist_v2_stats[polygeist_v2_nstats++];
  s->call = call;
  s->entry = entry;
  return s;
}

uint64_t polygeist_v2_call_count(const char *call) {
  for (int i = 0; i < polygeist_v2_nstats; ++i)
    if (strcmp(polygeist_v2_stats[i].call, call) == 0)
      return polygeist_v2_stats[i].count;
  return 0;
}

static double polygeist_v2_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static void polygeist_v2_done(const char *call, const char *entry,
                              double t0) {
  double dt = polygeist_v2_now_ms() - t0;
  polygeist_v2_stat_t *s = polygeist_v2_stat(call, entry);
  s->count++;
  s->ms += dt;
  if (polygeist_v2_tracing())
    fprintf(stderr, "POLYGEIST_V2_CALL backend=pva call=%s entry=%s ms=%.6f\n",
            call, entry, dt);
}

static void polygeist_v2_fatal(const char *call, const char *msg) {
  fprintf(stderr, "polygeist_v2 %s: %s\n", call, msg);
  abort();
}

static int polygeist_v2_overlaps(const void *a, int64_t an,
                                 const void *b, int64_t bn) {
  uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  return x <= y ? y - x < (uint64_t)an : x - y < (uint64_t)bn;
}

/* BEGIN GENERATED WRAPPERS */
void polygeist_v2_pvaMorphologySubmit_opencv_morphology_dilate_s16(int32_t h, int32_t w, const short *input, int64_t input_bytes, int32_t input_stride, short *output, int64_t output_bytes, int32_t output_stride) {
  if (h < 3 || h > 46340)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s16", "integer outside wrapper bounds");
  if (w < 3 || w > 46340)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s16", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(short);
  if ((int64_t)h <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)h)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s16", "array extent overflow");
  pv2_bytes_input *= (int64_t)h;
  if ((int64_t)w <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)w)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s16", "array extent overflow");
  pv2_bytes_input *= (int64_t)w;
  if (input_stride != (int64_t)w || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s16", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(short);
  if (((int64_t)h + (-1)) <= 0 || pv2_bytes_output > INT64_MAX / ((int64_t)h + (-1)))
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s16", "array extent overflow");
  pv2_bytes_output *= ((int64_t)h + (-1));
  if ((int64_t)w <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)w)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s16", "array extent overflow");
  pv2_bytes_output *= (int64_t)w;
  if (output_stride != (int64_t)w || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s16", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s16", "unsupported output overlap");
  double pv2_t0 = polygeist_v2_now_ms();
  polygeist_pva_morphology_dilate_3x3_s16(h, w, input, output);
  polygeist_v2_done("pvaMorphologySubmit_opencv_morphology_dilate_s16", "polygeist_pva_morphology_dilate_3x3_s16", pv2_t0);
}

void polygeist_v2_pvaMorphologySubmit_opencv_morphology_dilate_s8(int32_t h, int32_t w, const signed char *input, int64_t input_bytes, int32_t input_stride, signed char *output, int64_t output_bytes, int32_t output_stride) {
  if (h < 3 || h > 46340)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s8", "integer outside wrapper bounds");
  if (w < 3 || w > 46340)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s8", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(signed char);
  if ((int64_t)h <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)h)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s8", "array extent overflow");
  pv2_bytes_input *= (int64_t)h;
  if ((int64_t)w <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)w)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s8", "array extent overflow");
  pv2_bytes_input *= (int64_t)w;
  if (input_stride != (int64_t)w || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s8", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(signed char);
  if (((int64_t)h + (-1)) <= 0 || pv2_bytes_output > INT64_MAX / ((int64_t)h + (-1)))
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s8", "array extent overflow");
  pv2_bytes_output *= ((int64_t)h + (-1));
  if ((int64_t)w <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)w)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s8", "array extent overflow");
  pv2_bytes_output *= (int64_t)w;
  if (output_stride != (int64_t)w || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s8", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_s8", "unsupported output overlap");
  double pv2_t0 = polygeist_v2_now_ms();
  polygeist_pva_morphology_dilate_3x3_s8(h, w, input, output);
  polygeist_v2_done("pvaMorphologySubmit_opencv_morphology_dilate_s8", "polygeist_pva_morphology_dilate_3x3_s8", pv2_t0);
}

void polygeist_v2_pvaMorphologySubmit_opencv_morphology_dilate_u16(int32_t h, int32_t w, const unsigned short *input, int64_t input_bytes, int32_t input_stride, unsigned short *output, int64_t output_bytes, int32_t output_stride) {
  if (h < 3 || h > 46340)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u16", "integer outside wrapper bounds");
  if (w < 3 || w > 46340)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u16", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(unsigned short);
  if ((int64_t)h <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)h)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u16", "array extent overflow");
  pv2_bytes_input *= (int64_t)h;
  if ((int64_t)w <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)w)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u16", "array extent overflow");
  pv2_bytes_input *= (int64_t)w;
  if (input_stride != (int64_t)w || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u16", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(unsigned short);
  if (((int64_t)h + (-1)) <= 0 || pv2_bytes_output > INT64_MAX / ((int64_t)h + (-1)))
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u16", "array extent overflow");
  pv2_bytes_output *= ((int64_t)h + (-1));
  if ((int64_t)w <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)w)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u16", "array extent overflow");
  pv2_bytes_output *= (int64_t)w;
  if (output_stride != (int64_t)w || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u16", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u16", "unsupported output overlap");
  double pv2_t0 = polygeist_v2_now_ms();
  polygeist_pva_morphology_dilate_3x3_u16(h, w, input, output);
  polygeist_v2_done("pvaMorphologySubmit_opencv_morphology_dilate_u16", "polygeist_pva_morphology_dilate_3x3_u16", pv2_t0);
}

void polygeist_v2_pvaMorphologySubmit_opencv_morphology_dilate_u8(int32_t h, int32_t w, const unsigned char *input, int64_t input_bytes, int32_t input_stride, unsigned char *output, int64_t output_bytes, int32_t output_stride) {
  if (h < 3 || h > 46340)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u8", "integer outside wrapper bounds");
  if (w < 3 || w > 46340)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u8", "integer outside wrapper bounds");
  int64_t pv2_bytes_input = sizeof(unsigned char);
  if ((int64_t)h <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)h)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u8", "array extent overflow");
  pv2_bytes_input *= (int64_t)h;
  if ((int64_t)w <= 0 || pv2_bytes_input > INT64_MAX / (int64_t)w)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u8", "array extent overflow");
  pv2_bytes_input *= (int64_t)w;
  if (input_stride != (int64_t)w || !input || input_bytes < pv2_bytes_input)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u8", "array layout or capacity mismatch");
  int64_t pv2_bytes_output = sizeof(unsigned char);
  if (((int64_t)h + (-1)) <= 0 || pv2_bytes_output > INT64_MAX / ((int64_t)h + (-1)))
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u8", "array extent overflow");
  pv2_bytes_output *= ((int64_t)h + (-1));
  if ((int64_t)w <= 0 || pv2_bytes_output > INT64_MAX / (int64_t)w)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u8", "array extent overflow");
  pv2_bytes_output *= (int64_t)w;
  if (output_stride != (int64_t)w || !output || output_bytes < pv2_bytes_output)
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u8", "array layout or capacity mismatch");
  if (polygeist_v2_overlaps(input, input_bytes, output, output_bytes))
    polygeist_v2_fatal("pvaMorphologySubmit_opencv_morphology_dilate_u8", "unsupported output overlap");
  double pv2_t0 = polygeist_v2_now_ms();
  polygeist_pva_morphology_dilate_3x3_u8(h, w, input, output);
  polygeist_v2_done("pvaMorphologySubmit_opencv_morphology_dilate_u8", "polygeist_pva_morphology_dilate_3x3_u8", pv2_t0);
}

/* END GENERATED WRAPPERS */
