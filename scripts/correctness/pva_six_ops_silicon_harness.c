/* Independent numerical validation for the six typed PVA image adapters. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H 192
#define W 256
#define PIXELS ((size_t)H * W)

#define DECL_FILTER(op, suffix, type)                                        \
  void polygeist_pva_##op##_3x3_##suffix(                                   \
      int32_t, int32_t, const type *, type *)
DECL_FILTER(boxfilter, u8, uint8_t);
DECL_FILTER(boxfilter, s8, int8_t);
DECL_FILTER(boxfilter, u16, uint16_t);
DECL_FILTER(boxfilter, s16, int16_t);
DECL_FILTER(morphology_dilate, u8, uint8_t);
DECL_FILTER(morphology_dilate, s8, int8_t);
DECL_FILTER(morphology_dilate, u16, uint16_t);
DECL_FILTER(morphology_dilate, s16, int16_t);
#define DECL_GAUSSIAN(suffix, type)                                         \
  void polygeist_pva_gaussian_3x3_##suffix(                                \
      int32_t, int32_t, float, float, const type *, type *)
DECL_GAUSSIAN(u8, uint8_t);
DECL_GAUSSIAN(s8, int8_t);
DECL_GAUSSIAN(u16, uint16_t);
DECL_GAUSSIAN(s16, int16_t);
#undef DECL_GAUSSIAN
void polygeist_pva_bilateral_3x3_u8(int32_t, int32_t, float, float,
                                     const uint8_t *, uint8_t *);
void polygeist_pva_histogram_256_u8_u32(int32_t, int32_t, const uint8_t *,
                                        uint32_t *);
void polygeist_pva_histogram_256_u8_s32(int32_t, int32_t, const uint8_t *,
                                        int32_t *);
void polygeist_pva_histogram_256_u16_u32(int32_t, int32_t, const uint16_t *,
                                         uint32_t *);
void polygeist_pva_histogram_256_u16_s32(int32_t, int32_t, const uint16_t *,
                                         int32_t *);
void polygeist_pva_histeq_u8(int32_t, int32_t, const uint8_t *, uint8_t *);

typedef enum { U8, S8, U16, S16 } Kind;

static uint32_t rng_state = 0x12345678u;
static uint32_t random_u32(void) {
  uint32_t x = rng_state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  rng_state = x;
  return x;
}

static int64_t get_value(const void *data, Kind kind, size_t index) {
  switch (kind) {
  case U8: return ((const uint8_t *)data)[index];
  case S8: return ((const int8_t *)data)[index];
  case U16: return ((const uint16_t *)data)[index];
  case S16: return ((const int16_t *)data)[index];
  }
  return 0;
}

static int64_t minimum(Kind kind) {
  return kind == S8 ? -128 : kind == S16 ? -32768 : 0;
}

static int64_t maximum(Kind kind) {
  return kind == U8 ? 255 : kind == S8 ? 127 :
         kind == U16 ? 65535 : 32767;
}

static void set_value(void *data, Kind kind, size_t index, int64_t value) {
  if (value < minimum(kind)) value = minimum(kind);
  if (value > maximum(kind)) value = maximum(kind);
  switch (kind) {
  case U8: ((uint8_t *)data)[index] = (uint8_t)value; break;
  case S8: ((int8_t *)data)[index] = (int8_t)value; break;
  case U16: ((uint16_t *)data)[index] = (uint16_t)value; break;
  case S16: ((int16_t *)data)[index] = (int16_t)value; break;
  }
}

static size_t kind_bytes(Kind kind) {
  return kind == U8 || kind == S8 ? 1 : 2;
}

static void initialize_input(void *data, Kind kind) {
  for (size_t i = 0; i < PIXELS; ++i) {
    uint32_t bits = random_u32();
    switch (kind) {
    case U8: set_value(data, kind, i, bits & 255); break;
    case S8: set_value(data, kind, i, (int)(bits & 255) - 128); break;
    case U16: set_value(data, kind, i, bits & 65535); break;
    case S16: set_value(data, kind, i, (int)(bits & 65535) - 32768); break;
    }
  }
}

static void reference_filter(const char *operation, Kind kind,
                             const void *input, void *reference) {
  int64_t weights[9] = {0};
  int qbits = kind_bytes(kind) == 1 ? 8 : 16;
  int64_t scale = (int64_t)1 << qbits;
  if (!strcmp(operation, "box")) {
    int64_t sum = 0;
    for (int i = 0; i < 9; ++i) {
      weights[i] = (int64_t)((1.0 / 9.0) * scale);
      sum += weights[i];
    }
    int64_t difference = scale - sum;
    int step = 9 / (int)llabs(difference);
    if (!step) step = 1;
    for (int i = 0; i < 9 && difference; i += step, --difference)
      ++weights[i];
  } else if (!strcmp(operation, "gaussian")) {
    double axis[3], axis_sum = 0.0;
    for (int i = -1; i <= 1; ++i) {
      axis[i + 1] = exp(-0.5 * i * i) / sqrt(2.0 * M_PI);
      axis_sum += axis[i + 1];
    }
    for (int i = 0; i < 3; ++i) axis[i] /= axis_sum;
    int64_t sum = 0;
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 3; ++x) {
        int i = y * 3 + x;
        weights[i] = (int64_t)(axis[y] * axis[x] * scale);
        sum += weights[i];
      }
    weights[4] += scale - sum;
  }
  memcpy(reference, input, PIXELS * kind_bytes(kind));
  for (int y = 1; y < H - 1; ++y) {
    for (int x = 1; x < W - 1; ++x) {
      int64_t value = !strcmp(operation, "morphology") ? minimum(kind) : 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          int64_t sample = get_value(input, kind,
              (size_t)(y + dy) * W + x + dx);
          if (!strcmp(operation, "morphology")) {
            if (sample > value) value = sample;
          } else {
            value += sample * weights[(dy + 1) * 3 + dx + 1];
          }
        }
      }
      if (strcmp(operation, "morphology"))
        value = (value + (scale >> 1)) >> qbits;
      set_value(reference, kind, (size_t)y * W + x, value);
    }
  }
}

static void report_integer_image(const char *operation, const char *dtype,
                                 Kind kind, const void *actual,
                                 const void *reference, int tolerance) {
  size_t exact_mismatches = 0, tolerance_mismatches = 0;
  int64_t max_error = 0;
  for (int y = 1; y < H - 1; ++y) {
    for (int x = 1; x < W - 1; ++x) {
      size_t i = (size_t)y * W + x;
      int64_t error = llabs(get_value(actual, kind, i) -
                            get_value(reference, kind, i));
      if (error) ++exact_mismatches;
      if (error > tolerance) ++tolerance_mismatches;
      if (error > max_error) max_error = error;
    }
  }
  const char *status = exact_mismatches == 0 ? "EXACT_PASS" :
                       (tolerance_mismatches == 0 ? "TOLERANCE_PASS" : "FAIL");
  printf("RESULT,%s,%s,%s,exact_mismatches=%zu,tolerance_mismatches=%zu,"
         "tolerance=%d,max_error=%lld\n", operation, dtype,
         status, exact_mismatches,
         tolerance_mismatches, tolerance, (long long)max_error);
}

#define RUN_FILTER(operation_name, function_name, suffix, ctype, kind, tol)  \
  do {                                                                        \
    ctype *input = malloc(PIXELS * sizeof(ctype));                            \
    ctype *actual = malloc(PIXELS * sizeof(ctype));                           \
    ctype *reference = malloc(PIXELS * sizeof(ctype));                        \
    rng_state = 0x12345678u + (uint32_t)(kind);                               \
    initialize_input(input, kind);                                            \
    memset(actual, 0x5a, PIXELS * sizeof(ctype));                             \
    function_name(H, W, input, actual);                                       \
    reference_filter(operation_name, kind, input, reference);                 \
    report_integer_image(operation_name, #suffix, kind, actual, reference,    \
                         tol);                                                 \
    free(reference); free(actual); free(input);                               \
  } while (0)

#define RUN_GAUSSIAN(function_name, suffix, ctype, kind, tol)                \
  do {                                                                        \
    ctype *input = malloc(PIXELS * sizeof(ctype));                            \
    ctype *actual = malloc(PIXELS * sizeof(ctype));                           \
    ctype *reference = malloc(PIXELS * sizeof(ctype));                        \
    rng_state = 0x12345678u + (uint32_t)(kind);                               \
    initialize_input(input, kind);                                            \
    memset(actual, 0x5a, PIXELS * sizeof(ctype));                             \
    function_name(H, W, 1.0f, 1.0f, input, actual);                          \
    reference_filter("gaussian", kind, input, reference);                    \
    report_integer_image("gaussian", #suffix, kind, actual, reference, tol); \
    free(reference); free(actual); free(input);                               \
  } while (0)

static void run_bilateral(void) {
  uint8_t *input = malloc(PIXELS), *actual = malloc(PIXELS);
  uint8_t *reference = malloc(PIXELS);
  const float sigma_range = 25.0f, sigma_space = 10.0f;
  rng_state = 0xabcdef01u;
  initialize_input(input, U8);
  memset(actual, 0x5a, PIXELS);
  memcpy(reference, input, PIXELS);
  for (int y = 1; y < H - 1; ++y) {
    for (int x = 1; x < W - 1; ++x) {
      float center = input[(size_t)y * W + x];
      uint32_t weighted = 0, weights = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx && dy) continue;
          float sample = input[(size_t)(y + dy) * W + x + dx];
          double spatial = (double)(dx * dx + dy * dy);
          double range = (double)(sample - center) * (sample - center);
          double weight_float =
              exp(-0.5 * spatial / (sigma_space * sigma_space) -
                  0.5 * range / (sigma_range * sigma_range));
          int weight = (int)(weight_float * 128.0 + 0.5);
          if (weight > 255) weight = 255;
          weighted += (uint32_t)weight * (uint32_t)sample;
          weights += (uint32_t)weight;
        }
      }
      reference[(size_t)y * W + x] =
          (uint8_t)((float)weighted * (1.0f / (float)weights));
    }
  }
  polygeist_pva_bilateral_3x3_u8(H, W, sigma_range, sigma_space,
                                  input, actual);
  report_integer_image("bilateral", "u8", U8, actual, reference, 2);
  free(reference); free(actual); free(input);
}

static void run_histograms(void) {
  uint8_t *u8 = malloc(PIXELS);
  uint16_t *u16 = malloc(PIXELS * sizeof(uint16_t));
  uint32_t reference8[256] = {0}, reference16[256] = {0};
  uint32_t u32[256];
  int32_t s32[256];
  rng_state = 0x13579bdfu; initialize_input(u8, U8);
  rng_state = 0x2468ace0u; initialize_input(u16, U16);
  for (size_t i = 0; i < PIXELS; ++i) {
    ++reference8[u8[i]];
    ++reference16[u16[i] >> 8];
  }
#define CHECK_HIST(label, call, actual, reference)                            \
  do {                                                                        \
    memset(actual, 0, sizeof(actual));                                        \
    call;                                                                     \
    size_t bad_ = 0; uint32_t max_ = 0;                                      \
    for (int i_ = 0; i_ < 256; ++i_) {                                      \
      uint32_t got_ = (uint32_t)(actual)[i_];                                \
      uint32_t err_ = got_ > (reference)[i_] ?                              \
          got_ - (reference)[i_] : (reference)[i_] - got_;                  \
      if (err_) ++bad_;                                                       \
      if (err_ > max_) max_ = err_;                                          \
    }                                                                         \
    printf("RESULT,histogram,%s,%s,mismatches=%zu,max_error=%u\n", label,   \
           bad_ ? "FAIL" : "PASS", bad_, max_);                            \
  } while (0)
  CHECK_HIST("u8_u32", polygeist_pva_histogram_256_u8_u32(H,W,u8,u32),
             u32, reference8);
  CHECK_HIST("u8_s32", polygeist_pva_histogram_256_u8_s32(H,W,u8,s32),
             s32, reference8);
  CHECK_HIST("u16_u32", polygeist_pva_histogram_256_u16_u32(H,W,u16,u32),
             u32, reference16);
  CHECK_HIST("u16_s32", polygeist_pva_histogram_256_u16_s32(H,W,u16,s32),
             s32, reference16);
#undef CHECK_HIST
  free(u16); free(u8);
}

static void run_histeq(void) {
  uint8_t *input = malloc(PIXELS), *actual = malloc(PIXELS);
  uint8_t *reference = malloc(PIXELS);
  uint32_t histogram[256] = {0}, cdf[256], cdf_min = 0;
  rng_state = 0x10203040u; initialize_input(input, U8);
  for (size_t i = 0; i < PIXELS; ++i) ++histogram[input[i]];
  cdf[0] = histogram[0];
  for (int i = 1; i < 256; ++i) cdf[i] = cdf[i - 1] + histogram[i];
  for (int i = 0; i < 256; ++i) if (cdf[i]) { cdf_min = cdf[i]; break; }
  uint32_t denominator = (uint32_t)PIXELS - cdf_min;
  if (!denominator) denominator = 1;
  for (size_t i = 0; i < PIXELS; ++i) {
    int64_t fixed_scale = ((int64_t)255 << 15) / denominator;
    int64_t value = (((int64_t)cdf[input[i]] - cdf_min) * fixed_scale) >> 15;
    if (value < 0) value = 0;
    if (value > 255) value = 255;
    reference[i] = (uint8_t)value;
  }
  polygeist_pva_histeq_u8(H, W, input, actual);
  size_t mismatches = 0; int max_error = 0;
  for (size_t i = 0; i < PIXELS; ++i) {
    int error = abs((int)actual[i] - reference[i]);
    if (error) ++mismatches;
    if (error > max_error) max_error = error;
  }
  printf("RESULT,histogram_equalization,u8,%s,mismatches=%zu,max_error=%d\n",
         mismatches ? "FAIL" : "PASS", mismatches, max_error);
  free(reference); free(actual); free(input);
}

int main(void) {
  RUN_FILTER("box", polygeist_pva_boxfilter_3x3_u8, u8, uint8_t, U8, 0);
  RUN_FILTER("box", polygeist_pva_boxfilter_3x3_s8, s8, int8_t, S8, 0);
  RUN_FILTER("box", polygeist_pva_boxfilter_3x3_u16, u16, uint16_t, U16, 0);
  RUN_FILTER("box", polygeist_pva_boxfilter_3x3_s16, s16, int16_t, S16, 0);
  RUN_GAUSSIAN(polygeist_pva_gaussian_3x3_u8, u8, uint8_t, U8, 1);
  RUN_GAUSSIAN(polygeist_pva_gaussian_3x3_s8, s8, int8_t, S8, 1);
  RUN_GAUSSIAN(polygeist_pva_gaussian_3x3_u16, u16, uint16_t, U16, 4);
  RUN_GAUSSIAN(polygeist_pva_gaussian_3x3_s16, s16, int16_t, S16, 4);
  RUN_FILTER("morphology", polygeist_pva_morphology_dilate_3x3_u8,
             u8, uint8_t, U8, 0);
  RUN_FILTER("morphology", polygeist_pva_morphology_dilate_3x3_s8,
             s8, int8_t, S8, 0);
  RUN_FILTER("morphology", polygeist_pva_morphology_dilate_3x3_u16,
             u16, uint16_t, U16, 0);
  RUN_FILTER("morphology", polygeist_pva_morphology_dilate_3x3_s16,
             s16, int16_t, S16, 0);
  run_bilateral();
  run_histograms();
  run_histeq();
  return 0;
}
