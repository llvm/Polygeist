#include <stdint.h>
#include <stdio.h>
#include <string.h>

void pva_fixture_conv3x3_i8(int32_t, int32_t, const int8_t *, int8_t *,
                            const uint8_t *);
void pva_fixture_box_filter_3x3_i8(int32_t, int32_t, const int8_t *, int8_t *);
void pva_fixture_gaussian_filter_3x3_u8(int32_t, int32_t, const uint8_t *,
                                        uint8_t *, float);
void pva_fixture_bilateral_filter_3x3_u8(int32_t, int32_t, const uint8_t *,
                                         uint8_t *, float, float);
void pva_fixture_histogram_equalization_u8(int32_t, int32_t,
                                            const uint8_t *, uint8_t *);

enum { H = 8, W = 8, N = H * W };

static int check_i8_interior(const char *name, const int8_t *values,
                             int8_t expected) {
  for (int y = 1; y < H - 1; ++y)
    for (int x = 1; x < W - 1; ++x)
      if (values[y * W + x] != expected) {
        fprintf(stderr, "%s: mismatch at (%d,%d): got %d expected %d\n",
                name, y, x, values[y * W + x], expected);
        return 1;
      }
  return 0;
}

static int check_u8(const char *name, const uint8_t *values, uint8_t expected,
                    int interior_only) {
  const int begin = interior_only ? 1 : 0;
  const int end_y = interior_only ? H - 1 : H;
  const int end_x = interior_only ? W - 1 : W;
  for (int y = begin; y < end_y; ++y)
    for (int x = begin; x < end_x; ++x)
      if (values[y * W + x] != expected) {
        fprintf(stderr, "%s: mismatch at (%d,%d): got %u expected %u\n",
                name, y, x, values[y * W + x], expected);
        return 1;
      }
  return 0;
}

int main(void) {
  int failures = 0;
  int8_t signed_in[N], signed_out[N];
  uint8_t unsigned_in[N], unsigned_out[N];
  uint8_t coeff[9] = {0, 0, 0, 0, 128, 0, 0, 0, 0};

  memset(signed_in, 10, sizeof(signed_in));
  memset(signed_out, 0, sizeof(signed_out));
  pva_fixture_conv3x3_i8(H, W, signed_in, signed_out, coeff);
  failures += check_i8_interior("conv3x3_i8", signed_out, 5);

  memset(signed_in, 9, sizeof(signed_in));
  memset(signed_out, 0, sizeof(signed_out));
  pva_fixture_box_filter_3x3_i8(H, W, signed_in, signed_out);
  failures += check_i8_interior("box_filter_3x3_i8", signed_out, 9);

  // Exercise the upper half of U8 so signedness regressions are observable.
  memset(unsigned_in, 200, sizeof(unsigned_in));
  memset(unsigned_out, 0, sizeof(unsigned_out));
  pva_fixture_gaussian_filter_3x3_u8(H, W, unsigned_in, unsigned_out, 1.0f);
  failures += check_u8("gaussian_filter_3x3_u8", unsigned_out, 200, 1);

  memset(unsigned_in, 200, sizeof(unsigned_in));
  memset(unsigned_out, 0, sizeof(unsigned_out));
  pva_fixture_bilateral_filter_3x3_u8(H, W, unsigned_in, unsigned_out, 25.0f,
                                      10.0f);
  failures += check_u8("bilateral_filter_3x3_u8", unsigned_out, 200, 1);

  memset(unsigned_in, 17, sizeof(unsigned_in));
  memset(unsigned_out, 255, sizeof(unsigned_out));
  pva_fixture_histogram_equalization_u8(H, W, unsigned_in, unsigned_out);
  failures += check_u8("histogram_equalization_u8", unsigned_out, 0, 0);

  if (failures) return 1;
  puts("PASS: 5/5 independently authored PVA semantic fixtures");
  return 0;
}
