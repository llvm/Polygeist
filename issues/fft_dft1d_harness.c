#include <math.h>
#include <stdio.h>

#define FFT_N 4

void fft_dft1d_z2z_forward_interleaved(const double in[FFT_N][2],
                                       double out[FFT_N][2]);

int main(void) {
  const double in[FFT_N][2] = {
      {1.0, 0.0},
      {2.0, 0.0},
      {0.0, 0.0},
      {-1.0, 0.0},
  };
  const double expected[FFT_N][2] = {
      {2.0, 0.0},
      {1.0, -3.0},
      {0.0, 0.0},
      {1.0, 3.0},
  };
  double out[FFT_N][2] = {{0.0, 0.0}};
  fft_dft1d_z2z_forward_interleaved(in, out);
  for (int i = 0; i < FFT_N; ++i) {
    for (int c = 0; c < 2; ++c) {
      double diff = fabs(out[i][c] - expected[i][c]);
      if (diff > 1.0e-9) {
        fprintf(stderr,
                "fft mismatch at (%d,%d): got %.17g expected %.17g\n",
                i, c, out[i][c], expected[i][c]);
        return 1;
      }
    }
  }
  printf("fft_dft1d_interleaved ok\n");
  return 0;
}
