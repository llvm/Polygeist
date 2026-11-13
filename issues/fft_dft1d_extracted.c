#include <math.h>

#define FFT_N 4
#define FFT_TWOPI 6.28318530717958647692528676655900576

void fft_dft1d_z2z_forward(const double in[FFT_N][2],
                           double out[FFT_N][2]) {
  for (int k = 0; k < FFT_N; ++k) {
    double sum_re = 0.0;
    double sum_im = 0.0;
    for (int n = 0; n < FFT_N; ++n) {
      double angle = -FFT_TWOPI * (double)k * (double)n / (double)FFT_N;
      double c = cos(angle);
      double s = sin(angle);
      double ar = in[n][0];
      double ai = in[n][1];
      sum_re += ar * c - ai * s;
      sum_im += ar * s + ai * c;
    }
    out[k][0] = sum_re;
    out[k][1] = sum_im;
  }
}

void fft_dft1d_z2z_inverse(const double in[FFT_N][2],
                           double out[FFT_N][2]) {
  for (int k = 0; k < FFT_N; ++k) {
    double sum_re = 0.0;
    double sum_im = 0.0;
    for (int n = 0; n < FFT_N; ++n) {
      double angle = FFT_TWOPI * (double)k * (double)n / (double)FFT_N;
      double c = cos(angle);
      double s = sin(angle);
      double ar = in[n][0];
      double ai = in[n][1];
      sum_re += ar * c - ai * s;
      sum_im += ar * s + ai * c;
    }
    out[k][0] = sum_re;
    out[k][1] = sum_im;
  }
}

void fft_dft1d_z2z_forward_inplace_accum(const double in[FFT_N][2],
                                         double out[FFT_N][2]) {
  for (int k = 0; k < FFT_N; ++k) {
    out[k][0] = 0.0;
    out[k][1] = 0.0;
  }
  for (int k = 0; k < FFT_N; ++k) {
    for (int n = 0; n < FFT_N; ++n) {
      double angle = -FFT_TWOPI * (double)k * (double)n / (double)FFT_N;
      double c = cos(angle);
      double s = sin(angle);
      double ar = in[n][0];
      double ai = in[n][1];
      out[k][0] += ar * c - ai * s;
      out[k][1] += ar * s + ai * c;
    }
  }
}

void fft_dft1d_z2z_forward_interleaved(const double in[FFT_N][2],
                                       double out[FFT_N][2]) {
  for (int k = 0; k < FFT_N; ++k)
    for (int component = 0; component < 2; ++component)
      out[k][component] = 0.0;

  for (int k = 0; k < FFT_N; ++k) {
    for (int component = 0; component < 2; ++component) {
      for (int n = 0; n < FFT_N; ++n) {
        double angle = -FFT_TWOPI * (double)k * (double)n / (double)FFT_N;
        double c = cos(angle);
        double s = sin(angle);
        double ar = in[n][0];
        double ai = in[n][1];
        double value = component == 0 ? (ar * c - ai * s)
                                      : (ar * s + ai * c);
        out[k][component] += value;
      }
    }
  }
}
