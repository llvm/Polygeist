// Standalone kernelized extracts from the C proxy apps we are evaluating.
// The goal is to preserve loop/dataflow shapes while removing app ABI,
// MPI, BML, and solver-struct setup.

#include <stddef.h>

#define PK_ABS(x) ((x) < 0.0 ? -(x) : (x))
#define PK_MAX(a, b) ((a) > (b) ? (a) : (b))
#define PK_MIN(a, b) ((a) < (b) ? (a) : (b))

#define MX 12
#define MY 10
#define MZ 8
#define MMAT 4

void miniamr_stencil_calc_7(double out[MX][MY][MZ],
                            const double in[MX + 2][MY + 2][MZ + 2]) {
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++)
        out[i][j][k] = (in[i][j + 1][k + 1] + in[i + 1][j][k + 1] +
                        in[i + 1][j + 1][k] + in[i + 1][j + 1][k + 1] +
                        in[i + 1][j + 1][k + 2] + in[i + 1][j + 2][k + 1] +
                        in[i + 2][j + 1][k + 1]) /
                       7.0;
}

void miniamr_stencil_calc_27(double out[MX][MY][MZ],
                             const double in[MX + 2][MY + 2][MZ + 2]) {
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double sum = 0.0;
        for (int di = 0; di < 3; di++)
          for (int dj = 0; dj < 3; dj++)
            for (int dk = 0; dk < 3; dk++)
              sum += in[i + di][j + dj][k + dk];
        out[i][j][k] = sum / 27.0;
      }
}

void miniamr_stencil_0_coupled_sum(double out[MX][MY][MZ],
                                   const double material[MMAT][MX][MY][MZ],
                                   const double base[MX][MY][MZ]) {
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double acc = out[i][j][k];
        for (int v = 0; v < MMAT; v++)
          acc += material[v][i][j][k] * base[i][j][k];
        out[i][j][k] = acc;
      }
}

void miniamr_stencil_0_pointwise_update(double out[MX][MY][MZ],
                                        const double a[MX][MY][MZ],
                                        const double b[MX][MY][MZ],
                                        double a1) {
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++)
        out[i][j][k] += out[i][j][k] *
                        (a[i][j][k] + b[i][j][k] - a1 * out[i][j][k]);
}

void miniamr_stencil_x_directional(double out[MX][MY][MZ],
                                   const double in[MX + 2][MY][MZ],
                                   double a0, double a1) {
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double left = in[i][j][k];
        double center = in[i + 1][j][k];
        double right = in[i + 2][j][k];
        out[i][j][k] = center + a0 * (left - 2.0 * center + right) +
                       a1 * (right - left);
      }
}

void miniamr_stencil_y_directional(double out[MX][MY][MZ],
                                   const double in[MX][MY + 2][MZ],
                                   double a0, double a1) {
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double left = in[i][j][k];
        double center = in[i][j + 1][k];
        double right = in[i][j + 2][k];
        out[i][j][k] = center + a0 * (left - 2.0 * center + right) +
                       a1 * (right - left);
      }
}

void miniamr_stencil_z_directional(double out[MX][MY][MZ],
                                   const double in[MX][MY][MZ + 2],
                                   double a0, double a1) {
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double left = in[i][j][k];
        double center = in[i][j][k + 1];
        double right = in[i][j][k + 2];
        out[i][j][k] = center + a0 * (left - 2.0 * center + right) +
                       a1 * (right - left);
      }
}

void miniamr_stencil_7_weighted(double out[MX][MY][MZ],
                                const double in[MX + 2][MY + 2][MZ + 2],
                                const double coeff[MX][MY][MZ]) {
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double center = in[i + 1][j + 1][k + 1];
        double lap = in[i][j + 1][k + 1] + in[i + 2][j + 1][k + 1] +
                     in[i + 1][j][k + 1] + in[i + 1][j + 2][k + 1] +
                     in[i + 1][j + 1][k] + in[i + 1][j + 1][k + 2] -
                     6.0 * center;
        out[i][j][k] = center + coeff[i][j][k] * lap;
      }
}

void miniamr_stencil_27_weighted(double out[MX][MY][MZ],
                                 const double in[MX + 2][MY + 2][MZ + 2],
                                 const double coeff[3][3][3]) {
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double sum = 0.0;
        for (int di = 0; di < 3; di++)
          for (int dj = 0; dj < 3; dj++)
            for (int dk = 0; dk < 3; dk++)
              sum += coeff[di][dj][dk] * in[i + di][j + dj][k + dk];
        out[i][j][k] = sum;
      }
}

void miniamr_pack_face_x(double buf[MY][MZ],
                         const double grid[MX + 2][MY + 2][MZ + 2]) {
  for (int j = 0; j < MY; j++)
    for (int k = 0; k < MZ; k++)
      buf[j][k] = grid[1][j + 1][k + 1];
}

void miniamr_unpack_face_x(double grid[MX + 2][MY + 2][MZ + 2],
                           const double buf[MY][MZ]) {
  for (int j = 0; j < MY; j++)
    for (int k = 0; k < MZ; k++)
      grid[0][j + 1][k + 1] = buf[j][k];
}

void miniamr_pack_block(double buf[MX * MY * MZ],
                        const double grid[MX][MY][MZ]) {
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++)
        buf[(i * MY + j) * MZ + k] = grid[i][j][k];
}

void miniamr_unpack_block(double grid[MX][MY][MZ],
                          const double buf[MX * MY * MZ]) {
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++)
        grid[i][j][k] = buf[(i * MY + j) * MZ + k];
}

#define HX 12
#define HY 10
#define HZ 8
#define HV 960

void hpgmg_apply_op_7pt(double Ax[HX][HY][HZ],
                        const double x[HX + 2][HY + 2][HZ + 2], double a,
                        double b) {
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++) {
        double center = x[i + 1][j + 1][k + 1];
        double lap = 6.0 * center - x[i][j + 1][k + 1] -
                     x[i + 2][j + 1][k + 1] - x[i + 1][j][k + 1] -
                     x[i + 1][j + 2][k + 1] - x[i + 1][j + 1][k] -
                     x[i + 1][j + 1][k + 2];
        Ax[i][j][k] = a * center + b * lap;
      }
}

void hpgmg_apply_op_27pt(double Ax[HX][HY][HZ],
                         const double x[HX + 2][HY + 2][HZ + 2],
                         const double coeff[3][3][3], double a) {
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++) {
        double sum = a * x[i + 1][j + 1][k + 1];
        for (int di = 0; di < 3; di++)
          for (int dj = 0; dj < 3; dj++)
            for (int dk = 0; dk < 3; dk++)
              sum += coeff[di][dj][dk] * x[i + di][j + dj][k + dk];
        Ax[i][j][k] = sum;
      }
}

void hpgmg_residual_7pt(double res[HX][HY][HZ],
                        const double rhs[HX][HY][HZ],
                        const double x[HX + 2][HY + 2][HZ + 2], double a,
                        double b) {
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++) {
        double center = x[i + 1][j + 1][k + 1];
        double lap = 6.0 * center - x[i][j + 1][k + 1] -
                     x[i + 2][j + 1][k + 1] - x[i + 1][j][k + 1] -
                     x[i + 1][j + 2][k + 1] - x[i + 1][j + 1][k] -
                     x[i + 1][j + 1][k + 2];
        res[i][j][k] = rhs[i][j][k] - (a * center + b * lap);
      }
}

void hpgmg_jacobi_smooth_7pt(double next[HX][HY][HZ],
                             const double cur[HX + 2][HY + 2][HZ + 2],
                             const double rhs[HX][HY][HZ],
                             const double dinv[HX][HY][HZ], double a,
                             double b, double weight) {
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++) {
        double center = cur[i + 1][j + 1][k + 1];
        double lap = 6.0 * center - cur[i][j + 1][k + 1] -
                     cur[i + 2][j + 1][k + 1] - cur[i + 1][j][k + 1] -
                     cur[i + 1][j + 2][k + 1] - cur[i + 1][j + 1][k] -
                     cur[i + 1][j + 1][k + 2];
        double ax = a * center + b * lap;
        next[i][j][k] = center + weight * dinv[i][j][k] * (rhs[i][j][k] - ax);
      }
}

void hpgmg_gsrb_smooth_7pt(double x[HX + 2][HY + 2][HZ + 2],
                           const double rhs[HX][HY][HZ],
                           const double dinv[HX][HY][HZ], int color,
                           double a, double b) {
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++)
        if (((i + j + k) & 1) == color) {
          double center = x[i + 1][j + 1][k + 1];
          double lap = 6.0 * center - x[i][j + 1][k + 1] -
                       x[i + 2][j + 1][k + 1] - x[i + 1][j][k + 1] -
                       x[i + 1][j + 2][k + 1] - x[i + 1][j + 1][k] -
                       x[i + 1][j + 1][k + 2];
          double ax = a * center + b * lap;
          x[i + 1][j + 1][k + 1] =
              center + dinv[i][j][k] * (rhs[i][j][k] - ax);
        }
}

void hpgmg_zero_vector(double a[HV]) {
  for (int i = 0; i < HV; i++)
    a[i] = 0.0;
}

void hpgmg_init_vector(double a[HV], double scalar) {
  for (int i = 0; i < HV; i++)
    a[i] = scalar;
}

void hpgmg_add_vectors(double c[HV], const double a[HV], const double b[HV],
                       double scale_a, double scale_b) {
  for (int i = 0; i < HV; i++)
    c[i] = scale_a * a[i] + scale_b * b[i];
}

void hpgmg_mul_vectors(double c[HV], const double a[HV], const double b[HV],
                       double scale) {
  for (int i = 0; i < HV; i++)
    c[i] = scale * a[i] * b[i];
}

void hpgmg_invert_vector(double c[HV], const double a[HV], double scale) {
  for (int i = 0; i < HV; i++)
    c[i] = scale / a[i];
}

void hpgmg_scale_vector(double c[HV], const double a[HV], double scale) {
  for (int i = 0; i < HV; i++)
    c[i] = scale * a[i];
}

void hpgmg_shift_vector(double c[HV], const double a[HV], double shift) {
  for (int i = 0; i < HV; i++)
    c[i] = a[i] + shift;
}

double hpgmg_dot(const double a[HV], const double b[HV]) {
  double sum = 0.0;
  for (int i = 0; i < HV; i++)
    sum += a[i] * b[i];
  return sum;
}

double hpgmg_norm(const double a[HV]) {
  double n = 0.0;
  for (int i = 0; i < HV; i++) {
    double v = PK_ABS(a[i]);
    if (v > n)
      n = v;
  }
  return n;
}

double hpgmg_mean(const double a[HV]) {
  double sum = 0.0;
  for (int i = 0; i < HV; i++)
    sum += a[i];
  return sum / (double)HV;
}

double hpgmg_error_l2(const double a[HV], const double b[HV], double h3) {
  double sum = 0.0;
  for (int i = 0; i < HV; i++) {
    double d = a[i] - b[i];
    sum += d * d * h3;
  }
  return sum;
}

void hpgmg_color_vector(double grid[HX][HY][HZ], int colors, int ci, int cj,
                        int ck) {
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++) {
        double si = (((i + ci) % colors) == 0) ? 1.0 : 0.0;
        double sj = (((j + cj) % colors) == 0) ? 1.0 : 0.0;
        double sk = (((k + ck) % colors) == 0) ? 1.0 : 0.0;
        grid[i][j][k] = si * sj * sk;
      }
}

void hpgmg_random_vector(double grid[HX][HY][HZ]) {
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++)
        grid[i][j][k] = -1.0 + 2.0 * (double)((i ^ j ^ k ^ 1) & 1);
}

void hpgmg_restriction_cell(double coarse[HX][HY][HZ],
                            const double fine[2 * HX][2 * HY][2 * HZ]) {
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++) {
        int ii = 2 * i, jj = 2 * j, kk = 2 * k;
        coarse[i][j][k] =
            (fine[ii][jj][kk] + fine[ii + 1][jj][kk] +
             fine[ii][jj + 1][kk] + fine[ii + 1][jj + 1][kk] +
             fine[ii][jj][kk + 1] + fine[ii + 1][jj][kk + 1] +
             fine[ii][jj + 1][kk + 1] + fine[ii + 1][jj + 1][kk + 1]) *
            0.125;
      }
}

void hpgmg_restriction_face_i(double coarse[HX][HY][HZ],
                              const double fine[2 * HX][2 * HY][2 * HZ]) {
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++) {
        int ii = 2 * i, jj = 2 * j, kk = 2 * k;
        coarse[i][j][k] = (fine[ii][jj][kk] + fine[ii][jj + 1][kk] +
                           fine[ii][jj][kk + 1] +
                           fine[ii][jj + 1][kk + 1]) *
                          0.25;
      }
}

void hpgmg_restriction_face_j(double coarse[HX][HY][HZ],
                              const double fine[2 * HX][2 * HY][2 * HZ]) {
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++) {
        int ii = 2 * i, jj = 2 * j, kk = 2 * k;
        coarse[i][j][k] = (fine[ii][jj][kk] + fine[ii + 1][jj][kk] +
                           fine[ii][jj][kk + 1] +
                           fine[ii + 1][jj][kk + 1]) *
                          0.25;
      }
}

void hpgmg_restriction_face_k(double coarse[HX][HY][HZ],
                              const double fine[2 * HX][2 * HY][2 * HZ]) {
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++) {
        int ii = 2 * i, jj = 2 * j, kk = 2 * k;
        coarse[i][j][k] = (fine[ii][jj][kk] + fine[ii + 1][jj][kk] +
                           fine[ii][jj + 1][kk] +
                           fine[ii + 1][jj + 1][kk]) *
                          0.25;
      }
}

void hpgmg_interpolation_p0(double fine[2 * HX][2 * HY][2 * HZ],
                            const double coarse[HX][HY][HZ]) {
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++)
        for (int di = 0; di < 2; di++)
          for (int dj = 0; dj < 2; dj++)
            for (int dk = 0; dk < 2; dk++)
              fine[2 * i + di][2 * j + dj][2 * k + dk] = coarse[i][j][k];
}

void hpgmg_interpolation_p1(double fine[2 * HX][2 * HY][2 * HZ],
                            const double coarse[HX + 2][HY + 2][HZ + 2],
                            double prescale) {
  for (int i = 0; i < 2 * HX; i++)
    for (int j = 0; j < 2 * HY; j++)
      for (int k = 0; k < 2 * HZ; k++) {
        int ci = i >> 1, cj = j >> 1, ck = k >> 1;
        int di = (i & 1) ? 1 : -1;
        int dj = (j & 1) ? 1 : -1;
        int dk = (k & 1) ? 1 : -1;
        fine[i][j][k] =
            prescale * fine[i][j][k] +
            0.421875 * coarse[ci + 1][cj + 1][ck + 1] +
            0.140625 * coarse[ci + 1 + di][cj + 1][ck + 1] +
            0.140625 * coarse[ci + 1][cj + 1 + dj][ck + 1] +
            0.140625 * coarse[ci + 1][cj + 1][ck + 1 + dk] +
            0.046875 * coarse[ci + 1 + di][cj + 1 + dj][ck + 1] +
            0.046875 * coarse[ci + 1 + di][cj + 1][ck + 1 + dk] +
            0.046875 * coarse[ci + 1][cj + 1 + dj][ck + 1 + dk] +
            0.015625 * coarse[ci + 1 + di][cj + 1 + dj][ck + 1 + dk];
      }
}

void hpgmg_interpolation_p2(double fine[2 * HX][2 * HY][2 * HZ],
                            const double coarse[HX + 2][HY + 2][HZ + 2]) {
  for (int i = 0; i < 2 * HX; i++)
    for (int j = 0; j < 2 * HY; j++)
      for (int k = 0; k < 2 * HZ; k++) {
        int ci = i >> 1, cj = j >> 1, ck = k >> 1;
        fine[i][j][k] = 0.5 * coarse[ci + 1][cj + 1][ck + 1] +
                        0.08333333333333333 *
                            (coarse[ci][cj + 1][ck + 1] +
                             coarse[ci + 2][cj + 1][ck + 1] +
                             coarse[ci + 1][cj][ck + 1] +
                             coarse[ci + 1][cj + 2][ck + 1] +
                             coarse[ci + 1][cj + 1][ck] +
                             coarse[ci + 1][cj + 1][ck + 2]);
      }
}

void hpgmg_fv2_flux(double flux[HX][HY][HZ],
                    const double a[HX + 2][HY + 2][HZ + 2]) {
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++)
        flux[i][j][k] = a[i + 2][j + 1][k + 1] - a[i + 1][j + 1][k + 1];
}

void hpgmg_fv4_flux(double flux[HX][HY][HZ],
                    const double a[HX + 4][HY + 4][HZ + 4]) {
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++)
        flux[i][j][k] =
            (-a[i][j + 2][k + 2] + 7.0 * a[i + 1][j + 2][k + 2] -
             7.0 * a[i + 2][j + 2][k + 2] + a[i + 3][j + 2][k + 2]) /
            12.0;
}

void hpgmg_cg_update(double x[HV], double r[HV], double p[HV],
                     const double Ap[HV], double alpha, double beta) {
  for (int i = 0; i < HV; i++) {
    x[i] += alpha * p[i];
    r[i] -= alpha * Ap[i];
    p[i] = r[i] + beta * p[i];
  }
}

void hpgmg_bicgstab_update(double x[HV], double r[HV], const double p[HV],
                           const double v[HV], double alpha, double omega) {
  for (int i = 0; i < HV; i++) {
    double s = r[i] - alpha * v[i];
    x[i] += alpha * p[i] + omega * s;
    r[i] = s - omega * v[i];
  }
}

#define HL 32
#define HNV 4

void hypar_first_derivative_first_order(double df[HL][HNV],
                                        const double f[HL + 1][HNV]) {
  for (int i = 0; i < HL; i++)
    for (int v = 0; v < HNV; v++)
      df[i][v] = f[i + 1][v] - f[i][v];
}

void hypar_first_derivative_second_order(double df[HL][HNV],
                                         const double f[HL + 2][HNV]) {
  for (int i = 0; i < HL; i++)
    for (int v = 0; v < HNV; v++)
      df[i][v] = 0.5 * (f[i + 2][v] - f[i][v]);
}

void hypar_first_derivative_fourth_order(double df[HL][HNV],
                                         const double f[HL + 4][HNV]) {
  for (int i = 0; i < HL; i++)
    for (int v = 0; v < HNV; v++)
      df[i][v] = (f[i][v] - 8.0 * f[i + 1][v] + 8.0 * f[i + 3][v] -
                  f[i + 4][v]) /
                 12.0;
}

void hypar_interp_first_order_upwind(double fI[HL + 1][HNV],
                                     const double fC[HL + 2][HNV], int upw) {
  for (int i = 0; i < HL + 1; i++)
    for (int v = 0; v < HNV; v++)
      fI[i][v] = upw > 0 ? fC[i][v] : fC[i + 1][v];
}

void hypar_interp_second_order_central(double fI[HL + 1][HNV],
                                       const double fC[HL + 2][HNV]) {
  for (int i = 0; i < HL + 1; i++)
    for (int v = 0; v < HNV; v++)
      fI[i][v] = 0.5 * (fC[i][v] + fC[i + 1][v]);
}

void hypar_interp_second_order_muscl(double fI[HL + 1][HNV],
                                     const double fC[HL + 3][HNV]) {
  for (int i = 0; i < HL + 1; i++)
    for (int v = 0; v < HNV; v++) {
      double dl = fC[i + 1][v] - fC[i][v];
      double dr = fC[i + 2][v] - fC[i + 1][v];
      double slope = (dl * dr <= 0.0) ? 0.0
                                      : ((PK_ABS(dl) < PK_ABS(dr)) ? dl : dr);
      fI[i][v] = fC[i + 1][v] - 0.5 * slope;
    }
}

void hypar_interp_fourth_order_central(double fI[HL + 1][HNV],
                                       const double fC[HL + 4][HNV]) {
  for (int i = 0; i < HL + 1; i++)
    for (int v = 0; v < HNV; v++)
      fI[i][v] = (-fC[i][v] + 7.0 * fC[i + 1][v] +
                  7.0 * fC[i + 2][v] - fC[i + 3][v]) /
                 12.0;
}

void hypar_interp_fifth_order_weno(double fI[HL + 1][HNV],
                                   const double fC[HL + 5][HNV],
                                   const double w1[HL + 1][HNV],
                                   const double w2[HL + 1][HNV],
                                   const double w3[HL + 1][HNV]) {
  for (int i = 0; i < HL + 1; i++)
    for (int v = 0; v < HNV; v++) {
      double f1 = (2.0 * fC[i][v] - 7.0 * fC[i + 1][v] +
                   11.0 * fC[i + 2][v]) /
                  6.0;
      double f2 = (-fC[i + 1][v] + 5.0 * fC[i + 2][v] +
                   2.0 * fC[i + 3][v]) /
                  6.0;
      double f3 = (2.0 * fC[i + 2][v] + 5.0 * fC[i + 3][v] -
                   fC[i + 4][v]) /
                  6.0;
      fI[i][v] = w1[i][v] * f1 + w2[i][v] * f2 + w3[i][v] * f3;
    }
}

void hypar_weno_weights_js(double w1[HL + 1][HNV], double w2[HL + 1][HNV],
                           double w3[HL + 1][HNV],
                           const double fC[HL + 5][HNV], double eps) {
  for (int i = 0; i < HL + 1; i++)
    for (int v = 0; v < HNV; v++) {
      double fm2 = fC[i][v], fm1 = fC[i + 1][v], f0 = fC[i + 2][v];
      double fp1 = fC[i + 3][v], fp2 = fC[i + 4][v];
      double b1 = (13.0 / 12.0) * (fm2 - 2.0 * fm1 + f0) *
                      (fm2 - 2.0 * fm1 + f0) +
                  0.25 * (fm2 - 4.0 * fm1 + 3.0 * f0) *
                      (fm2 - 4.0 * fm1 + 3.0 * f0);
      double b2 = (13.0 / 12.0) * (fm1 - 2.0 * f0 + fp1) *
                      (fm1 - 2.0 * f0 + fp1) +
                  0.25 * (fm1 - fp1) * (fm1 - fp1);
      double b3 = (13.0 / 12.0) * (f0 - 2.0 * fp1 + fp2) *
                      (f0 - 2.0 * fp1 + fp2) +
                  0.25 * (3.0 * f0 - 4.0 * fp1 + fp2) *
                      (3.0 * f0 - 4.0 * fp1 + fp2);
      double a1 = 0.1 / ((b1 + eps) * (b1 + eps));
      double a2 = 0.6 / ((b2 + eps) * (b2 + eps));
      double a3 = 0.3 / ((b3 + eps) * (b3 + eps));
      double sum = a1 + a2 + a3;
      w1[i][v] = a1 / sum;
      w2[i][v] = a2 / sum;
      w3[i][v] = a3 / sum;
    }
}

void hypar_limiter_minmod(double out[HL][HNV], const double a[HL][HNV],
                          const double b[HL][HNV]) {
  for (int i = 0; i < HL; i++)
    for (int v = 0; v < HNV; v++)
      out[i][v] = (a[i][v] * b[i][v] <= 0.0)
                      ? 0.0
                      : ((PK_ABS(a[i][v]) < PK_ABS(b[i][v])) ? a[i][v]
                                                              : b[i][v]);
}

void hypar_limiter_superbee(double out[HL][HNV], const double a[HL][HNV],
                            const double b[HL][HNV]) {
  for (int i = 0; i < HL; i++)
    for (int v = 0; v < HNV; v++) {
      double s1 = PK_MIN(2.0 * PK_ABS(a[i][v]), PK_ABS(b[i][v]));
      double s2 = PK_MIN(PK_ABS(a[i][v]), 2.0 * PK_ABS(b[i][v]));
      double mag = PK_MAX(s1, s2);
      out[i][v] = (a[i][v] * b[i][v] <= 0.0) ? 0.0
                                             : (a[i][v] < 0.0 ? -mag : mag);
    }
}

void hypar_limiter_generalized_minmod(double out[HL][HNV],
                                      const double a[HL][HNV],
                                      const double b[HL][HNV], double theta) {
  for (int i = 0; i < HL; i++)
    for (int v = 0; v < HNV; v++) {
      double aa = theta * a[i][v];
      double bb = 0.5 * (a[i][v] + b[i][v]);
      double cc = theta * b[i][v];
      double same = (aa * bb > 0.0 && bb * cc > 0.0) ? 1.0 : 0.0;
      double mag = PK_MIN(PK_ABS(aa), PK_MIN(PK_ABS(bb), PK_ABS(cc)));
      out[i][v] = same == 0.0 ? 0.0 : (aa < 0.0 ? -mag : mag);
    }
}

void hypar_limiter_vanleer(double out[HL][HNV], const double a[HL][HNV],
                           const double b[HL][HNV]) {
  for (int i = 0; i < HL; i++)
    for (int v = 0; v < HNV; v++)
      out[i][v] = (a[i][v] * b[i][v] <= 0.0)
                      ? 0.0
                      : (2.0 * a[i][v] * b[i][v]) / (a[i][v] + b[i][v]);
}

void hypar_linear_adr_advection_const(double f[HL][HNV],
                                      const double u[HL][HNV],
                                      const double a[HNV]) {
  for (int i = 0; i < HL; i++)
    for (int v = 0; v < HNV; v++)
      f[i][v] = a[v] * u[i][v];
}

void hypar_linear_adr_advection_var(double f[HL][HNV],
                                    const double u[HL][HNV],
                                    const double a[HL][HNV]) {
  for (int i = 0; i < HL; i++)
    for (int v = 0; v < HNV; v++)
      f[i][v] = a[i][v] * u[i][v];
}

void hypar_linear_adr_diffusion_g(double f[HL][HNV], const double u[HL][HNV],
                                  const double d[HNV]) {
  for (int i = 0; i < HL; i++)
    for (int v = 0; v < HNV; v++)
      f[i][v] = d[v] * u[i][v];
}

void hypar_linear_adr_diffusion_h(double f[HL][HNV], const double u[HL][HNV],
                                  const double d[HNV], int same_dir) {
  for (int i = 0; i < HL; i++)
    for (int v = 0; v < HNV; v++)
      f[i][v] = same_dir ? d[v] * u[i][v] : 0.0;
}

void hypar_linear_adr_upwind_const(double fI[HL + 1][HNV],
                                   const double fL[HL + 1][HNV],
                                   const double fR[HL + 1][HNV],
                                   const double a[HNV]) {
  for (int i = 0; i < HL + 1; i++)
    for (int v = 0; v < HNV; v++)
      fI[i][v] = a[v] > 0.0 ? fL[i][v] : fR[i][v];
}

void hypar_linear_adr_upwind_var(double fI[HL + 1][HNV],
                                 const double fL[HL + 1][HNV],
                                 const double fR[HL + 1][HNV],
                                 const double uL[HL + 1][HNV],
                                 const double uR[HL + 1][HNV],
                                 const double eigL[HL + 1][HNV],
                                 const double eigR[HL + 1][HNV]) {
  for (int i = 0; i < HL + 1; i++)
    for (int v = 0; v < HNV; v++) {
      if (eigL[i][v] > 0.0 && eigR[i][v] > 0.0) {
        fI[i][v] = fL[i][v];
      } else if (eigL[i][v] < 0.0 && eigR[i][v] < 0.0) {
        fI[i][v] = fR[i][v];
      } else {
        double alpha = PK_MAX(PK_ABS(eigL[i][v]), PK_ABS(eigR[i][v]));
        fI[i][v] =
            0.5 * (fL[i][v] + fR[i][v] - alpha * (uR[i][v] - uL[i][v]));
      }
    }
}

void hypar_linear_adr_centered_flux(double fI[HL + 1][HNV],
                                    const double fL[HL + 1][HNV],
                                    const double fR[HL + 1][HNV]) {
  for (int i = 0; i < HL + 1; i++)
    for (int v = 0; v < HNV; v++)
      fI[i][v] = 0.5 * (fL[i][v] + fR[i][v]);
}

void hypar_linear_adr_reaction(double r[HL][HNV], const double u[HL][HNV],
                               const double source[HL][HNV], double lambda) {
  for (int i = 0; i < HL; i++)
    for (int v = 0; v < HNV; v++)
      r[i][v] = source[i][v] - lambda * u[i][v];
}

void hypar_burgers_advection(double f[HL], const double u[HL]) {
  for (int i = 0; i < HL; i++)
    f[i] = 0.5 * u[i] * u[i];
}

void hypar_burgers_upwind(double fI[HL + 1], const double fL[HL + 1],
                          const double fR[HL + 1], const double uL[HL + 1],
                          const double uR[HL + 1]) {
  for (int i = 0; i < HL + 1; i++) {
    double alpha = PK_MAX(PK_ABS(uL[i]), PK_ABS(uR[i]));
    fI[i] = 0.5 * (fL[i] + fR[i] - alpha * (uR[i] - uL[i]));
  }
}

void hypar_euler1d_flux(double f[HL][3], const double u[HL][3],
                        double gamma) {
  for (int i = 0; i < HL; i++) {
    double rho = u[i][0];
    double mom = u[i][1];
    double eng = u[i][2];
    double vel = mom / rho;
    double p = (gamma - 1.0) * (eng - 0.5 * rho * vel * vel);
    f[i][0] = mom;
    f[i][1] = mom * vel + p;
    f[i][2] = (eng + p) * vel;
  }
}

void hypar_euler1d_llf(double fI[HL + 1][3], const double fL[HL + 1][3],
                       const double fR[HL + 1][3],
                       const double uL[HL + 1][3],
                       const double uR[HL + 1][3], double alpha) {
  for (int i = 0; i < HL + 1; i++)
    for (int v = 0; v < 3; v++)
      fI[i][v] = 0.5 * (fL[i][v] + fR[i][v] - alpha * (uR[i][v] - uL[i][v]));
}

void hypar_euler2d_flux_x(double f[HL][4], const double u[HL][4],
                          double gamma) {
  for (int i = 0; i < HL; i++) {
    double rho = u[i][0];
    double mx = u[i][1];
    double my = u[i][2];
    double eng = u[i][3];
    double vx = mx / rho;
    double vy = my / rho;
    double p = (gamma - 1.0) * (eng - 0.5 * rho * (vx * vx + vy * vy));
    f[i][0] = mx;
    f[i][1] = mx * vx + p;
    f[i][2] = my * vx;
    f[i][3] = (eng + p) * vx;
  }
}

void hypar_euler2d_flux_y(double f[HL][4], const double u[HL][4],
                          double gamma) {
  for (int i = 0; i < HL; i++) {
    double rho = u[i][0];
    double mx = u[i][1];
    double my = u[i][2];
    double eng = u[i][3];
    double vx = mx / rho;
    double vy = my / rho;
    double p = (gamma - 1.0) * (eng - 0.5 * rho * (vx * vx + vy * vy));
    f[i][0] = my;
    f[i][1] = mx * vy;
    f[i][2] = my * vy + p;
    f[i][3] = (eng + p) * vy;
  }
}

void hypar_euler2d_llf(double fI[HL + 1][4], const double fL[HL + 1][4],
                       const double fR[HL + 1][4],
                       const double uL[HL + 1][4],
                       const double uR[HL + 1][4], double alpha) {
  for (int i = 0; i < HL + 1; i++)
    for (int v = 0; v < 4; v++)
      fI[i][v] = 0.5 * (fL[i][v] + fR[i][v] - alpha * (uR[i][v] - uL[i][v]));
}

#define SX 8
#define SY 8
#define SZ 8

void swfft_redistribute_2_to_3_pack(double chunk[SX][SY][SZ],
                                    const double pencil[SX * SY * SZ]) {
  for (int x = 0; x < SX; x++)
    for (int y = 0; y < SY; y++)
      for (int z = 0; z < SZ; z++)
        chunk[x][y][z] = pencil[(x * SY + y) * SZ + z];
}

void swfft_redistribute_3_to_2_unpack(double pencil[SX * SY * SZ],
                                      const double chunk[SX][SY][SZ]) {
  for (int x = 0; x < SX; x++)
    for (int y = 0; y < SY; y++)
      for (int z = 0; z < SZ; z++)
        pencil[(x * SY + y) * SZ + z] = chunk[x][y][z];
}

void swfft_slab_pack(double slab[SX][SY], const double cube[SX][SY][SZ],
                     int z) {
  for (int x = 0; x < SX; x++)
    for (int y = 0; y < SY; y++)
      slab[x][y] = cube[x][y][z];
}

void swfft_slab_unpack(double cube[SX][SY][SZ], const double slab[SX][SY],
                       int z) {
  for (int x = 0; x < SX; x++)
    for (int y = 0; y < SY; y++)
      cube[x][y][z] = slab[x][y];
}

void swfft_transpose_xy(double out[SY][SX][SZ],
                        const double in[SX][SY][SZ]) {
  for (int x = 0; x < SX; x++)
    for (int y = 0; y < SY; y++)
      for (int z = 0; z < SZ; z++)
        out[y][x][z] = in[x][y][z];
}

void swfft_transpose_yz(double out[SX][SZ][SY],
                        const double in[SX][SY][SZ]) {
  for (int x = 0; x < SX; x++)
    for (int y = 0; y < SY; y++)
      for (int z = 0; z < SZ; z++)
        out[x][z][y] = in[x][y][z];
}

#define EN 16

void exasp2_normalize_dense(double rho[EN][EN], const double h[EN][EN],
                            double emax, double emin) {
  double range = emax - emin;
  for (int i = 0; i < EN; i++)
    for (int j = 0; j < EN; j++) {
      double val = -h[i][j] / range;
      if (i == j)
        val += emax / range;
      rho[i][j] = val;
    }
}

void exasp2_normalize_dense_split(double rho[EN][EN], const double h[EN][EN],
                                  double emax, double emin) {
  double range = emax - emin;
  for (int i = 0; i < EN; i++)
    for (int j = 0; j < EN; j++)
      rho[i][j] = -h[i][j] / range;

  for (int i = 0; i < EN; i++)
    rho[i][i] += emax / range;
}

void exasp2_dense_square(double x2[EN][EN], const double x[EN][EN]) {
  for (int i = 0; i < EN; i++)
    for (int j = 0; j < EN; j++) {
      double sum = 0.0;
      for (int k = 0; k < EN; k++)
        sum += x[i][k] * x[k][j];
      x2[i][j] = sum;
    }
}

void exasp2_sp2_update_2x_minus_x2(double x[EN][EN],
                                   const double x2[EN][EN]) {
  for (int i = 0; i < EN; i++)
    for (int j = 0; j < EN; j++)
      x[i][j] = 2.0 * x[i][j] - x2[i][j];
}

void exasp2_sp2_select_square(double x[EN][EN], const double x2[EN][EN],
                              int take_square) {
  for (int i = 0; i < EN; i++)
    for (int j = 0; j < EN; j++)
      x[i][j] = take_square ? x2[i][j] : 2.0 * x[i][j] - x2[i][j];
}

double exasp2_trace(const double x[EN][EN]) {
  double tr = 0.0;
  for (int i = 0; i < EN; i++)
    tr += x[i][i];
  return tr;
}

void exasp2_axpby(double c[HV], const double a[HV], const double b[HV],
                  double alpha, double beta) {
  for (int i = 0; i < HV; i++)
    c[i] = alpha * a[i] + beta * b[i];
}

void exasp2_spmv(double y[EN], const double a[EN][EN], const double x[EN]) {
  for (int i = 0; i < EN; i++) {
    double sum = 0.0;
    for (int j = 0; j < EN; j++)
      sum += a[i][j] * x[j];
    y[i] = sum;
  }
}

void exasp2_conjugate_gradient_step(double x[EN], double r[EN], double p[EN],
                                    const double Ap[EN], double alpha,
                                    double beta) {
  for (int i = 0; i < EN; i++) {
    x[i] += alpha * p[i];
    r[i] -= alpha * Ap[i];
    p[i] = r[i] + beta * p[i];
  }
}
