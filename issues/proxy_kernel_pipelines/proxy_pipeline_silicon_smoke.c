#include <stdio.h>

#include "miniamr_pipeline_easy.c"
#include "hpgmg_pipeline_easy.c"
#include "hypar_pipeline_easy.c"
#include "swfft_pipeline_easy.c"
#include "exasp2_pipeline_easy.c"

static double seed_value(int a, int b, int c, int salt) {
  int v = (a * 17 + b * 13 + c * 7 + salt * 19 + 11) % 101;
  return ((double)v - 50.0) * 0.01;
}

static double checksum_1d(const double *data, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i)
    sum += data[i] * (1.0 + 0.0001 * (double)(i % 17));
  return sum;
}

static double miniamr_smoke(void) {
  static double grid[MX + 2][MY + 2][MZ + 2];
  static double x_in[MX + 2][MY][MZ];
  static double y_in[MX][MY + 2][MZ];
  static double z_in[MX][MY][MZ + 2];
  static double coeff[MX][MY][MZ];
  static double coeff27[3][3][3];
  static double face_buf[MY][MZ];
  static double block_buf[MX * MY * MZ];
  static double avg7[MX][MY][MZ];
  static double dir_x[MX][MY][MZ];
  static double dir_y[MX][MY][MZ];
  static double dir_z[MX][MY][MZ];
  static double weighted7[MX][MY][MZ];
  static double weighted27[MX][MY][MZ];
  static double final_out[MX][MY][MZ];

  for (int i = 0; i < MX + 2; ++i)
    for (int j = 0; j < MY + 2; ++j)
      for (int k = 0; k < MZ + 2; ++k)
        grid[i][j][k] = seed_value(i, j, k, 1);
  for (int i = 0; i < MX + 2; ++i)
    for (int j = 0; j < MY; ++j)
      for (int k = 0; k < MZ; ++k)
        x_in[i][j][k] = seed_value(i, j, k, 2);
  for (int i = 0; i < MX; ++i)
    for (int j = 0; j < MY + 2; ++j)
      for (int k = 0; k < MZ; ++k)
        y_in[i][j][k] = seed_value(i, j, k, 3);
  for (int i = 0; i < MX; ++i)
    for (int j = 0; j < MY; ++j)
      for (int k = 0; k < MZ + 2; ++k)
        z_in[i][j][k] = seed_value(i, j, k, 4);
  for (int i = 0; i < MX; ++i)
    for (int j = 0; j < MY; ++j)
      for (int k = 0; k < MZ; ++k) {
        coeff[i][j][k] = 0.05 + seed_value(i, j, k, 5);
        avg7[i][j][k] = seed_value(i, j, k, 6);
      }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        coeff27[i][j][k] = 0.001 * (double)(1 + i + 2 * j + 3 * k);

  miniamr_pipeline_easy(grid, x_in, y_in, z_in, coeff, coeff27, face_buf,
                        block_buf, avg7, dir_x, dir_y, dir_z, weighted7,
                        weighted27, final_out, 0.125, 0.03125);
  return checksum_1d(&final_out[0][0][0], MX * MY * MZ);
}

static double hpgmg_smoke(void) {
  static double x[HX + 2][HY + 2][HZ + 2];
  static double rhs[HX][HY][HZ];
  static double dinv[HX][HY][HZ];
  static double fine[2 * HX][2 * HY][2 * HZ];
  static double Ax[HX][HY][HZ];
  static double res[HX][HY][HZ];
  static double next[HX][HY][HZ];
  static double coarse[HX][HY][HZ];
  static double prolong[2 * HX][2 * HY][2 * HZ];
  static double vx[HV], vr[HV], vp[HV], vAp[HV];

  for (int i = 0; i < HX + 2; ++i)
    for (int j = 0; j < HY + 2; ++j)
      for (int k = 0; k < HZ + 2; ++k)
        x[i][j][k] = seed_value(i, j, k, 10);
  for (int i = 0; i < HX; ++i)
    for (int j = 0; j < HY; ++j)
      for (int k = 0; k < HZ; ++k) {
        rhs[i][j][k] = seed_value(i, j, k, 11);
        dinv[i][j][k] = 0.5 + 0.01 * (double)((i + j + k) % 9);
      }
  for (int i = 0; i < 2 * HX; ++i)
    for (int j = 0; j < 2 * HY; ++j)
      for (int k = 0; k < 2 * HZ; ++k)
        fine[i][j][k] = seed_value(i, j, k, 12);
  for (int i = 0; i < HV; ++i) {
    vx[i] = seed_value(i, 0, 0, 13);
    vr[i] = seed_value(i, 0, 0, 14);
    vp[i] = seed_value(i, 0, 0, 15);
    vAp[i] = seed_value(i, 0, 0, 16);
  }

  hpgmg_pipeline_easy(x, rhs, dinv, fine, Ax, res, next, coarse, prolong, vx,
                      vr, vp, vAp, 0.7, 0.2, 0.8, 0.05, 0.25);
  return checksum_1d(&next[0][0][0], HX * HY * HZ) +
         checksum_1d(vx, HV) + checksum_1d(vr, HV) + checksum_1d(vp, HV);
}

static double hypar_smoke(void) {
  static double fC[HL + 5][HNV];
  static double u[HL][HNV], source[HL][HNV];
  static double eigL[HL + 1][HNV], eigR[HL + 1][HNV];
  static double uL[HL + 1][HNV], uR[HL + 1][HNV];
  static double df[HL][HNV], w1[HL + 1][HNV], w2[HL + 1][HNV];
  static double w3[HL + 1][HNV], fL[HL + 1][HNV], fR[HL + 1][HNV];
  static double limited[HL][HNV], flux[HL + 1][HNV];
  static double reaction[HL][HNV], u_next[HL][HNV];

  for (int i = 0; i < HL + 5; ++i)
    for (int v = 0; v < HNV; ++v)
      fC[i][v] = seed_value(i, v, 0, 20);
  for (int i = 0; i < HL; ++i)
    for (int v = 0; v < HNV; ++v) {
      u[i][v] = seed_value(i, v, 0, 21);
      source[i][v] = seed_value(i, v, 0, 22);
    }
  for (int i = 0; i < HL + 1; ++i)
    for (int v = 0; v < HNV; ++v) {
      eigL[i][v] = seed_value(i, v, 0, 23);
      eigR[i][v] = seed_value(i, v, 0, 24);
      uL[i][v] = seed_value(i, v, 0, 25);
      uR[i][v] = seed_value(i, v, 0, 26);
    }

  hypar_pipeline_easy(fC, u, source, eigL, eigR, uL, uR, df, w1, w2, w3, fL,
                      fR, limited, flux, reaction, u_next, 1.0e-3, 0.2, 0.01);
  return checksum_1d(&df[0][0], HL * HNV) +
         checksum_1d(&limited[0][0], HL * HNV) +
         checksum_1d(&u_next[0][0], HL * HNV);
}

static double swfft_smoke(void) {
  static double pencil_in[SX * SY * SZ];
  static double cube_in[SX][SY][SZ];
  static double chunk[SX][SY][SZ], slab[SX][SY], cube_work[SX][SY][SZ];
  static double xy[SY][SX][SZ], yz[SX][SZ][SY], pencil_out[SX * SY * SZ];

  for (int i = 0; i < SX * SY * SZ; ++i)
    pencil_in[i] = seed_value(i, 0, 0, 30);
  for (int x = 0; x < SX; ++x)
    for (int y = 0; y < SY; ++y)
      for (int z = 0; z < SZ; ++z)
        cube_in[x][y][z] = seed_value(x, y, z, 31);

  swfft_pipeline_easy(pencil_in, cube_in, chunk, slab, cube_work, xy, yz,
                      pencil_out);
  return checksum_1d(pencil_out, SX * SY * SZ) +
         checksum_1d(&xy[0][0][0], SX * SY * SZ) +
         checksum_1d(&yz[0][0][0], SX * SY * SZ);
}

static double exasp2_smoke(void) {
  static double rho[EN][EN], h[EN][EN], x[EN][EN], x2[EN][EN], a[EN][EN];
  static double v[EN], y[EN], cg_x[EN], cg_r[EN], cg_p[EN], cg_Ap[EN];
  static double axpby_out[EN], axpby_a[EN], axpby_b[EN], trace_diag[EN];

  for (int i = 0; i < EN; ++i) {
    v[i] = seed_value(i, 0, 0, 40);
    cg_x[i] = seed_value(i, 0, 0, 41);
    cg_r[i] = seed_value(i, 0, 0, 42);
    cg_p[i] = seed_value(i, 0, 0, 43);
    cg_Ap[i] = seed_value(i, 0, 0, 44);
    axpby_a[i] = seed_value(i, 0, 0, 45);
    axpby_b[i] = seed_value(i, 0, 0, 46);
    for (int j = 0; j < EN; ++j) {
      h[i][j] = seed_value(i, j, 0, 47);
      a[i][j] = seed_value(i, j, 0, 48);
      rho[i][j] = 0.0;
      x[i][j] = seed_value(i, j, 0, 49);
    }
  }

  exasp2_pipeline_easy(rho, h, x, x2, a, v, y, cg_x, cg_r, cg_p, cg_Ap,
                       axpby_out, axpby_a, axpby_b, trace_diag, 2.0, -1.0,
                       0.07, 0.3, 1);
  return checksum_1d(&x[0][0], EN * EN) + checksum_1d(y, EN) +
         checksum_1d(cg_x, EN) + checksum_1d(trace_diag, EN);
}

int main(void) {
  double c0 = miniamr_smoke();
  double c1 = hpgmg_smoke();
  double c2 = hypar_smoke();
  double c3 = swfft_smoke();
  double c4 = exasp2_smoke();
  printf("miniamr %.12f\n", c0);
  printf("hpgmg %.12f\n", c1);
  printf("hypar %.12f\n", c2);
  printf("swfft %.12f\n", c3);
  printf("exasp2 %.12f\n", c4);
  printf("total %.12f\n", c0 + c1 + c2 + c3 + c4);
  return 0;
}
