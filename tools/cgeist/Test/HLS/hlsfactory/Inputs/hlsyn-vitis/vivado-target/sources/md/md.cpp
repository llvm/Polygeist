#include "md.h"
void md_kernel(double force_x[256], double force_y[256], double force_z[256],
               double position_x[256], double position_y[256],
               double position_z[256], int NL[4096]) {
  double force_x_buf[256];
  double force_y_buf[256];
  double force_z_buf[256];
  double position_x_buf[256];
  double position_y_buf[256];
  double position_z_buf[256];
  int NL_buf[4096];
  memcpy(position_x_buf, position_x, sizeof(double) * 256);
  memcpy(position_y_buf, position_y, sizeof(double) * 256);
  memcpy(position_z_buf, position_z, sizeof(double) * 256);
  memcpy(NL_buf, NL, sizeof(int) * 4096);

  double delx;
  double dely;
  double delz;
  double r2inv;
  double r6inv;
  double potential;
  double force;
  double j_x;
  double j_y;
  double j_z;
  double i_x;
  double i_y;
  double i_z;
  double fx;
  double fy;
  double fz;
  int i;
  int j;
  int jidx;

  L0: for (i = 0; i < 256; i++) {
#pragma HLS pipeline 
#pragma HLS unroll factor=2

    i_x = position_x_buf[i];
    i_y = position_y_buf[i];
    i_z = position_z_buf[i];
    fx = ((double)0);
    fy = ((double)0);
    fz = ((double)0);
    L1: for (j = 0; j < 16; j++) {
      // Get neighbor
      jidx = NL_buf[i * 16 + j];
      // Look up x,y,z positions
      j_x = position_x_buf[jidx];
      j_y = position_y_buf[jidx];
      j_z = position_z_buf[jidx];
      // Calc distance
      delx = i_x - j_x;
      dely = i_y - j_y;
      delz = i_z - j_z;
      r2inv = 1.0 / (delx * delx + dely * dely + delz * delz);
      // Assume no cutoff and aways account for all nodes in area
      r6inv = r2inv * r2inv * r2inv;
      potential = r6inv * (1.5 * r6inv - 2.0);
      // Sum changes in force
      force = r2inv * potential;
      fx += delx * force;
      fy += dely * force;
      fz += delz * force;
    }
    // Update forces after all neighbors accounted for.
    force_x_buf[i] = fx;
    force_y_buf[i] = fy;
    force_z_buf[i] = fz;
    // printf("dF=%lf,%lf,%lf\n", fx, fy, fz);
  }
  memcpy(force_x, force_x_buf, sizeof(double) * 256);
  memcpy(force_y, force_y_buf, sizeof(double) * 256);
  memcpy(force_z, force_z_buf, sizeof(double) * 256);
}
