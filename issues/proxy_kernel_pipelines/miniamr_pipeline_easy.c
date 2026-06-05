// miniAMR-style pipeline fixture designed to raise cleanly.
//
// This is intentionally not the full miniAMR app. It keeps the pipeline
// sequence we care about for the paper story, but uses direct array operands
// and Linalg-friendly kernel shapes:
//   - no block/AMR metadata
//   - no double **** pointer chasing
//   - no local scalar alloca accumulator for the unweighted 27-point average

#define MX 12
#define MY 10
#define MZ 8

void miniamr_pipeline_easy(double grid[MX + 2][MY + 2][MZ + 2],
                           const double x_in[MX + 2][MY][MZ],
                           const double y_in[MX][MY + 2][MZ],
                           const double z_in[MX][MY][MZ + 2],
                           const double coeff[MX][MY][MZ],
                           const double coeff27[3][3][3],
                           double face_buf[MY][MZ],
                           double block_buf[MX * MY * MZ],
                           double avg7[MX][MY][MZ],
                           double dir_x[MX][MY][MZ],
                           double dir_y[MX][MY][MZ],
                           double dir_z[MX][MY][MZ],
                           double weighted7[MX][MY][MZ],
                           double weighted27[MX][MY][MZ],
                           double final_out[MX][MY][MZ],
                           double a0, double a1) {
  // Halo face pack/unpack.
  for (int j = 0; j < MY; j++)
    for (int k = 0; k < MZ; k++)
      face_buf[j][k] = grid[1][j + 1][k + 1];

  for (int j = 0; j < MY; j++)
    for (int k = 0; k < MZ; k++)
      grid[0][j + 1][k + 1] = face_buf[j][k];

  // Block pack/unpack.
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++)
        block_buf[(i * MY + j) * MZ + k] = avg7[i][j][k];

  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++)
        final_out[i][j][k] = block_buf[(i * MY + j) * MZ + k];

  // 7-point average stencil.
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++)
        avg7[i][j][k] = (grid[i][j + 1][k + 1] +
                         grid[i + 1][j][k + 1] +
                         grid[i + 1][j + 1][k] +
                         grid[i + 1][j + 1][k + 1] +
                         grid[i + 1][j + 1][k + 2] +
                         grid[i + 1][j + 2][k + 1] +
                         grid[i + 2][j + 1][k + 1]) /
                        7.0;

  // Directional stencils.
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double left = x_in[i][j][k];
        double center = x_in[i + 1][j][k];
        double right = x_in[i + 2][j][k];
        dir_x[i][j][k] = center + a0 * (left - 2.0 * center + right) +
                         a1 * (right - left);
      }

  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double left = y_in[i][j][k];
        double center = y_in[i][j + 1][k];
        double right = y_in[i][j + 2][k];
        dir_y[i][j][k] = center + a0 * (left - 2.0 * center + right) +
                         a1 * (right - left);
      }

  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double left = z_in[i][j][k];
        double center = z_in[i][j][k + 1];
        double right = z_in[i][j][k + 2];
        dir_z[i][j][k] = center + a0 * (left - 2.0 * center + right) +
                         a1 * (right - left);
      }

  // Weighted 7-point stencil.
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double center = grid[i + 1][j + 1][k + 1];
        double lap = grid[i][j + 1][k + 1] +
                     grid[i + 2][j + 1][k + 1] +
                     grid[i + 1][j][k + 1] +
                     grid[i + 1][j + 2][k + 1] +
                     grid[i + 1][j + 1][k] +
                     grid[i + 1][j + 1][k + 2] -
                     6.0 * center;
        weighted7[i][j][k] = center + coeff[i][j][k] * lap;
      }

  // Weighted 27-point stencil. This raises better than the unweighted
  // local-scalar-accumulator average because the coefficient tensor exposes
  // the inner 3x3x3 loop as a clean Linalg contraction.
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double sum = 0.0;
        for (int di = 0; di < 3; di++)
          for (int dj = 0; dj < 3; dj++)
            for (int dk = 0; dk < 3; dk++)
              sum += coeff27[di][dj][dk] * grid[i + di][j + dj][k + dk];
        weighted27[i][j][k] = sum;
      }

  // Final pointwise combine so the fixture is a true pipeline, not just a
  // bag of independent kernels.
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++)
        final_out[i][j][k] = avg7[i][j][k] + dir_x[i][j][k] +
                             dir_y[i][j][k] + dir_z[i][j][k] +
                             weighted7[i][j][k] + weighted27[i][j][k];
}
