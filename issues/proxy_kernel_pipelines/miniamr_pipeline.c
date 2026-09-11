// miniAMR pipeline-shaped fixture built from the standalone extracted kernel
// bodies. This intentionally keeps app/MPI/block metadata out of the way while
// preserving the stencil, material update, and pack/unpack loop shapes.

#define MX 12
#define MY 10
#define MZ 8
#define MMAT 4

void miniamr_pipeline(double grid[MX + 2][MY + 2][MZ + 2],
                      double state[MX][MY][MZ],
                      double tmp[MX][MY][MZ],
                      double tmp2[MX][MY][MZ],
                      const double material[MMAT][MX][MY][MZ],
                      const double coeff[MX][MY][MZ],
                      const double coeff27[3][3][3],
                      const double x_in[MX + 2][MY][MZ],
                      const double y_in[MX][MY + 2][MZ],
                      const double z_in[MX][MY][MZ + 2],
                      double face_buf[MY][MZ],
                      double block_buf[MX * MY * MZ],
                      double a0, double a1, double a2) {
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
        block_buf[(i * MY + j) * MZ + k] = state[i][j][k];

  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++)
        tmp[i][j][k] = block_buf[(i * MY + j) * MZ + k];

  // 7-point average stencil.
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++)
        state[i][j][k] = (grid[i][j + 1][k + 1] +
                          grid[i + 1][j][k + 1] +
                          grid[i + 1][j + 1][k] +
                          grid[i + 1][j + 1][k + 1] +
                          grid[i + 1][j + 1][k + 2] +
                          grid[i + 1][j + 2][k + 1] +
                          grid[i + 2][j + 1][k + 1]) /
                         7.0;

  // 27-point average stencil.
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double sum = 0.0;
        for (int di = 0; di < 3; di++)
          for (int dj = 0; dj < 3; dj++)
            for (int dk = 0; dk < 3; dk++)
              sum += grid[i + di][j + dj][k + dk];
        tmp[i][j][k] = sum / 27.0;
      }

  // Material coupled sum and pointwise update.
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double acc = tmp[i][j][k];
        for (int v = 0; v < MMAT; v++)
          acc += material[v][i][j][k] * state[i][j][k];
        tmp[i][j][k] = acc;
      }

  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++)
        tmp[i][j][k] += tmp[i][j][k] *
                        (state[i][j][k] + tmp2[i][j][k] -
                         a2 * tmp[i][j][k]);

  // Directional stencils.
  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double left = x_in[i][j][k];
        double center = x_in[i + 1][j][k];
        double right = x_in[i + 2][j][k];
        state[i][j][k] = center + a0 * (left - 2.0 * center + right) +
                         a1 * (right - left);
      }

  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double left = y_in[i][j][k];
        double center = y_in[i][j + 1][k];
        double right = y_in[i][j + 2][k];
        tmp[i][j][k] = center + a0 * (left - 2.0 * center + right) +
                       a1 * (right - left);
      }

  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double left = z_in[i][j][k];
        double center = z_in[i][j][k + 1];
        double right = z_in[i][j][k + 2];
        tmp2[i][j][k] = center + a0 * (left - 2.0 * center + right) +
                        a1 * (right - left);
      }

  // Weighted 7-point and 27-point stencils.
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
        state[i][j][k] = center + coeff[i][j][k] * lap;
      }

  for (int i = 0; i < MX; i++)
    for (int j = 0; j < MY; j++)
      for (int k = 0; k < MZ; k++) {
        double sum = 0.0;
        for (int di = 0; di < 3; di++)
          for (int dj = 0; dj < 3; dj++)
            for (int dk = 0; dk < 3; dk++)
              sum += coeff27[di][dj][dk] * grid[i + di][j + dj][k + dk];
        tmp[i][j][k] = sum;
      }
}
