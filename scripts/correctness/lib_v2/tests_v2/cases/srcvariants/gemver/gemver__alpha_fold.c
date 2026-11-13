//@ transform: alpha_fold
// beta / alpha applied to the vector operand instead of the matrix element.
void gemver__alpha_fold(int n, double alpha, double beta, double A[n][n], double u1[n], double v1[n], double u2[n], double v2[n], double w[n], double x[n], double y[n], double z[n]) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      x[i] = x[i] + A[j][i] * (beta * y[j]);
  for (int i = 0; i < n; i++)
    x[i] = x[i] + z[i];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      w[i] = w[i] + A[i][j] * (alpha * x[j]);
}
