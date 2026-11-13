/* Negative (layout transpose): a symmetric rank-2k update in which only B is
 * stored transposed relative to A.  The manifest has no rule that flips a
 * flag for B alone (TRANS applies to A and B together), so no launch. */
void syr2k_halfT(int n, int k, double alpha, double beta, double C[n][n], double A[n][k],
                 double B[k][n]) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j <= i; j++) {
      C[i][j] *= beta;
      for (int l = 0; l < k; l++)
        C[i][j] += alpha * A[i][l] * B[l][j] + alpha * B[l][i] * A[j][l];
    }
}
