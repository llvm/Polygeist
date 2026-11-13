/* Negative (layout transpose): row-major syr2k without the beta scaling.
 * The proof needs a transposed binding whose flag-flipped equivalent
 * (dsyr2k TRANS='T', BETA=1) is not a library specification, and the
 * general-beta specs must not bind beta to 1.0 -> no launch. */
void syr2k_beta1(int n, int m, double alpha, double C[n][n], double A[n][m], double B[n][m]) {
  for (int i = 0; i < n; i++)
    for (int k = 0; k < m; k++)
      for (int j = 0; j <= i; j++)
        C[i][j] += A[j][k] * alpha * B[i][k] + B[j][k] * alpha * A[i][k];
}
