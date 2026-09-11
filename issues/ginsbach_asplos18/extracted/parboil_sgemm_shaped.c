/* Diagnostic shaped-ABI form of Parboil SGEMM. */
void parboil_basic_sgemm_shaped(
    int m, int n, int k, float alpha,
    const float A[k][m], const float B[k][n],
    float beta, float C[n][m]) {
  for (int mm = 0; mm < m; ++mm)
    for (int nn = 0; nn < n; ++nn)
      C[nn][mm] = beta * C[nn][mm];

  for (int mm = 0; mm < m; ++mm)
    for (int nn = 0; nn < n; ++nn)
      for (int i = 0; i < k; ++i)
        C[nn][mm] += alpha * A[i][mm] * B[i][nn];
}
