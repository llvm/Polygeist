//@ routine: symm
//@ kind: wrong_triangle
//@ expect: nomatch
// Second statement reads the upper triangle A[k][i]: inconsistent triangles.
void symm__wrong_triangle(int m, int n, double alpha, double beta,
                          double C[m][n], double A[m][m], double B[m][n]) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      double temp2 = 0;
      for (int k = 0; k < i; k++) {
        C[k][j] += alpha * B[i][j] * A[i][k];
        temp2 += B[k][j] * A[k][i];
      }
      C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
    }
}
