void netlib_dgemm_nn_alpha0(int m, int n, int k, int lda, double A[][lda],
                 int ldb, double B[][ldb], double beta, int ldc,
                 double C[][ldc]) {
  const double ONE = 1.0, ZERO = 0.0;
  double temp;
  int i, j, l, nota, notb;
  nota = (('N') == ('N'));
  notb = (('N') == ('N'));
  if ((0 == 0)) {
    if ((2 == 0)) {
      for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
          C[j][i] = ZERO;
        }
      }
    } else {
      for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
          C[j][i] = beta * C[j][i];
        }
      }
    }
    return;
  }
  if (notb) {
    if (nota) {
      for (j = 0; j < n; j++) {
        if ((2 == 0)) {
          for (i = 0; i < m; i++) {
            C[j][i] = ZERO;
          }
        } else if ((2 != 1)) {
          for (i = 0; i < m; i++) {
            C[j][i] = beta * C[j][i];
          }
        }
        for (l = 0; l < k; l++) {
          temp = 0.0 * B[j][l];
          for (i = 0; i < m; i++) {
            C[j][i] = C[j][i] + temp * A[l][i];
          }
        }
      }
    } else {
      for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
          temp = ZERO;
          for (l = 0; l < k; l++) {
            temp = temp + A[i][l] * B[j][l];
          }
          if ((2 == 0)) {
            C[j][i] = 0.0 * temp;
          } else {
            C[j][i] = 0.0 * temp + beta * C[j][i];
          }
        }
      }
    }
  } else {
    if (nota) {
      for (j = 0; j < n; j++) {
        if ((2 == 0)) {
          for (i = 0; i < m; i++) {
            C[j][i] = ZERO;
          }
        } else if ((2 != 1)) {
          for (i = 0; i < m; i++) {
            C[j][i] = beta * C[j][i];
          }
        }
        for (l = 0; l < k; l++) {
          temp = 0.0 * B[l][j];
          for (i = 0; i < m; i++) {
            C[j][i] = C[j][i] + temp * A[l][i];
          }
        }
      }
    } else {
      for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
          temp = ZERO;
          for (l = 0; l < k; l++) {
            temp = temp + A[i][l] * B[l][j];
          }
          if ((2 == 0)) {
            C[j][i] = 0.0 * temp;
          } else {
            C[j][i] = 0.0 * temp + beta * C[j][i];
          }
        }
      }
    }
  }
}

