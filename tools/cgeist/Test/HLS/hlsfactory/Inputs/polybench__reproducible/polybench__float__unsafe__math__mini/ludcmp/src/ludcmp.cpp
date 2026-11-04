#include "ludcmp.h"


void kernel_ludcmp(
		   double A[ 40 + 0][40 + 0],
		   double b[ 40 + 0],
		   double x[ 40 + 0],
		   double y[ 40 + 0])
{
  #pragma HLS top name=kernel_ludcmp

    const int n = 40;

  int i, j, k;

  double w;

  lp1: for (i = 0; i < n; i++) {
    lp2: for (j = 0; j <i; j++) {
       w = A[i][j];
       lp3: for (k = 0; k < j; k++) {
          w -= A[i][k] * A[k][j];
       }
        A[i][j] = w / A[j][j];
    }
   lp4: for (j = i; j < n; j++) {
       w = A[i][j];
       lp5: for (k = 0; k < i; k++) {
          w -= A[i][k] * A[k][j];
       }
       A[i][j] = w;
    }
  }

  lp6: for (i = 0; i < n; i++) {
     w = b[i];
     lp7: for (j = 0; j < i; j++)
        w -= A[i][j] * y[j];
     y[i] = w;
  }

   lp8: for (i = n-1; i >=0; i--) {
     w = y[i];
     lp9: for (j = i+1; j < n; j++)
        w -= A[i][j] * x[j];
     x[i] = w / A[i][i];
  }

}