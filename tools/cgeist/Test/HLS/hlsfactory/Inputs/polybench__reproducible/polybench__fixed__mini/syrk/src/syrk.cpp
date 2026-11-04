#include "syrk.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_syrk( 
		 t_ap_fixed alpha,
		 t_ap_fixed beta,
		 t_ap_fixed C[ 30 + 0][30 + 0],
		 t_ap_fixed A[ 30 + 0][20 + 0])
{
  #pragma HLS top name=kernel_syrk

    const int n = 30;
    const int m = 20;

  int i, j, k;
  lp1: for (i = 0; i < n; i++) {
    lp2: for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    lp3: for (k = 0; k < m; k++) {
      lp4: for (j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
    }
  }

}