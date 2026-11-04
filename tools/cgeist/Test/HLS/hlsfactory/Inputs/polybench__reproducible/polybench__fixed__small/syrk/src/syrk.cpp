#include "syrk.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_syrk( 
		 t_ap_fixed alpha,
		 t_ap_fixed beta,
		 t_ap_fixed C[ 80 + 0][80 + 0],
		 t_ap_fixed A[ 80 + 0][60 + 0])
{
  #pragma HLS top name=kernel_syrk

    const int n = 80;
    const int m = 60;

  int i, j, k;
  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (k = 0; k < m; k++) {
      for (j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
    }
  }

}