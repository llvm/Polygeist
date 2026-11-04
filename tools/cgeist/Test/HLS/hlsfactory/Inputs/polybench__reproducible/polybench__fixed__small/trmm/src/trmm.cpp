#include "trmm.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_trmm( 
		 t_ap_fixed alpha,
		 t_ap_fixed A[ 60 + 0][60 + 0],
		 t_ap_fixed B[ 60 + 0][80 + 0])
{
  #pragma HLS top name=kernel_trmm

    const int m = 60;
    const int n = 80;

  int i, j, k;
  for (i = 0; i < m; i++)
     for (j = 0; j < n; j++) {
        for (k = i+1; k < m; k++)
           B[i][j] += A[k][i] * B[k][j];
        B[i][j] = alpha * B[i][j];
     }

}