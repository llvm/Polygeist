#include "symm.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_symm( 
		 t_ap_fixed alpha,
		 t_ap_fixed beta,
		 t_ap_fixed C[ 200 + 0][240 + 0],
		 t_ap_fixed A[ 200 + 0][200 + 0],
		 t_ap_fixed B[ 200 + 0][240 + 0])
{
  #pragma HLS top name=kernel_symm

    const int m = 200;
    const int n = 240;

  int i, j, k;
  t_ap_fixed temp2;
   for (i = 0; i < m; i++)
      for (j = 0; j < n; j++ )
      {
        temp2 = 0;
        for (k = 0; k < i; k++) {
           C[k][j] += alpha*B[i][j] * A[i][k];
           temp2 += B[k][j] * A[i][k];
        }
        C[i][j] = beta * C[i][j] + alpha*B[i][j] * A[i][i] + alpha * temp2;
     }

}