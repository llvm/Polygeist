#include "symm.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_symm( 
		 t_ap_fixed alpha,
		 t_ap_fixed beta,
		 t_ap_fixed C[ 20 + 0][30 + 0],
		 t_ap_fixed A[ 20 + 0][20 + 0],
		 t_ap_fixed B[ 20 + 0][30 + 0])
{
  #pragma HLS top name=kernel_symm

    const int m = 20;
    const int n = 30;

  int i, j, k;
  t_ap_fixed temp2;
   lp1: for (i = 0; i < m; i++)
      lp2: for (j = 0; j < n; j++ )
      {
        temp2 = 0;
        lp3: for (k = 0; k < i; k++) {
           C[k][j] += alpha*B[i][j] * A[i][k];
           temp2 += B[k][j] * A[i][k];
        }
        C[i][j] = beta * C[i][j] + alpha*B[i][j] * A[i][i] + alpha * temp2;
     }

}