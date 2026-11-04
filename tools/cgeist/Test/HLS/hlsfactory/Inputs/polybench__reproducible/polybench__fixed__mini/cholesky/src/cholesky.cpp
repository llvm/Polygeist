#include "cholesky.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_cholesky(
		     t_ap_fixed A[ 40 + 0][40 + 0])
{
  #pragma HLS top name=kernel_cholesky

    const int n = 40;

  int i, j, k;


  lp1: for (i = 0; i < n; i++) {

     lp2: for (j = 0; j < i; j++) {
        lp3: for (k = 0; k < j; k++) {
           A[i][j] -= A[i][k] * A[j][k];
        }
        A[i][j] /= A[j][j];
     }

     lp4: for (k = 0; k < i; k++) {
        A[i][i] -= A[i][k] * A[i][k];
     }
     A[i][i] = hls::sqrt(A[i][i]);
  }

}