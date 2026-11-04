#include "lu.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_lu(
	       t_ap_fixed A[ 40 + 0][40 + 0])
{
  #pragma HLS top name=kernel_lu

    const int n = 40;

  int i, j, k;

  lp1: for (i = 0; i < n; i++) {
    lp2: for (j = 0; j <i; j++) {
       lp3: for (k = 0; k < j; k++) {
          A[i][j] -= A[i][k] * A[k][j];
       }
        A[i][j] /= A[j][j];
    }
   lp4: for (j = i; j < n; j++) {
       lp5: for (k = 0; k < i; k++) {
          A[i][j] -= A[i][k] * A[k][j];
       }
    }
  }
}