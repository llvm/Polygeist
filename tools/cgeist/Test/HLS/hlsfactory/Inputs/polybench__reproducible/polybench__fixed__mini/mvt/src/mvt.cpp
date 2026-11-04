#include "mvt.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_mvt(
		t_ap_fixed x1[ 40 + 0],
		t_ap_fixed x2[ 40 + 0],
		t_ap_fixed y_1[ 40 + 0],
		t_ap_fixed y_2[ 40 + 0],
		t_ap_fixed A[ 40 + 0][40 + 0])
{
  #pragma HLS top name=kernel_mvt

    const int n = 40;

  int i, j;

  lp1: for (i = 0; i < n; i++)
    lp2: for (j = 0; j < n; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  lp3: for (i = 0; i < n; i++)
    lp4: for (j = 0; j < n; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];

}