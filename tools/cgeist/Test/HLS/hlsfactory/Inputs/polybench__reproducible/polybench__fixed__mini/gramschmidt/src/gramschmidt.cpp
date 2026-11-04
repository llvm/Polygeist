#include "gramschmidt.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_gramschmidt( 
			t_ap_fixed A[ 20 + 0][30 + 0],
			t_ap_fixed R[ 30 + 0][30 + 0],
			t_ap_fixed Q[ 20 + 0][30 + 0])
{
  #pragma HLS top name=kernel_gramschmidt

    const int m = 20;
    const int n = 30;

  int i, j, k;

  t_ap_fixed nrm;


  const t_ap_fixed eps = hls::nextafter(t_ap_fixed(0.0), t_ap_fixed(1.0));

  lp1: for (k = 0; k < n; k++)
    {
      nrm = (t_ap_fixed(0.0));
      lp2: for (i = 0; i < m; i++)
        nrm += A[i][k] * A[i][k];
      R[k][k] = hls::sqrt(nrm);
      if (R[k][k] == t_ap_fixed(0.0)) R[k][k] += eps;
      lp3: for (i = 0; i < m; i++)
        Q[i][k] = A[i][k] / R[k][k];
      lp4: for (j = k + 1; j < n; j++)
	{
	  R[k][j] = (t_ap_fixed(0.0));
	  lp5: for (i = 0; i < m; i++)
	    R[k][j] += Q[i][k] * A[i][j];
	  lp6: for (i = 0; i < m; i++)
	    A[i][j] = A[i][j] - Q[i][k] * R[k][j];
	}
    }

}