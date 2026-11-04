#include "gramschmidt.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_gramschmidt( 
			t_ap_fixed A[ 60 + 0][80 + 0],
			t_ap_fixed R[ 80 + 0][80 + 0],
			t_ap_fixed Q[ 60 + 0][80 + 0])
{
  #pragma HLS top name=kernel_gramschmidt

    const int m = 60;
    const int n = 80;

  int i, j, k;

  t_ap_fixed nrm;


  const t_ap_fixed eps = hls::nextafter(t_ap_fixed(0.0), t_ap_fixed(1.0));

  for (k = 0; k < n; k++)
    {
      nrm = (t_ap_fixed(0.0));
      for (i = 0; i < m; i++)
        nrm += A[i][k] * A[i][k];
      R[k][k] = hls::sqrt(nrm);
      if (R[k][k] == t_ap_fixed(0.0)) R[k][k] += eps;
      for (i = 0; i < m; i++)
        Q[i][k] = A[i][k] / R[k][k];
      for (j = k + 1; j < n; j++)
	{
	  R[k][j] = (t_ap_fixed(0.0));
	  for (i = 0; i < m; i++)
	    R[k][j] += Q[i][k] * A[i][j];
	  for (i = 0; i < m; i++)
	    A[i][j] = A[i][j] - Q[i][k] * R[k][j];
	}
    }

}