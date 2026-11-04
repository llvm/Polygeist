#include "covariance.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_covariance( 
		       t_ap_fixed float_n,
		       t_ap_fixed data[ 100 + 0][80 + 0],
		       t_ap_fixed cov[ 80 + 0][80 + 0],
		       t_ap_fixed mean[ 80 + 0])
{
  #pragma HLS top name=kernel_covariance

    const int n = 100;
    const int m = 80;

  int i, j, k;

  for (j = 0; j < m; j++)
    {
      mean[j] = (t_ap_fixed(0.0));
      for (i = 0; i < n; i++)
        mean[j] += data[i][j];
      mean[j] /= float_n;
    }

  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      data[i][j] -= mean[j];

  for (i = 0; i < m; i++)
    for (j = i; j < m; j++)
      {
        cov[i][j] = (t_ap_fixed(0.0));
        for (k = 0; k < n; k++)
	  cov[i][j] += data[k][i] * data[k][j];
        cov[i][j] /= (float_n - (t_ap_fixed(1.0)));
        cov[j][i] = cov[i][j];
      }

}