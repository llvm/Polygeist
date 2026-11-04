#include "correlation.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_correlation( 
			t_ap_fixed float_n,
			t_ap_fixed data[ 100 + 0][80 + 0],
			t_ap_fixed corr[ 80 + 0][80 + 0],
			t_ap_fixed mean[ 80 + 0],
			t_ap_fixed stddev[ 80 + 0])
{
  #pragma HLS top name=kernel_correlation

    const int n = 100;
    const int m = 80;

  int i, j, k;

  t_ap_fixed eps = (t_ap_fixed(0.1));


  for (j = 0; j < m; j++)
    {
      mean[j] = (t_ap_fixed(0.0));
      for (i = 0; i < n; i++)
	mean[j] += data[i][j];
      mean[j] /= float_n;
    }


   for (j = 0; j < m; j++)
    {
      stddev[j] = (t_ap_fixed(0.0));
      for (i = 0; i < n; i++)
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = hls::sqrt(stddev[j]);


      stddev[j] = stddev[j] <= eps ? (t_ap_fixed(1.0)) : stddev[j];
    }


  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      {
        data[i][j] -= mean[j];
        data[i][j] /= hls::sqrt(float_n) * stddev[j];
      }


  for (i = 0; i < m-1; i++)
    {
      corr[i][i] = (t_ap_fixed(1.0));
      for (j = i+1; j < m; j++)
        {
          corr[i][j] = (t_ap_fixed(0.0));
          for (k = 0; k < n; k++)
            corr[i][j] += (data[k][i] * data[k][j]);
          corr[j][i] = corr[i][j];
        }
    }
  corr[m-1][m-1] = (t_ap_fixed(1.0));

}