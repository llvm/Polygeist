#include "correlation.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_correlation( 
			t_ap_fixed float_n,
			t_ap_fixed data[ 32 + 0][28 + 0],
			t_ap_fixed corr[ 28 + 0][28 + 0],
			t_ap_fixed mean[ 28 + 0],
			t_ap_fixed stddev[ 28 + 0])
{
  #pragma HLS top name=kernel_correlation

    const int n = 32;
    const int m = 28;

  int i, j, k;

  t_ap_fixed eps = (t_ap_fixed(0.1));


  lp1: for (j = 0; j < m; j++)
    {
      mean[j] = (t_ap_fixed(0.0));
      lp2: for (i = 0; i < n; i++)
	mean[j] += data[i][j];
      mean[j] /= float_n;
    }


   lp3: for (j = 0; j < m; j++)
    {
      stddev[j] = (t_ap_fixed(0.0));
      lp4: for (i = 0; i < n; i++)
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = hls::sqrt(stddev[j]);


      stddev[j] = stddev[j] <= eps ? (t_ap_fixed(1.0)) : stddev[j];
    }


  lp5: for (i = 0; i < n; i++)
    lp6: for (j = 0; j < m; j++)
      {
        data[i][j] -= mean[j];
        data[i][j] /= hls::sqrt(float_n) * stddev[j];
      }


  lp7: for (i = 0; i < m-1; i++)
    {
      corr[i][i] = (t_ap_fixed(1.0));
      lp8: for (j = i+1; j < m; j++)
        {
          corr[i][j] = (t_ap_fixed(0.0));
          lp9: for (k = 0; k < n; k++)
            corr[i][j] += (data[k][i] * data[k][j]);
          corr[j][i] = corr[i][j];
        }
    }
  corr[m-1][m-1] = (t_ap_fixed(1.0));

}