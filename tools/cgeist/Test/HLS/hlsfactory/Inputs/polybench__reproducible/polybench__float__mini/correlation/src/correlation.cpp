#include "correlation.h"


void kernel_correlation( 
			double float_n,
			double data[ 32 + 0][28 + 0],
			double corr[ 28 + 0][28 + 0],
			double mean[ 28 + 0],
			double stddev[ 28 + 0])
{
  #pragma HLS top name=kernel_correlation

    const int n = 32;
    const int m = 28;

  int i, j, k;

  double eps = 0.1;


  lp1: for (j = 0; j < m; j++)
    {
      mean[j] = 0.0;
      lp2: for (i = 0; i < n; i++)
	mean[j] += data[i][j];
      mean[j] /= float_n;
    }


   lp3: for (j = 0; j < m; j++)
    {
      stddev[j] = 0.0;
      lp4: for (i = 0; i < n; i++)
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = sqrt(stddev[j]);


      stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
    }


  lp5: for (i = 0; i < n; i++)
    lp6: for (j = 0; j < m; j++)
      {
        data[i][j] -= mean[j];
        data[i][j] /= sqrt(float_n) * stddev[j];
      }


  lp7: for (i = 0; i < m-1; i++)
    {
      corr[i][i] = 1.0;
      lp8: for (j = i+1; j < m; j++)
        {
          corr[i][j] = 0.0;
          lp9: for (k = 0; k < n; k++)
            corr[i][j] += (data[k][i] * data[k][j]);
          corr[j][i] = corr[i][j];
        }
    }
  corr[m-1][m-1] = 1.0;

}