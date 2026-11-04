#include "correlation.h"


void kernel_correlation( 
			double float_n,
			double data[ 260 + 0][240 + 0],
			double corr[ 240 + 0][240 + 0],
			double mean[ 240 + 0],
			double stddev[ 240 + 0])
{
  #pragma HLS top name=kernel_correlation

    const int n = 260;
    const int m = 240;

  int i, j, k;

  double eps = 0.1;


  for (j = 0; j < m; j++)
    {
      mean[j] = 0.0;
      for (i = 0; i < n; i++)
	mean[j] += data[i][j];
      mean[j] /= float_n;
    }


   for (j = 0; j < m; j++)
    {
      stddev[j] = 0.0;
      for (i = 0; i < n; i++)
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = sqrt(stddev[j]);


      stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
    }


  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      {
        data[i][j] -= mean[j];
        data[i][j] /= sqrt(float_n) * stddev[j];
      }


  for (i = 0; i < m-1; i++)
    {
      corr[i][i] = 1.0;
      for (j = i+1; j < m; j++)
        {
          corr[i][j] = 0.0;
          for (k = 0; k < n; k++)
            corr[i][j] += (data[k][i] * data[k][j]);
          corr[j][i] = corr[i][j];
        }
    }
  corr[m-1][m-1] = 1.0;

}