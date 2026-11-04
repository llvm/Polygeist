#include "gramschmidt.h"


void kernel_gramschmidt( 
			double A[ 20 + 0][30 + 0],
			double R[ 30 + 0][30 + 0],
			double Q[ 20 + 0][30 + 0])
{
  #pragma HLS top name=kernel_gramschmidt

    const int m = 20;
    const int n = 30;

  int i, j, k;

  double nrm;

  lp1: for (k = 0; k < n; k++)
    {
      nrm = 0.0;
      lp2: for (i = 0; i < m; i++)
        nrm += A[i][k] * A[i][k];
      R[k][k] = sqrt(nrm);
      lp3: for (i = 0; i < m; i++)
        Q[i][k] = A[i][k] / R[k][k];
      lp4: for (j = k + 1; j < n; j++)
	{
	  R[k][j] = 0.0;
	  lp5: for (i = 0; i < m; i++)
	    R[k][j] += Q[i][k] * A[i][j];
	  lp6: for (i = 0; i < m; i++)
	    A[i][j] = A[i][j] - Q[i][k] * R[k][j];
	}
    }

}