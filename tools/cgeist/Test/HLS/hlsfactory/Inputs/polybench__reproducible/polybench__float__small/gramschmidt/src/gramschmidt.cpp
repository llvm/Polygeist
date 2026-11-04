#include "gramschmidt.h"


void kernel_gramschmidt( 
			double A[ 60 + 0][80 + 0],
			double R[ 80 + 0][80 + 0],
			double Q[ 60 + 0][80 + 0])
{
  #pragma HLS top name=kernel_gramschmidt

    const int m = 60;
    const int n = 80;

  int i, j, k;

  double nrm;

  for (k = 0; k < n; k++)
    {
      nrm = 0.0;
      for (i = 0; i < m; i++)
        nrm += A[i][k] * A[i][k];
      R[k][k] = sqrt(nrm);
      for (i = 0; i < m; i++)
        Q[i][k] = A[i][k] / R[k][k];
      for (j = k + 1; j < n; j++)
	{
	  R[k][j] = 0.0;
	  for (i = 0; i < m; i++)
	    R[k][j] += Q[i][k] * A[i][j];
	  for (i = 0; i < m; i++)
	    A[i][j] = A[i][j] - Q[i][k] * R[k][j];
	}
    }

}