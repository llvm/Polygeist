#include "doitgen.h"


void kernel_doitgen(  
		    double A[ 50 + 0][40 + 0][60 + 0],
		    double C4[ 60 + 0][60 + 0],
		    double sum[ 60 + 0])
{
  #pragma HLS top name=kernel_doitgen

    const int nr = 50;
    const int nq = 40;
    const int np = 60;

  int r, q, p, s;

  for (r = 0; r < nr; r++)
    for (q = 0; q < nq; q++)  {
      for (p = 0; p < np; p++)  {
	sum[p] = 0.0;
	for (s = 0; s < np; s++)
	  sum[p] += A[r][q][s] * C4[s][p];
      }
      for (p = 0; p < np; p++)
	A[r][q][p] = sum[p];
    }

}