#include "fdtd-2d.h"


void kernel_fdtd_2d(
		    
		    
		    double ex[ 60 + 0][80 + 0],
		    double ey[ 60 + 0][80 + 0],
		    double hz[ 60 + 0][80 + 0],
		    double _fict_[ 40 + 0])
{
  #pragma HLS top name=kernel_fdtd_2d

    const int tmax = 40;
    const int nx = 60;
    const int ny = 80;

  int t, i, j;


  for(t = 0; t < tmax; t++)
    {
      for (j = 0; j < ny; j++)
	ey[0][j] = _fict_[t];
      for (i = 1; i < nx; i++)
	for (j = 0; j < ny; j++)
	  ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]);
      for (i = 0; i < nx; i++)
	for (j = 1; j < ny; j++)
	  ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]);
      for (i = 0; i < nx - 1; i++)
	for (j = 0; j < ny - 1; j++)
	  hz[i][j] = hz[i][j] - 0.7*  (ex[i][j+1] - ex[i][j] +
				       ey[i+1][j] - ey[i][j]);
    }

}