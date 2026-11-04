#include "fdtd-2d.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_fdtd_2d(
		    
		    
		    t_ap_fixed ex[ 200 + 0][240 + 0],
		    t_ap_fixed ey[ 200 + 0][240 + 0],
		    t_ap_fixed hz[ 200 + 0][240 + 0],
		    t_ap_fixed _fict_[ 100 + 0])
{
  #pragma HLS top name=kernel_fdtd_2d

    const int tmax = 100;
    const int nx = 200;
    const int ny = 240;

  int t, i, j;


  for(t = 0; t < tmax; t++)
    {
      for (j = 0; j < ny; j++)
	ey[0][j] = _fict_[t];
      for (i = 1; i < nx; i++)
	for (j = 0; j < ny; j++)
	  ey[i][j] = ey[i][j] - (t_ap_fixed(0.5))*(hz[i][j]-hz[i-1][j]);
      for (i = 0; i < nx; i++)
	for (j = 1; j < ny; j++)
	  ex[i][j] = ex[i][j] - (t_ap_fixed(0.5))*(hz[i][j]-hz[i][j-1]);
      for (i = 0; i < nx - 1; i++)
	for (j = 0; j < ny - 1; j++)
	  hz[i][j] = hz[i][j] - (t_ap_fixed(0.7))*  (ex[i][j+1] - ex[i][j] +
				       ey[i+1][j] - ey[i][j]);
    }

}