#include "doitgen.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_doitgen(  
		    t_ap_fixed A[ 10 + 0][8 + 0][12 + 0],
		    t_ap_fixed C4[ 12 + 0][12 + 0],
		    t_ap_fixed sum[ 12 + 0])
{
  #pragma HLS top name=kernel_doitgen

    const int nr = 10;
    const int nq = 8;
    const int np = 12;

  int r, q, p, s;

  lp1: for (r = 0; r < nr; r++)
    lp2: for (q = 0; q < nq; q++)  {
      lp3: for (p = 0; p < np; p++)  {
	sum[p] = (t_ap_fixed(0.0));
	lp4: for (s = 0; s < np; s++)
	  sum[p] += A[r][q][s] * C4[s][p];
      }
      lp5: for (p = 0; p < np; p++)
	A[r][q][p] = sum[p];
    }

}