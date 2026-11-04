#include "durbin.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_durbin(
		   t_ap_fixed r[ 40 + 0],
		   t_ap_fixed y[ 40 + 0])
{
  #pragma HLS top name=kernel_durbin

    const int n = 40;

 t_ap_fixed z[40];
 t_ap_fixed alpha;
 t_ap_fixed beta;
 t_ap_fixed sum;

 int i,k;

 y[0] = -r[0];
 beta = (t_ap_fixed(1.0));
 alpha = -r[0];

 lp1: for (k = 1; k < n; k++) {
   beta = (1-alpha*alpha)*beta;
   sum = (t_ap_fixed(0.0));
   lp2: for (i=0; i<k; i++) {
      sum += r[k-i-1]*y[i];
   }
   alpha = - (r[k] + sum)/beta;

   lp3: for (i=0; i<k; i++) {
      z[i] = y[i] + alpha*y[k-i-1];
   }
   lp4: for (i=0; i<k; i++) {
     y[i] = z[i];
   }
   y[k] = alpha;
 }

}