#include "durbin.h"


void kernel_durbin(
		   double r[ 40 + 0],
		   double y[ 40 + 0])
{
  #pragma HLS top name=kernel_durbin

    const int n = 40;

 double z[40];
 double alpha;
 double beta;
 double sum;

 int i,k;

 y[0] = -r[0];
 beta = 1.0;
 alpha = -r[0];

 lp1: for (k = 1; k < n; k++) {
   beta = (1-alpha*alpha)*beta;
   sum = 0.0;
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