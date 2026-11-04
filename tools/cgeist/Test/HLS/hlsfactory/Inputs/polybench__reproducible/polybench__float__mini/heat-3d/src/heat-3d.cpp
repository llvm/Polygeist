#include "heat-3d.h"


void kernel_heat_3d(
		      
		      double A[ 10 + 0][10 + 0][10 + 0],
		      double B[ 10 + 0][10 + 0][10 + 0])
{
  #pragma HLS top name=kernel_heat_3d

    const int n = 10;
    const int tsteps = 20;

  int t, i, j, k;

    lp1: for (t = 1; t <= 20; t++) {
        lp2: for (i = 1; i < n-1; i++) {
            lp3: for (j = 1; j < n-1; j++) {
                lp4: for (k = 1; k < n-1; k++) {
                    B[i][j][k] =   0.125 * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                                 + 0.125 * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                                 + 0.125 * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                                 + A[i][j][k];
                }
            }
        }
        lp5: for (i = 1; i < n-1; i++) {
           lp6: for (j = 1; j < n-1; j++) {
               lp7: for (k = 1; k < n-1; k++) {
                   A[i][j][k] =   0.125 * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
                                + 0.125 * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
                                + 0.125 * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
                                + B[i][j][k];
               }
           }
       }
    }

}