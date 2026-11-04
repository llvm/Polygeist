// conv5x5

#include "net_hls.h"
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"


void compute_DW_CONV_5x5(FIX_FM bottom[32][44][84], FIX_FM_acc top[32][44][84], FIX_WT weights[32][5][5],  int imin, int imax, int jmin, int jmax, int hmin, int hmax, int wmin, int wmax, int offsetx, int offsety) {
//#pragma HLS array_partition variable=top dim=1 complete
//#pragma HLS array_partition variable=bottom dim=1 complete
//#pragma HLS array_partition variable=weights dim=1 complete
	for (int i = imin; i < imax; i++) {
		for (int j = jmin; j < jmax; j++) {
			for (int h = hmin; h <= hmax; h++) {
				for (int w = wmin; w <= wmax; w++) {

					FIX_FM_acc a = 0;
					FIX_FM_acc b = 0;
					for (int co = 0; co < 32; co++) {
//#pragma HLS pipeline II=1
//#pragma HLS unroll
						a = weights[co][i][j] * bottom[co][h + offsetx*22 + i - 2][w + offsety*42 + j - 2];
						b = top[co][h + offsetx*22][w + offsety*42] + a;
						top[co][h + offsetx*22][w + offsety*42] = b;
					}
				}
			}
		}
	}
}


void DW_CONV_5x5(FIX_FM bottom[32][44][84],
	FIX_FM_acc top[32][44][84],
	FIX_WT weights[32][5][5],
	FIX_WT bias[32],
	int offsetx,
	int offsety
	)
{

//#pragma HLS array_partition variable=top dim=1 complete
//#pragma HLS array_partition variable=bottom dim=1 complete
//#pragma HLS array_partition variable=weights dim=1 complete
//#pragma HLS array_partition variable=bias dim=1 complete

	compute_DW_CONV_5x5(bottom, top, weights, 0, 5, 0, 5, 2 + offsetx * 22, 19 + offsetx * 22, 2 + offsety * 42, 39 + offsety * 42, 0, 0);
	compute_DW_CONV_5x5(bottom, top, weights, 1, 5, 0, 5, 1, 1, 2 + offsety * 42, 39 + offsety * 42, offsetx, 0);
	compute_DW_CONV_5x5(bottom, top, weights, 0, 4, 0, 5, 20, 20, 2 + offsety * 42, 39 + offsety * 42, offsetx, 0);
	compute_DW_CONV_5x5(bottom, top, weights, 0, 5, 1, 5, 2 + offsetx * 22, 19 + offsetx * 22, 1, 1, 0, offsety);
	compute_DW_CONV_5x5(bottom, top, weights, 0, 5, 0, 4, 2 + offsetx * 22, 19 + offsetx * 22, 40, 40, 0, offsety);
	compute_DW_CONV_5x5(bottom, top, weights, 1, 5, 1, 5, 1, 1, 1, 1, offsetx, offsety);
	compute_DW_CONV_5x5(bottom, top, weights, 0, 4, 1, 5, 20, 20, 1, 1, offsetx, offsety);
	compute_DW_CONV_5x5(bottom, top, weights, 1, 5, 0, 4, 1, 1, 40, 40, offsetx, offsety);
	compute_DW_CONV_5x5(bottom, top, weights, 0, 4, 0, 4, 20, 20, 40, 40, offsetx, offsety);
	
}
