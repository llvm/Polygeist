// dw_conv 3x3

#include "net_hls.h"
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include "hls_math.h"

//using namespace hls;
//using namespace std;

inline FIX_FM relu_single(FIX_FM d) {
	if (d > 6)
		return 6;
	if (d < 0)
		return 0;
	return d;
}

void compute_3x3(FIX_FM bottom[32][44][84], FIX_FM top[32][44][84], FIX_WT weights[32][3][3], uint8 xmin, uint8 xmax, uint8 ymin, uint8 ymax) {
//#pragma HLS array_partition variable=top dim=1 complete
//#pragma HLS array_partition variable=bottom dim=1 complete
//#pragma HLS array_partition variable=weights dim=1 complete
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (uint8 h =xmin ; h <xmax ; h++) {
				for (uint8 w = ymin; w < ymax; w++) {


					FIX_FM a = 0;
					FIX_FM b = 0;
					for (uint8 co = 0; co < 32; co++) {
//#pragma HLS pipeline II=1						
//#pragma HLS unroll
						    a = weights[co][i][j] * bottom[co][h + i - 1][w +j - 1];
						    b = top[co][h][w] + a;
						    top[co][h][w] = b;
					}
					}
				}

		}
	}
}

void compute_relu(FIX_FM top[32][44][84], uint8 xmin, uint8 xmax, uint8 ymin, uint8 ymax) {
	for (uint8 h = xmin; h < xmax; h++) {
		for (uint8 w = ymin; w < ymax; w++) {

			for (int co = 0; co < 32; co++) {
//#pragma HLS pipeline II = 1
				top[co][h][w] = relu_single(top[co][h][w]);

			}
		}
	}
}


void DW_CONV_3x3_new(FIX_FM bottom[32][44][84],
	FIX_FM top[32][44][84],
	FIX_WT weights[32][3][3],
	FIX_WT bias[32],
	int relu, int index, int sigmoid,uint8 mask_bound[16])
{

//#pragma HLS array_partition variable=top dim=1 complete
//#pragma HLS array_partition variable=bottom dim=1 complete
//#pragma HLS array_partition variable=weights dim=1 complete
//#pragma HLS array_partition variable=bias dim=1 complete

//#pragma HLS array_partition variable=mask_bound dim=1 complete
	

	if (relu == 1) {
		compute_3x3(bottom, top, weights, 1, 43, 1, 83);
		compute_relu(top, 1, 43, 1, 83);
	}

	if (index == 1) {
		compute_3x3(bottom, top, weights, mask_bound[0] + 1, mask_bound[1] + 1, mask_bound[2] + 1, mask_bound[3] + 1);
		compute_3x3(bottom, top, weights, mask_bound[4] + 1, mask_bound[5] + 1, mask_bound[6] + 3, mask_bound[7] + 3);
		compute_3x3(bottom, top, weights, mask_bound[8] + 3, mask_bound[9] + 3, mask_bound[10] + 1, mask_bound[11] + 1);
		compute_3x3(bottom, top, weights, mask_bound[12] + 3, mask_bound[13] + 3, mask_bound[14] + 3, mask_bound[15] + 3);
	}

	if (sigmoid == 1) {
		compute_3x3(bottom, top, weights, 1, 13, 1, 13);
	}
	
	

}



	
