

// dw_conv 3x3

#include "net_hls.h"
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"


inline FIX_FM relu_single( FIX_FM d ) {
        if( d > 6 )
                return 6;
        if( d < 0 )
                return 0;
        return d;
}


void DW_CONV_3x3(FIX_FM bottom[32][44][84],
					FIX_FM top[32][44][84],
					FIX_WT weights[32][3][3],
					FIX_WT bias[32],
					int relu)
{

#pragma HLS array_partition variable=top dim=1 complete
#pragma HLS array_partition variable=bottom dim=1 complete
#pragma HLS array_partition variable=weights dim=1 complete
#pragma HLS array_partition variable=bias dim=1 complete


	for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 3; j++) {
			for(int h = 1; h <= 42; h++){
				for(int w = 1; w <= 82; w++){
#pragma HLS pipeline
					for(int co = 0; co < 32; co++){
#pragma HLS unroll
						top[co][h][w] += (weights[co][i][j] * bottom[co][h+i-1][w+j-1]);
					}
				}
			}
		}
	}

	if(relu == 1) {
		for(int h = 1; h <= 42; h+=2){
			for(int w = 1; w <= 82; w++){
	#pragma HLS pipeline
				for(int co = 0; co < 32; co++){
					top[co][h  ][w] = relu_single( top[co][h  ][w ]);
					top[co][h+1][w] = relu_single( top[co][h+1][w ]);
				}
			}
		}
	}
}


