#include "utils.h"
#include <hls_math.h>


void maxpool2D (
	fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
	fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH]
)
{
#pragma HLS inline off

//#pragma HLS array_partition variable = X_buf dim = 2


//#pragma HLS array_partition variable = Y_buf dim = 2


			OD:
			for (int c = 0; c < OUT_BUF_DEPTH; c++)
			{
	OW:
	for (int w = 0; w < OUT_BUF_WIDTH; w++)
			{
		//#pragma HLS pipeline II=1
	  	OH:
		for (int h = 0; h < OUT_BUF_HEIGHT; h++)
		{
		//#pragma HLS unroll	
				Y_buf[c][h][w] = hls::max(hls::max(X_buf[c][h*STRIDE][w*STRIDE],X_buf[c][h*STRIDE][w*STRIDE + 1]),hls::max(X_buf[c][h*STRIDE+1][w*STRIDE],X_buf[c][h*STRIDE + 1][w*STRIDE+1]));
			}
		}
	}

}
