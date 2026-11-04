// Conv 1x1 PE

#include "net_hls.h"
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"

FIX_FM_acc compute_engine_16_new(FIX_WT w0, FIX_FM b0,
	FIX_WT w1, FIX_FM b1,
	FIX_WT w2, FIX_FM b2,
	FIX_WT w3, FIX_FM b3,
	FIX_WT w4, FIX_FM b4,
	FIX_WT w5, FIX_FM b5,
	FIX_WT w6, FIX_FM b6,
	FIX_WT w7, FIX_FM b7,
	FIX_WT w8, FIX_FM b8,
	FIX_WT w9, FIX_FM b9,
	FIX_WT w10, FIX_FM b10,
	FIX_WT w11, FIX_FM b11,
	FIX_WT w12, FIX_FM b12,
	FIX_WT w13, FIX_FM b13,
	FIX_WT w14, FIX_FM b14,
	FIX_WT w15, FIX_FM b15)
{
	FIX_32_10 mul0, mul1, mul2, mul3, mul4, mul5, mul6, mul7;
	FIX_32_10 mul8, mul9, mul10, mul11, mul12, mul13, mul14, mul15;
	FIX_32_10 add0, add1, add2, add3, add4, add5, add6;
	FIX_32_10 add7, add8, add9, add10, add11, add12, add13, add14;

	mul0 = w0 * b0;
	mul1 = w1 * b1;
	mul2 = w2 * b2;
	mul3 = w3 * b3;
	mul4 = w4 * b4;
	mul5 = w5 * b5;
	mul6 = w6 * b6;
	mul7 = w7 * b7;
	mul8 = w8 * b8;
	mul9 = w9 * b9;
	mul10 = w10 * b10;
	mul11 = w11 * b11;
	mul12 = w12 * b12;
	mul13 = w13 * b13;
	mul14 = w14 * b14;
	mul15 = w15 * b15;

	add0 = mul0 + mul1;
	add1 = mul2 + mul3;
	add2 = mul4 + mul5;
	add3 = mul6 + mul7;
	add4 = mul8 + mul9;
	add5 = mul10 + mul11;
	add6 = mul12 + mul13;
	add7 = mul14 + mul15;

	add8 = add0 + add1;
	add9 = add2 + add3;
	add10 = add4 + add5;
	add11 = add6 + add7;

	add12 = add8 + add9;
	add13 = add10 + add11;

	add14 = add12 + add13;

	return add14;

}




void load_weights_new(FIX_WT weights[32][32], FIX_WT weight_buf[32][16], int CI)
{
	for (int ci = 0; ci < 16; ci++) {
		for (int co = 0; co < 32; co++) {
#pragma HLS pipeline
			weight_buf[co][ci] = weights[co][ci + CI];
		}
	}
}

void compute_1x1(FIX_FM bottom[32][44][84], FIX_FM_acc top[32][44][84], FIX_WT weight_buf[32][16], uint8 xmin, uint8 xmax, uint8 ymin, uint8 ymax, int ci) {
	for (uint8 h = xmin; h < xmax; h++) {
		for (uint8 w = ymin; w < ymax; w++) {


			FIX_FM_acc a = 0;
			FIX_FM_acc b = 0;
			for (int coo = 0; coo < 32; coo++) {
//#pragma HLS pipeline II=1
//#pragma HLS unroll
				a = compute_engine_16_new(
					weight_buf[coo][0], bottom[ci + 0][h][w],
					weight_buf[coo][1], bottom[ci + 1][h][w],
					weight_buf[coo][2], bottom[ci + 2][h][w],
					weight_buf[coo][3], bottom[ci + 3][h][w],
					weight_buf[coo][4], bottom[ci + 4][h][w],
					weight_buf[coo][5], bottom[ci + 5][h][w],
					weight_buf[coo][6], bottom[ci + 6][h][w],
					weight_buf[coo][7], bottom[ci + 7][h][w],
					weight_buf[coo][8], bottom[ci + 8][h][w],
					weight_buf[coo][9], bottom[ci + 9][h][w],
					weight_buf[coo][10], bottom[ci + 10][h][w],
					weight_buf[coo][11], bottom[ci + 11][h][w],
					weight_buf[coo][12], bottom[ci + 12][h][w],
					weight_buf[coo][13], bottom[ci + 13][h][w],
					weight_buf[coo][14], bottom[ci + 14][h][w],
					weight_buf[coo][15], bottom[ci + 15][h][w]);
				b = top[coo][h][w] + a;
				top[coo][h][w] = b;

			}
		}
	}
}




void CONV_1x1_new(FIX_FM bottom[32][44][84],
	FIX_FM_acc top[32][44][84],
	FIX_WT weights[32][32],
	int index,uint8 mask_bound[16])
{

#pragma HLS array_partition variable=top dim=1 complete
#pragma HLS array_partition variable=bottom dim=1 complete

#pragma HLS array_partition variable=weights dim=1 complete
#pragma HLS array_partition variable=mask_bound dim=1 complete


	FIX_WT weight_buf[32][16];
#pragma HLS array_partition variable=weight_buf dim=1 complete
#pragma HLS array_partition variable=weight_buf dim=2 complete

	
	for (int ci = 0; ci < 32; ci += 16) {

		load_weights_new(weights, weight_buf, ci);

		int buf_idx = ci / 16;
		if (index == 0) {
			compute_1x1(bottom, top, weight_buf, 1, 43, 1, 83, ci);
		}
		if (index == 1) {
			compute_1x1(bottom, top, weight_buf, mask_bound[0] + 1, mask_bound[1] + 1, mask_bound[2] + 1, mask_bound[3] + 1, ci);
			compute_1x1(bottom, top, weight_buf, mask_bound[4] + 1, mask_bound[5] + 1, mask_bound[6] + 3, mask_bound[7] + 3, ci);
			compute_1x1(bottom, top, weight_buf, mask_bound[8] + 3, mask_bound[9] + 3, mask_bound[10] + 1, mask_bound[11] + 1, ci);
			compute_1x1(bottom, top, weight_buf, mask_bound[12] + 3, mask_bound[13] + 3, mask_bound[14] + 3, mask_bound[15] + 3, ci);
		}

	}

}
