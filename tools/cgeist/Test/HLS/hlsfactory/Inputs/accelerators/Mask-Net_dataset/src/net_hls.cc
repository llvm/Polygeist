#include "net_hls.h"
#include <math.h>
#include <fstream>
#include <hls_math.h>
#include <ap_fixed.h>

#include <string.h>

using namespace std;
extern FILE* fo;

// feature map buffers
FIX_FM FM_buf1[32][44][84];
FIX_FM FM_buf2[32][44][84];
FIX_FM FM_buf3[32][44][84];
FIX_FM FM_buf4[32][44][84];
FIX_FM_acc FM_buf_acc[32][44][84];

//confidence mask buffer
FIX_FM mask[13][13];


// weight buffers
FIX_WT weight_buf_1x1[4][32][32];
FIX_WT weight_buf_3x3[4][32][3][3];
FIX_WT bias_buf[4][32];
FIX_WT weight_buf_5x5[3][32][5][5];

void compute_bounding_box(float predict_box[4][5], int constant[4][3],uint8 mask_bound[16])
{
	FIX_32_4 conf_thresh = -100.0;
	int conf_j = 0;
	int conf_m = 0;
	int conf_n = 0;
	FIX_32_4 conf_box1 = 0.0;
	FIX_32_4 conf_box2 = 0.0;
	//int h = 20;
	//int w = 40;
	uint8 xmin1 = mask_bound[0] + 1;
	uint8 xmax1 = mask_bound[1] + 1;
	uint8 ymin1 = mask_bound[2] + 1;
	uint8 ymax1 = mask_bound[3] + 1;

	uint8 xmin2 = mask_bound[4] + 1;
	uint8 xmax2 = mask_bound[5] + 1;
	uint8 ymin2 = mask_bound[6] + 3;
	uint8 ymax2 = mask_bound[7] + 3;

	uint8 xmin3 = mask_bound[8] + 3;
	uint8 xmax3 = mask_bound[9] + 3;
	uint8 ymin3 = mask_bound[10] + 1;
	uint8 ymax3 = mask_bound[11] + 1;

	uint8 xmin4 = mask_bound[12] + 3;
	uint8 xmax4 = mask_bound[13] + 3;
	uint8 ymin4 = mask_bound[14] + 3;
	uint8 ymax4 = mask_bound[15] + 3;

	//// 0
	for (uint8 m =xmin1 ; m < xmax1; m++) {
		for (uint8 n = ymin1; n < ymax1; n++) {
			
			conf_box1 = FM_buf_acc[4][m][n];
			if (conf_box1 > conf_thresh) {
				conf_thresh = conf_box1;
				conf_j = 0;
				conf_m = m;
				conf_n = n;

			}
		}
	}

	for (uint8 m = xmin1; m < xmax1; m++) {
		for (uint8 n = ymin1; n < ymax1; n++) {
			
			conf_box2 = FM_buf_acc[9][m][n];
			if (conf_box2 > conf_thresh) {
				conf_thresh = conf_box2;
				conf_j = 1;
				conf_m = m;
				conf_n = n;
			}
		}
	}

	if (conf_j == 0) {
		// first bounding box
		predict_box[0][0] = FM_buf_acc[0][conf_m][conf_n];
		predict_box[0][1] = FM_buf_acc[1][conf_m][conf_n];
		predict_box[0][2] = FM_buf_acc[2][conf_m][conf_n];
		predict_box[0][3] = FM_buf_acc[3][conf_m][conf_n];
		predict_box[0][4] = conf_thresh;

	}
	else if (conf_j == 1) {
		// second bounding box
		predict_box[0][0] = FM_buf_acc[5][conf_m][conf_n];
		predict_box[0][1] = FM_buf_acc[6][conf_m][conf_n];
		predict_box[0][2] = FM_buf_acc[7][conf_m][conf_n];
		predict_box[0][3] = FM_buf_acc[8][conf_m][conf_n];
		predict_box[0][4] = conf_thresh;
	}

	constant[0][0] = conf_j;
	constant[0][1] = conf_n - 1;
	constant[0][2] = conf_m - 1;

	//// 1
	conf_thresh = -100.0;
	conf_j = 0;
	conf_m = 0;
	conf_n = 0;
	conf_box1 = 0.0;
	conf_box2 = 0.0;

	for (int m = xmin2; m < xmax2; m++) {
		for (int n = ymin2; n < ymax2; n++) {
			
			conf_box1 = FM_buf_acc[4][m][n];
			if (conf_box1 > conf_thresh) {
				conf_thresh = conf_box1;
				conf_j = 0;
				conf_m = m;
				conf_n = n;

			}
		}
	}

	for (int m = xmin2; m < xmax2; m++) {
		for (int n = ymin2; n < ymax2; n++) {
			
			conf_box2 = FM_buf_acc[9][m][n];
			if (conf_box2 > conf_thresh) {
				conf_thresh = conf_box2;
				conf_j = 1;
				conf_m = m;
				conf_n = n;
			}
		}
	}

	if (conf_j == 0) {
		// first bounding box
		predict_box[1][0] = FM_buf_acc[0][conf_m][conf_n];
		predict_box[1][1] = FM_buf_acc[1][conf_m][conf_n];
		predict_box[1][2] = FM_buf_acc[2][conf_m][conf_n];
		predict_box[1][3] = FM_buf_acc[3][conf_m][conf_n];
		predict_box[1][4] = conf_thresh;

	}
	else if (conf_j == 1) {
		// second bounding box
		predict_box[1][0] = FM_buf_acc[5][conf_m][conf_n];
		predict_box[1][1] = FM_buf_acc[6][conf_m][conf_n];
		predict_box[1][2] = FM_buf_acc[7][conf_m][conf_n];
		predict_box[1][3] = FM_buf_acc[8][conf_m][conf_n];
		predict_box[1][4] = conf_thresh;
	}

	constant[1][0] = conf_j;
	constant[1][1] = conf_n - 43;
	constant[1][2] = conf_m - 1;

	//// 2
	conf_thresh = -100.0;
	conf_j = 0;
	conf_m = 0;
	conf_n = 0;
	conf_box1 = 0.0;
	conf_box2 = 0.0;

	for (int m = xmin3; m < xmax3; m++) {
		for (int n = ymin3; n < ymax3; n++) {
			
			conf_box1 = FM_buf_acc[4][m][n];
			if (conf_box1 > conf_thresh) {
				conf_thresh = conf_box1;
				conf_j = 0;
				conf_m = m;
				conf_n = n;

			}
		}
	}

	for (int m = xmin3; m < xmax3; m++) {
		for (int n = ymin3; n < ymax3; n++) {
			
			conf_box2 = FM_buf_acc[9][m][n];
			if (conf_box2 > conf_thresh) {
				conf_thresh = conf_box2;
				conf_j = 1;
				conf_m = m;
				conf_n = n;
			}
		}
	}

	if (conf_j == 0) {
		// first bounding box
		predict_box[2][0] = FM_buf_acc[0][conf_m][conf_n];
		predict_box[2][1] = FM_buf_acc[1][conf_m][conf_n];
		predict_box[2][2] = FM_buf_acc[2][conf_m][conf_n];
		predict_box[2][3] = FM_buf_acc[3][conf_m][conf_n];
		predict_box[2][4] = conf_thresh;

	}
	else if (conf_j == 1) {
		// second bounding box
		predict_box[2][0] = FM_buf_acc[5][conf_m][conf_n];
		predict_box[2][1] = FM_buf_acc[6][conf_m][conf_n];
		predict_box[2][2] = FM_buf_acc[7][conf_m][conf_n];
		predict_box[2][3] = FM_buf_acc[8][conf_m][conf_n];
		predict_box[2][4] = conf_thresh;
	}

	constant[2][0] = conf_j;
	constant[2][1] = conf_n - 1;
	constant[2][2] = conf_m - 23;

	//// 3
	conf_thresh = -100.0;
	conf_j = 0;
	conf_m = 0;
	conf_n = 0;
	conf_box1 = 0.0;
	conf_box2 = 0.0;

	for (int m = xmin4; m < xmax4; m++) {
		for (int n = ymin4; n < ymax4; n++) {
			
			conf_box1 = FM_buf_acc[4][m][n];
			if (conf_box1 > conf_thresh) {
				conf_thresh = conf_box1;
				conf_j = 0;
				conf_m = m;
				conf_n = n;

			}
		}
	}

	for (int m = xmin4; m < xmax4; m++) {
		for (int n = ymin4; n < ymax4; n++) {
			
			conf_box2 = FM_buf_acc[9][m][n];
			if (conf_box2 > conf_thresh) {
				conf_thresh = conf_box2;
				conf_j = 1;
				conf_m = m;
				conf_n = n;
			}
		}
	}

	if (conf_j == 0) {
		// first bounding box
		predict_box[3][0] = FM_buf_acc[0][conf_m][conf_n];
		predict_box[3][1] = FM_buf_acc[1][conf_m][conf_n];
		predict_box[3][2] = FM_buf_acc[2][conf_m][conf_n];
		predict_box[3][3] = FM_buf_acc[3][conf_m][conf_n];
		predict_box[3][4] = conf_thresh;

	}
	else if (conf_j == 1) {
		// second bounding box
		predict_box[3][0] = FM_buf_acc[5][conf_m][conf_n];
		predict_box[3][1] = FM_buf_acc[6][conf_m][conf_n];
		predict_box[3][2] = FM_buf_acc[7][conf_m][conf_n];
		predict_box[3][3] = FM_buf_acc[8][conf_m][conf_n];
		predict_box[3][4] = conf_thresh;
	}

	constant[3][0] = conf_j;
	constant[3][1] = conf_n - 43;
	constant[3][2] = conf_m - 23;


}








inline FIX_FM relu_single( FIX_FM d ) {
	if( d > 6 )
		return 6;
	if( d < 0 )
		return 0;
	return d;
}



void compute_relu_copy_buf_to_DDR_new(uint512* dest, int buf_id, FIX_FM src[32][44][84], uint8 xmin, uint8 xmax, uint8 ymin, uint8 ymax) {
	uint512* dest_ptr = dest + 44 * 84 * buf_id + (xmin) * 84;
	for (uint8 h = xmin; h < xmax; h++) {


		for (uint8 w = ymin; w < ymax; w++) {


			uint512 DATA = 0;
			for (int c = 0; c < 32; c++) {
//#pragma HLS pipeline II = 1
//#pragma HLS unroll
				FIX_FM d = relu_single(src[c][h][w]);
				DATA.range(FM_RG + 16 * c, 16 * c) = d.range(FM_RG, 0);
			}
			dest_ptr[w].range(511, 0) = DATA.range(511, 0);
		}

		dest_ptr += 84;
	}
}

void relu_copy_buf_to_DDR_new(uint512* dest, int buf_id, FIX_FM src[32][44][84], int offset_h, int offset_w, int index,uint8 mask_bound[16])
{
	uint512* dest_ptr = dest + 44 * 84 * buf_id + (offset_h * 22) * 84 + (offset_w * 42);
	if (index == 0) {
		compute_relu_copy_buf_to_DDR_new(dest, buf_id, src, 0, 44, 0, 84);
	}
	else {
		compute_relu_copy_buf_to_DDR_new(dest, buf_id, src, mask_bound[0] + 1, mask_bound[1] + 1, mask_bound[2] + 1, mask_bound[3] + 1);
		compute_relu_copy_buf_to_DDR_new(dest, buf_id, src, mask_bound[4] + 1, mask_bound[5] + 1, mask_bound[6] + 3, mask_bound[7] + 3);
		compute_relu_copy_buf_to_DDR_new(dest, buf_id, src, mask_bound[8] + 3, mask_bound[9] + 3, mask_bound[10] + 1, mask_bound[11] + 1);
		compute_relu_copy_buf_to_DDR_new(dest, buf_id, src, mask_bound[12] + 3, mask_bound[13] + 3, mask_bound[14] + 3, mask_bound[15] + 3);
	}	

}



void compute_relu_copy_buf_to_DDR_acc_new(uint512* dest, int buf_id, FIX_FM_acc src[32][44][84], uint8 xmin, uint8 xmax, uint8 ymin, uint8 ymax) {
	uint512* dest_ptr = dest + 44 * 84 * buf_id + (xmin) * 84;
	for (uint8 h = xmin; h < xmax; h++) {


		for (uint8 w = ymin; w < ymax; w++) {


			uint512 DATA = 0;
			for (int c = 0; c < 32; c++) {
//#pragma HLS pipeline II = 1
//#pragma HLS unroll
				FIX_FM d = relu_single(src[c][h][w]);
				DATA.range(FM_RG + 16 * c, 16 * c) = d.range(FM_RG, 0);
			}
			dest_ptr[w].range(511, 0) = DATA.range(511, 0);
		}

		dest_ptr += 84;
	}
}



void relu_copy_buf_to_DDR_acc_new(uint512* dest, int buf_id, FIX_FM_acc src[32][44][84], int offset_h, int offset_w, int index,uint8 mask_bound[16])
{   
	uint512* dest_ptr = dest + 44 * 84 * buf_id + (offset_h * 22) * 84 + (offset_w * 42);
	if (index == 0) {
		compute_relu_copy_buf_to_DDR_acc_new(dest, buf_id, src, 0, 44, 0, 84);
	}
	else {
		compute_relu_copy_buf_to_DDR_acc_new(dest, buf_id, src, mask_bound[0] + 1, mask_bound[1] + 1, mask_bound[2] + 1, mask_bound[3] + 1);
		compute_relu_copy_buf_to_DDR_acc_new(dest, buf_id, src, mask_bound[4] + 1, mask_bound[5] + 1, mask_bound[6] + 3, mask_bound[7] + 3);
		compute_relu_copy_buf_to_DDR_acc_new(dest, buf_id, src, mask_bound[8] + 3, mask_bound[9] + 3, mask_bound[10] + 1, mask_bound[11] + 1);
		compute_relu_copy_buf_to_DDR_acc_new(dest, buf_id, src, mask_bound[12] + 3, mask_bound[13] + 3, mask_bound[14] + 3, mask_bound[15] + 3);
	}


}


void compute_load_buf_from_DDR_new(FIX_FM dest[32][44][84], uint512* src, int buf_id, uint8 xmin, uint8 xmax, uint8 ymin, uint8 ymax) {
	uint512* src_ptr = src + 44 * 84 * buf_id + (xmin) * 84;

	for (int h = xmin; h < xmax; h++) {

		for (int w = ymin; w < ymax; w++) {


			uint512 DATA = src_ptr[w];
			for (int c = 0; c < 32; c++) {
//#pragma HLS pipeline II=1
//#pragma HLS unroll
				dest[c][h][w].range(FM_RG, 0) = DATA.range(FM_RG + c * 16, 16 * c);
			}
		}

		src_ptr += 84;
	}
}

void load_buf_from_DDR_new(FIX_FM dest[32][44][84], uint512* src, int buf_id, int index, uint8 mask_bound[16])
{   
	if (index == 0) {
		compute_load_buf_from_DDR_new(dest, src, buf_id, 0, 44, 0, 84);
	}
	else {
		compute_load_buf_from_DDR_new(dest, src, buf_id, mask_bound[0] + 1, mask_bound[1] + 1, mask_bound[2] + 1, mask_bound[3] + 1);
		compute_load_buf_from_DDR_new(dest, src, buf_id, mask_bound[4] + 1, mask_bound[5] + 1, mask_bound[6] + 3, mask_bound[7] + 3);
		compute_load_buf_from_DDR_new(dest, src, buf_id, mask_bound[8] + 3, mask_bound[9] + 3, mask_bound[10] + 1, mask_bound[11] + 1);
		compute_load_buf_from_DDR_new(dest, src, buf_id, mask_bound[12] + 3, mask_bound[13] + 3, mask_bound[14] + 3, mask_bound[15] + 3);
	}	
}

void compute_load_buf_from_DDR_new_branch(FIX_FM dest[32][44][84], uint512* src, int buf_id, int number,int cmin, int offset) {
	uint512* src_ptr = src;

	src_ptr = src + 44 * 84 * (buf_id + number);

	for (int h = 0; h < 44; h++) {

		for (int w = 0; w < 84; w++) {


			uint512 DATA = src_ptr[w];
			for (int c = 0; c < 32; c = c + 6) {
//#pragma HLS pipeline II=1
//#pragma HLS unroll
				if (c < 32 - cmin) {
					dest[c / 6 + offset][h][w].range(FM_RG, 0) = DATA.range(FM_RG + (c+cmin) * 16, 16 * (c+cmin));
				}


			}
		}
		src_ptr += 84;
	}


}

void compute_load_buf_from_DDR_new_branch_channel(FIX_FM dest[32][44][84], uint512* src, int buf_id, int offset1, int offset2, int offset3) {

	compute_load_buf_from_DDR_new_branch(dest, src, buf_id, 0, 0 + offset1 - ((0 + offset1) / 6) * 6, 0);
	compute_load_buf_from_DDR_new_branch(dest, src, buf_id, 1, 4 + offset1 - ((4 + offset1) / 6) * 6, 6 - offset2);
	compute_load_buf_from_DDR_new_branch(dest, src, buf_id, 2, 2 + offset1 - ((2 + offset1) / 6) * 6, 11 - offset3);
	compute_load_buf_from_DDR_new_branch(dest, src, buf_id, 3, 0 + offset1 - ((0 + offset1) / 6) * 6, 16);
	compute_load_buf_from_DDR_new_branch(dest, src, buf_id, 4, 4 + offset1 - ((4 + offset1) / 6) * 6, 22 - offset2);
	compute_load_buf_from_DDR_new_branch(dest, src, buf_id, 5, 2 + offset1 - ((2 + offset1) / 6) * 6, 27 - offset3);
}

void load_weight_1x1_from_axi( FIX_WT dest[32][32], uint512 src[32])
{

	for(int ci = 0; ci < 32; ci++) {

		uint512 DATA = 0;
		DATA.range(511, 0) = src[ci].range(511, 0);
		for(int co = 0; co < 32; co++) {
//#pragma HLS pipeline
//#pragma HLS unroll
			dest[co][ci].range(WT_RG, 0) = DATA.range(WT_RG + co*16, co*16);
		}
	}
}



void load_weight_3x3_from_axi( FIX_WT dest[32][3][3], uint512 src[3][3])
{
	for(int m = 0; m < 3; m++) {
		for(int n = 0; n < 3; n++) {

			uint512 DATA = 0;
			DATA.range(511, 0) = src[m][n].range(511, 0);
			for(int c = 0; c < 32; c++) {
//#pragma HLS pipeline
//#pragma HLS unroll

				dest[c][m][n].range(WT_RG, 0) = DATA.range(WT_RG + c*16, c*16);
			}
		}
	}
}


void load_weight_5x5_from_axi(FIX_WT dest[32][5][5], uint512 src[5][5])
{
	
		for (int m = 0; m < 5; m++) {
			for (int n = 0; n < 5; n++) {
//
				uint512 DATA = 0;
				DATA.range(511, 0) = src[m][n].range(511, 0);
				for (int co = 0; co < 32; co++) {
//#pragma HLS pipeline
//#pragma HLS unroll

					dest[co][m][n].range(WT_RG, 0) = DATA.range(WT_RG + co * 16, co * 16);
				}
			}
		}	
}


void load_bias_from_axi(FIX_WT dest[32], uint512 src)
{
	for(int c = 0; c < 32; c++) {
//#pragma HLS unroll
//#pragma HLS pipeline
		dest[c].range(WT_RG, 0) = src.range(WT_RG + c*16, c*16);
	}
}

void compute_set_bias_1x1_new(FIX_FM_acc buf[32][44][84], FIX_WT bias[32], uint8 xmin, uint8 xmax, uint8 ymin, uint8 ymax) {
	for (uint8 h = xmin; h < xmax; h+=2) {
		for (uint8 w = ymin; w < ymax; w++) {
//#pragma HLS pipeline
			for (int c = 0; c < 32; c++) {
//#pragma HLS unroll
//#pragma HLS pipeline
				buf[c][h][w] = bias[c];
				buf[c][h + 1][w] = bias[c];

			}
		}
	}
}

void set_bias_1x1_new(FIX_FM_acc buf[32][44][84], FIX_WT bias[32],int index, uint8 mask_bound[16])
{   
	if (index == 0) {
		compute_set_bias_1x1_new(buf, bias, 1, 43, 1, 83);

	}
	else {
		compute_set_bias_1x1_new(buf, bias, mask_bound[0] + 1, mask_bound[1] + 1, mask_bound[2] + 1, mask_bound[3] + 1);
		compute_set_bias_1x1_new(buf, bias, mask_bound[4] + 1, mask_bound[5] + 1, mask_bound[6] + 3, mask_bound[7] + 3);
		compute_set_bias_1x1_new(buf, bias, mask_bound[8] + 3, mask_bound[9] + 3, mask_bound[10] + 1, mask_bound[11] + 1);
		compute_set_bias_1x1_new(buf, bias, mask_bound[12] + 3, mask_bound[13] + 3, mask_bound[14] + 3, mask_bound[15] + 3);
	}

}

void compute_set_bias_3x3_new(FIX_FM buf[32][44][84], FIX_WT bias[32], uint8 xmin, uint8 xmax, uint8 ymin, uint8 ymax) {
	for (uint8 h = xmin; h < xmax; h+=2) {
		for (uint8 w = ymin; w < ymax; w++) {
//#pragma HLS pipeline
			for (int c = 0; c < 32; c++) {
//#pragma HLS unroll
//#pragma HLS pipeline
				buf[c][h][w] = bias[c];
				buf[c][h + 1][w] = bias[c];

			}
		}
	}
}






void set_bias_3x3_new(FIX_FM buf[32][44][84], FIX_WT bias[32], int index, uint8 mask_bound[16])
{
	if (index == 0) {
		compute_set_bias_3x3_new(buf, bias, 1, 43, 1, 83);

	}
	else {
		compute_set_bias_3x3_new(buf, bias, mask_bound[0] + 1, mask_bound[1] + 1, mask_bound[2] + 1, mask_bound[3] + 1);
		compute_set_bias_3x3_new(buf, bias, mask_bound[4] + 1, mask_bound[5] + 1, mask_bound[6] + 3, mask_bound[7] + 3);
		compute_set_bias_3x3_new(buf, bias, mask_bound[8] + 3, mask_bound[9] + 3, mask_bound[10] + 1, mask_bound[11] + 1);
		compute_set_bias_3x3_new(buf, bias, mask_bound[12] + 3, mask_bound[13] + 3, mask_bound[14] + 3, mask_bound[15] + 3);
	}


}





void set_bias_5x5(FIX_FM_acc buf[32][44][84], FIX_WT bias[32])
{
	for (int h = 1; h <= 42; h += 2) {
		for (int w = 1; w <= 82; w++) {
//#pragma HLS pipeline
			for (int c = 0; c < 32; c++) {
//#pragma HLS unroll
//#pragma HLS pipeline
				buf[c][h][w] = bias[c];
				buf[c][h + 1][w] = bias[c];
			}
		}
	}
}



void load_dw1_pool_from_DDR( uint512* ddr_dw1_pool_burst,
							 FIX_FM buf[32][44][84],
							 int ch, int col, int row, int offset_h, int offset_w)
{
	uint512* ddr_dw1_pool_burst_ptr =ddr_dw1_pool_burst + ch*82*2*162*2 + (col*40 + offset_h*2)*162*2 + (row*80 + offset_w*2);

	for(int h = 0; h < 42; h++) {
		for(int w = 0; w < 82; w++) {
//#pragma HLS pipeline
			uint512 DATA = 0;
			DATA.range(511, 0) = ddr_dw1_pool_burst_ptr[w].range(511, 0);
			for(int c = 0; c < 32; c++) {
//#pragma HLS pipeline
//#pragma HLS unroll

				buf[c][h][w].range(FM_RG, 0) = DATA.range(FM_RG + c*16, c*16);
			}
		}
		ddr_dw1_pool_burst_ptr += 162*2;
	}
}



void load_dw2_pool_from_DDR( uint512* ddr_dw2_pool_burst,
							 FIX_FM buf[32][44][84],
							 int ch, int col, int row, int offset_h, int offset_w)
{
	uint512* ddr_dw2_pool_burst_ptr = ddr_dw2_pool_burst + ch*42*2*82*2 + (col*40 + offset_h*2)*82*2 + (row*80 + offset_w*2);

	for(int h = 0; h < 42; h++) {
		for(int w = 0; w < 82; w++) {
//#pragma HLS pipeline
			uint512 DATA = 0;
			DATA.range(511, 0) = ddr_dw2_pool_burst_ptr[w].range(511, 0);
			for(int c = 0; c < 32; c++) {
//#pragma HLS unroll
//#pragma HLS pipeline
				buf[c][h][w].range(FM_RG, 0) = DATA.range(FM_RG + c*16, c*16);
			}
		}
		ddr_dw2_pool_burst_ptr += 82*2;
	}
}




void clear_buffer( FIX_FM_acc buf[32][44][84] )
{
	for(int h = 0; h < 44; h+=2) {
		for(int w = 0; w < 84; w++) {
//#pragma HLS pipeline
			for(int c = 0; c < 32; c++) {
//#pragma HLS unroll
//#pragma HLS pipeline
				buf[c][h  ][w] = 0;
				buf[c][h+1][w] = 0;
			}
		}
	}
}

void clear_buffer_new(FIX_FM buf[32][44][84])
{
	for (int h = 0; h < 44; h += 2) {
		for (int w = 0; w < 84; w++) {
//#pragma HLS pipeline
			for (int c = 0; c < 32; c++) {
//#pragma HLS pipeline
//#pragma HLS unroll
				buf[c][h][w] = 0;
				buf[c][h + 1][w] = 0;
			}
		}
	}
}

void clear_mask(FIX_FM_acc buf[16][16])
{
	for (int h = 0; h < 16; h += 2) {

//#pragma HLS pipeline
		for (int w = 0; w < 16; w++) {

//#pragma HLS pipeline			
//#pragma HLS unroll
				buf[h][w] = 0;
				buf[h + 1][w] = 0;
			}
		}

	

}



FIX_FM img_norm_ch[256] = {
		-2.000000, -1.984314, -1.968627, -1.952941, -1.937255, -1.921569, -1.905882, -1.890196, -1.874510, -1.858824, -1.843137, -1.827451, -1.811765, -1.796078, -1.780392, -1.764706, -1.749020,
		-1.733333, -1.717647, -1.701961, -1.686275, -1.670588, -1.654902, -1.639216, -1.623529, -1.607843, -1.592157, -1.576471, -1.560784, -1.545098, -1.529412, -1.513725, -1.498039,
		-1.482353, -1.466667, -1.450980, -1.435294, -1.419608, -1.403922, -1.388235, -1.372549, -1.356863, -1.341176, -1.325490, -1.309804, -1.294118, -1.278431, -1.262745, -1.247059,
		-1.231373, -1.215686, -1.200000, -1.184314, -1.168627, -1.152941, -1.137255, -1.121569, -1.105882, -1.090196, -1.074510, -1.058824, -1.043137, -1.027451, -1.011765, -0.996078,
		-0.980392, -0.964706, -0.949020, -0.933333, -0.917647, -0.901961, -0.886275, -0.870588, -0.854902, -0.839216, -0.823529, -0.807843, -0.792157, -0.776471, -0.760784, -0.745098,
		-0.729412, -0.713725, -0.698039, -0.682353, -0.666667, -0.650980, -0.635294, -0.619608, -0.603922, -0.588235, -0.572549, -0.556863, -0.541176, -0.525490, -0.509804, -0.494118,
		-0.478431, -0.462745, -0.447059, -0.431373, -0.415686, -0.400000, -0.384314, -0.368627, -0.352941, -0.337255, -0.321569, -0.305882, -0.290196, -0.274510, -0.258824, -0.243137,
		-0.227451, -0.211765, -0.196078, -0.180392, -0.164706, -0.149020, -0.133333, -0.117647, -0.101961, -0.086275, -0.070588, -0.054902, -0.039216, -0.023529, -0.007843, 0.007843,
		0.023529, 0.039216, 0.054902, 0.070588, 0.086275, 0.101961, 0.117647, 0.133333, 0.149020, 0.164706, 0.180392, 0.196078, 0.211765, 0.227451, 0.243137, 0.258824,
		0.274510, 0.290196, 0.305882, 0.321569, 0.337255, 0.352941, 0.368627, 0.384314, 0.400000, 0.415686, 0.431373, 0.447059, 0.462745, 0.478431, 0.494118, 0.509804,
		0.525490, 0.541176, 0.556863, 0.572549, 0.588235, 0.603922, 0.619608, 0.635294, 0.650980, 0.666667, 0.682353, 0.698039, 0.713725, 0.729412, 0.745098, 0.760784,
		0.776471, 0.792157, 0.807843, 0.823529, 0.839216, 0.854902, 0.870588, 0.886275, 0.901961, 0.917647, 0.933333, 0.949020, 0.964706, 0.980392, 0.996078, 1.011765,
		1.027451, 1.043137, 1.058824, 1.074510, 1.090196, 1.105882, 1.121569, 1.137255, 1.152941, 1.168627, 1.184314, 1.200000, 1.215686, 1.231373, 1.247059, 1.262745,
		1.278431, 1.294118, 1.309804, 1.325490, 1.341176, 1.356863, 1.372549, 1.388235, 1.403922, 1.419608, 1.435294, 1.450980, 1.466667, 1.482353, 1.498039, 1.513725,
		1.529412, 1.545098, 1.560784, 1.576471, 1.592157, 1.607843, 1.623529, 1.639216, 1.654902, 1.670588, 1.686275, 1.701961, 1.717647, 1.733333, 1.749020, 1.764706,
		1.780392, 1.796078, 1.811765, 1.827451, 1.843137, 1.858824, 1.874510, 1.890196, 1.905882, 1.921569, 1.937255, 1.952941, 1.968627, 1.984314, 2.000000
};



void load_image_chunk_norm(FIX_FM img_buf[32][44][84], uint8 image_in_raw_pad_burst[3*162*2*322*2],
							int col, int row, int offset_h = 0, int offset_w = 0)
{
	uint8* image_in_raw_pad_burst_ptr;

	image_in_raw_pad_burst_ptr = image_in_raw_pad_burst + (col*40 + offset_h*2)*322*2 + row*80 + offset_w*2;

	for(int i = 0; i < 44; i++) {
		for(int j = 0; j < 84; j++) {
//#pragma HLS pipeline
				img_buf[0][i][j] = img_norm_ch[((image_in_raw_pad_burst_ptr[j])).to_uint()];
		}
		image_in_raw_pad_burst_ptr += 322*2;
	}

	image_in_raw_pad_burst_ptr = image_in_raw_pad_burst + 1*162*2*322*2 + (col*40 + offset_h*2)*322*2 + row*80 + offset_w*2;
	for(int i = 0; i < 44; i++) {
		for(int j = 0; j < 84; j++) {
//#pragma HLS pipeline
				img_buf[1][i][j] = img_norm_ch[((image_in_raw_pad_burst_ptr[j])).to_uint()];
		}
		image_in_raw_pad_burst_ptr += 322*2;
	}

	image_in_raw_pad_burst_ptr = image_in_raw_pad_burst + 2*162*2*322*2 + (col*40 + offset_h*2)*322*2 + row*80 + offset_w*2;
	for(int i = 0; i < 44; i++) {
		for(int j = 0; j < 84; j++) {
//#pragma HLS pipeline
				img_buf[2][i][j] = img_norm_ch[((image_in_raw_pad_burst_ptr[j])).to_uint()];
		}
		image_in_raw_pad_burst_ptr += 322*2;
	}
}


void Relu6_3x3( FIX_FM buf[32][44][84])
{
	for(int h = 1; h <= 42; h+=2) {
		for(int w = 1; w <= 82; w++) {

			for(int c = 0; c < 32; c++) {
//#pragma HLS unroll
//#pragma HLS pipeline
				buf[c][h  ][w] = relu_single(buf[c][h  ][w]);
				buf[c][h+1][w] = relu_single(buf[c][h+1][w]);
			}
		}
	}
}




//relu and maxpool2x2 used in backbone
inline FIX_FM relu_max(FIX_FM a, FIX_FM b, FIX_FM c, FIX_FM d)  
{
	FIX_FM t1, t2;

	if(a > b) t1 = relu_single(a);
	else t1 = relu_single(b);

	if(c > d) t2 = relu_single(c);
	else t2 = relu_single(d);

	if(t1 > t2) return t1;
	else return t2;
}


//relu and maxpool5x5 used in new branch
inline FIX_FM relu_max_5x5(FIX_FM a, FIX_FM b, FIX_FM c, FIX_FM d, FIX_FM e)
{
	FIX_FM t1, t2 , t3;

	if (a > b) t1 = relu_single(a);
	else t1 = relu_single(b);

	if (c > d) t2 = relu_single(c);
	else t2 = relu_single(d);

	if (t1 > t2) t3 = t1;
	else t3 = t2;

	if (relu_single(e) > t3) return relu_single(e);
	else return t3;

	
}

//maxpool 1x2 used in new branch
inline FIX_FM max_1x2(FIX_FM a, FIX_FM b) {
	if (a > b) return a;
	else return b;
}

//activation function in after the final layer of new branch
inline FIX_FM sigmoid_single(FIX_FM d) {

	FIX_FM f = 1 / (1 + expf(-d));
	return f;
}

void compute_sigmoid(FIX_FM top[32][44][84], FIX_FM bottom[13][13]) {
	
	for (int h = 1; h <= 12; h++) {
//#pragma HLS pipeline
		for (int w = 1; w <= 12; w++) {
			bottom[h][w] = sigmoid_single(top[h][w]);
		}
	}
}



void accumulate_sigmoid(FIX_FM top[32][44][84], FIX_FM bottom[13][13]) {
//#pragma HLS array_partition variable=top dim=1 complete
//#pragma HLS array_partition variable=bottom dim=1 complete
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int h = 1; h < 13; h++) {
				for (int w = 1; w < 13; w++) {
//#pragma HLS pipeline
					FIX_FM  d = 0;

					for (int co = 0; co < 32; co++) {
//#pragma HLS pipeline
//#pragma HLS unroll      
						d += top[co][h][w];

					}
					d = sigmoid_single(d);
					bottom[h][w] = d;
				}
			}
		}
	}

}

void Relu_Max_Pooling(
		 FIX_FM_acc buf_in[32][44][84],
		 uint512* ddr_dw1_pool_burst,
		 uint512* ddr_dw2_pool_burst,
		 uint512* ddr_buf, int buf_id,
		 int ch, int col, int row, int offset_h, int offset_w, int layer)
{

	uint512* buf_in_ptr = ddr_buf + buf_id*44*84 + (1 + offset_h*22)*84 + offset_w*42;
	uint512* ddr_dw1_pool_burst_ptr = ddr_dw1_pool_burst + ch*82*2*162*2 + (1 + col*20 + offset_h*2)*162*2 + (row*40 + offset_w*2);
	uint512* ddr_dw2_pool_burst_ptr = ddr_dw2_pool_burst + ch*42*2*82*2 + (1 + col*20 + offset_h*2)*82*2 + (row*40 + offset_w*2);

//#pragma HLS array_partition variable=buf_in dim=1 complete

	for(int h = 1; h <= 20; h++) {
		for(int w = 1; w <= 40; w++) {
//#pragma HLS pipeline II=2
			uint512 DATA = 0;
			for(int c = 0; c < 32; c++) {
//#pragma HLS pipeline II=1
//#pragma HLS unroll
				FIX_FM d = relu_max(buf_in[c][h*2-1][w*2-1], buf_in[c][h*2-1][w*2],
							   buf_in[c][h*2][w*2-1], buf_in[c][h*2][w*2]);

				DATA.range(FM_RG + c*16, c*16) = d.range(FM_RG, 0);
			}

			if( layer == 1 ) {
				ddr_dw1_pool_burst_ptr[w].range(511, 0) = DATA.range(511, 0);
			}
			else if( layer == 2 ) {
				ddr_dw2_pool_burst_ptr[w].range(511, 0) = DATA.range(511, 0);
			}
			else if( layer == 3 ) {
				buf_in_ptr[w].range(511, 0) = DATA.range(511, 0);
			}
		}

		buf_in_ptr += 84;
		ddr_dw1_pool_burst_ptr += 162*2;
		ddr_dw2_pool_burst_ptr += 82*2;
	}
}


void compute_Relu_Max_Pooling_2x2(FIX_FM buf_in[32][44][84], FIX_FM buf_out[32][44][84], int hmin, int hmax, int wmin, int wmax, int offsetx, int offsety) {
//#pragma HLS array_partition variable=buf_in dim=1 complete
//#pragma HLS array_partition variable=buf_out dim=1 complete
	for (int h = hmin; h <= hmax; h++) {
		for (int w = wmin; w <= wmax; w++) {
//#pragma HLS pipeline II=2
			for (int c = 0; c < 32; c++) {
//#pragma HLS pipeline II=1
//#pragma HLS unroll
				buf_out[c][h][w] = relu_max(buf_in[c][h * 2 - 1 - 2 * offsetx][w * 2 - 1 - 2 * offsety], buf_in[c][h * 2 - 1 - 2 * offsetx][w * 2 - 0 - 2 * offsety],
					buf_in[c][h * 2 - 0 - 2 * offsetx][w * 2 - 1 - 2 * offsety], buf_in[c][h * 2 - 0 - 2 * offsetx][w * 2 - 0 - 2 * offsety]);
			}

		}
	}
}


void compute_Relu_Max_Pooling_1x2(FIX_FM buf_in[32][44][84], FIX_FM buf_out[32][44][84], int hmin, int hmax,int wmin, int wmax, int offsety) {
//#pragma HLS array_partition variable=buf_in dim=1 complete
//#pragma HLS array_partition variable=buf_out dim=1 complete
	for (int h = hmin; h <= hmax; h++) {
		for (int w = wmin; w <= wmax; w++) {
//#pragma HLS pipeline II=2
			for (int c = 0; c < 32; c++) {
//#pragma HLS pipeline II=1
//#pragma HLS unroll
				buf_out[c][h][w] = max_1x2(buf_in[c][h][w * 2 - 1 -2*offsety], buf_in[c][h][w * 2 - 2*offsety]);
			}
		}
	}
}

void Relu_Max_Pooling_1x2(FIX_FM buf_in[32][44][84], FIX_FM buf_out[32][44][84]) {
//#pragma HLS array_partition variable=buf_in dim=1 complete
//#pragma HLS array_partition variable=buf_out dim=1 complete
	compute_Relu_Max_Pooling_1x2(buf_in, buf_out, 1, 5, 1, 5, 0);
	compute_Relu_Max_Pooling_1x2(buf_in, buf_out, 1, 5, 8, 12, 1);
	compute_Relu_Max_Pooling_1x2(buf_in, buf_out, 8, 12, 1, 5, 0);
	compute_Relu_Max_Pooling_1x2(buf_in, buf_out, 8, 12, 8, 12, 1);
}


void load_and_reorg_part_new(uint512* buf_in, int buf_id,
	FIX_FM buf_out_1[32][44][84],
	FIX_FM buf_out_2[32][44][84],
	FIX_FM buf_out_3[32][44][84],
	FIX_FM_acc buf_out_4[32][44][84],	// borrow one
	int offset_h, int offset_w,uint8 xmin, uint8 xmax,uint8 ymin, uint8 ymax)
{
	uint512* buf_in_ptr = buf_in + buf_id * 44 * 84;

	for (int h = xmin; h <= xmax; h += 4) {
		for (int w = ymin; w <= ymax; w += 4) {

			uint512 DATA[16] = { 0 };
			DATA[0].range(511, 0) = buf_in_ptr[(h) * 84 + w].range(511, 0);
			DATA[1].range(511, 0) = buf_in_ptr[(h + 2) * 84 + w].range(511, 0);
			DATA[2].range(511, 0) = buf_in_ptr[(h) * 84 + w + 2].range(511, 0);
			DATA[3].range(511, 0) = buf_in_ptr[(h + 2) * 84 + w + 2].range(511, 0);
			DATA[4].range(511, 0) = buf_in_ptr[(h) * 84 + w + 1].range(511, 0);
			DATA[5].range(511, 0) = buf_in_ptr[(h + 2) * 84 + w + 1].range(511, 0);
			DATA[6].range(511, 0) = buf_in_ptr[(h) * 84 + w + 3].range(511, 0);
			DATA[7].range(511, 0) = buf_in_ptr[(h + 2) * 84 + w + 3].range(511, 0);
			DATA[8].range(511, 0) = buf_in_ptr[(h + 1) * 84 + w].range(511, 0);
			DATA[9].range(511, 0) = buf_in_ptr[(h + 3) * 84 + w].range(511, 0);
			DATA[10].range(511, 0) = buf_in_ptr[(h + 1) * 84 + w + 2].range(511, 0);
			DATA[11].range(511, 0) = buf_in_ptr[(h + 3) * 84 + w + 2].range(511, 0);
			DATA[12].range(511, 0) = buf_in_ptr[(h + 1) * 84 + w + 1].range(511, 0);
			DATA[13].range(511, 0) = buf_in_ptr[(h + 3) * 84 + w + 1].range(511, 0);
			DATA[14].range(511, 0) = buf_in_ptr[(h + 1) * 84 + w + 3].range(511, 0);
			DATA[15].range(511, 0) = buf_in_ptr[(h + 3) * 84 + w + 3].range(511, 0);

			for (int c = 0; c < 8; c++) {
				///////////////////
				buf_out_1[c * 4][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[0].range(FM_RG + c * 16, c * 16);
				buf_out_1[c * 4][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[1].range(FM_RG + c * 16, c * 16);
				buf_out_1[c * 4][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[2].range(FM_RG + c * 16, c * 16);
				buf_out_1[c * 4][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[3].range(FM_RG + c * 16, c * 16);

				buf_out_1[c * 4 + 1][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[4].range(FM_RG + c * 16, c * 16);
				buf_out_1[c * 4 + 1][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[5].range(FM_RG + c * 16, c * 16);
				buf_out_1[c * 4 + 1][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[6].range(FM_RG + c * 16, c * 16);
				buf_out_1[c * 4 + 1][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[7].range(FM_RG + c * 16, c * 16);

				buf_out_1[c * 4 + 2][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[8].range(FM_RG + c * 16, c * 16);
				buf_out_1[c * 4 + 2][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[9].range(FM_RG + c * 16, c * 16);
				buf_out_1[c * 4 + 2][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[10].range(FM_RG + c * 16, c * 16);
				buf_out_1[c * 4 + 2][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[11].range(FM_RG + c * 16, c * 16);

				buf_out_1[c * 4 + 3][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[12].range(FM_RG + c * 16, c * 16);
				buf_out_1[c * 4 + 3][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[13].range(FM_RG + c * 16, c * 16);
				buf_out_1[c * 4 + 3][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[14].range(FM_RG + c * 16, c * 16);
				buf_out_1[c * 4 + 3][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[15].range(FM_RG + c * 16, c * 16);

				///////////////////
				buf_out_2[c * 4][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[0].range(FM_RG + (c + 8) * 16, (c + 8) * 16);
				buf_out_2[c * 4][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[1].range(FM_RG + (c + 8) * 16, (c + 8) * 16);
				buf_out_2[c * 4][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[2].range(FM_RG + (c + 8) * 16, (c + 8) * 16);
				buf_out_2[c * 4][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[3].range(FM_RG + (c + 8) * 16, (c + 8) * 16);

				buf_out_2[c * 4 + 1][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[4].range(FM_RG + (c + 8) * 16, (c + 8) * 16);
				buf_out_2[c * 4 + 1][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[5].range(FM_RG + (c + 8) * 16, (c + 8) * 16);
				buf_out_2[c * 4 + 1][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[6].range(FM_RG + (c + 8) * 16, (c + 8) * 16);
				buf_out_2[c * 4 + 1][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[7].range(FM_RG + (c + 8) * 16, (c + 8) * 16);

				buf_out_2[c * 4 + 2][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[8].range(FM_RG + (c + 8) * 16, (c + 8) * 16);
				buf_out_2[c * 4 + 2][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[9].range(FM_RG + (c + 8) * 16, (c + 8) * 16);
				buf_out_2[c * 4 + 2][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[10].range(FM_RG + (c + 8) * 16, (c + 8) * 16);
				buf_out_2[c * 4 + 2][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[11].range(FM_RG + (c + 8) * 16, (c + 8) * 16);

				buf_out_2[c * 4 + 3][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[12].range(FM_RG + (c + 8) * 16, (c + 8) * 16);
				buf_out_2[c * 4 + 3][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[13].range(FM_RG + (c + 8) * 16, (c + 8) * 16);
				buf_out_2[c * 4 + 3][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[14].range(FM_RG + (c + 8) * 16, (c + 8) * 16);
				buf_out_2[c * 4 + 3][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[15].range(FM_RG + (c + 8) * 16, (c + 8) * 16);

				buf_out_3[c * 4][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[0].range(FM_RG + (c + 16) * 16, (c + 16) * 16);
				buf_out_3[c * 4][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[1].range(FM_RG + (c + 16) * 16, (c + 16) * 16);
				buf_out_3[c * 4][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[2].range(FM_RG + (c + 16) * 16, (c + 16) * 16);
				buf_out_3[c * 4][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[3].range(FM_RG + (c + 16) * 16, (c + 16) * 16);

				buf_out_3[c * 4 + 1][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[4].range(FM_RG + (c + 16) * 16, (c + 16) * 16);
				buf_out_3[c * 4 + 1][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[5].range(FM_RG + (c + 16) * 16, (c + 16) * 16);
				buf_out_3[c * 4 + 1][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[6].range(FM_RG + (c + 16) * 16, (c + 16) * 16);
				buf_out_3[c * 4 + 1][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[7].range(FM_RG + (c + 16) * 16, (c + 16) * 16);

				buf_out_3[c * 4 + 2][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[8].range(FM_RG + (c + 16) * 16, (c + 16) * 16);
				buf_out_3[c * 4 + 2][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[9].range(FM_RG + (c + 16) * 16, (c + 16) * 16);
				buf_out_3[c * 4 + 2][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[10].range(FM_RG + (c + 16) * 16, (c + 16) * 16);
				buf_out_3[c * 4 + 2][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[11].range(FM_RG + (c + 16) * 16, (c + 16) * 16);

				buf_out_3[c * 4 + 3][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[12].range(FM_RG + (c + 16) * 16, (c + 16) * 16);
				buf_out_3[c * 4 + 3][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(FM_RG, 0) = DATA[13].range(FM_RG + (c + 16) * 16, (c + 16) * 16);
				buf_out_3[c * 4 + 3][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[14].range(FM_RG + (c + 16) * 16, (c + 16) * 16);
				buf_out_3[c * 4 + 3][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(FM_RG, 0) = DATA[15].range(FM_RG + (c + 16) * 16, (c + 16) * 16);

				/////////////////////
				buf_out_4[c * 4][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(WT_RG, 0) = DATA[0].range(WT_RG + (c + 24) * 16, (c + 24) * 16);
				buf_out_4[c * 4][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(WT_RG, 0) = DATA[1].range(WT_RG + (c + 24) * 16, (c + 24) * 16);
				buf_out_4[c * 4][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(WT_RG, 0) = DATA[2].range(WT_RG + (c + 24) * 16, (c + 24) * 16);
				buf_out_4[c * 4][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(WT_RG, 0) = DATA[3].range(WT_RG + (c + 24) * 16, (c + 24) * 16);

				buf_out_4[c * 4 + 1][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(WT_RG, 0) = DATA[4].range(WT_RG + (c + 24) * 16, (c + 24) * 16);
				buf_out_4[c * 4 + 1][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(WT_RG, 0) = DATA[5].range(WT_RG + (c + 24) * 16, (c + 24) * 16);
				buf_out_4[c * 4 + 1][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(WT_RG, 0) = DATA[6].range(WT_RG + (c + 24) * 16, (c + 24) * 16);
				buf_out_4[c * 4 + 1][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(WT_RG, 0) = DATA[7].range(WT_RG + (c + 24) * 16, (c + 24) * 16);

				buf_out_4[c * 4 + 2][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(WT_RG, 0) = DATA[8].range(WT_RG + (c + 24) * 16, (c + 24) * 16);
				buf_out_4[c * 4 + 2][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(WT_RG, 0) = DATA[9].range(WT_RG + (c + 24) * 16, (c + 24) * 16);
				buf_out_4[c * 4 + 2][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(WT_RG, 0) = DATA[10].range(WT_RG + (c + 24) * 16, (c + 24) * 16);
				buf_out_4[c * 4 + 2][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(WT_RG, 0) = DATA[11].range(WT_RG + (c + 24) * 16, (c + 24) * 16);

				buf_out_4[c * 4 + 3][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(WT_RG, 0) = DATA[12].range(WT_RG + (c + 24) * 16, (c + 24) * 16);
				buf_out_4[c * 4 + 3][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + offset_w * 42].range(WT_RG, 0) = DATA[13].range(WT_RG + (c + 24) * 16, (c + 24) * 16);
				buf_out_4[c * 4 + 3][(h + 1) / 2 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(WT_RG, 0) = DATA[14].range(WT_RG + (c + 24) * 16, (c + 24) * 16);
				buf_out_4[c * 4 + 3][(h + 1) / 2 + 1 + offset_h * 22][(w + 1) / 2 + 1 + offset_w * 42].range(WT_RG, 0) = DATA[15].range(WT_RG + (c + 24) * 16, (c + 24) * 16);
			}
		}
	}
}



void load_and_reorg_new(uint512* buf_1, int buf_id_1, uint512* buf_2, int buf_id_2,
	uint512* buf_3, int buf_id_3, uint512* buf_4, int buf_id_4,
	FIX_FM buf_out_1[32][44][84],
	FIX_FM buf_out_2[32][44][84],
	FIX_FM buf_out_3[32][44][84],
	FIX_FM_acc buf_out_4[32][44][84],
	uint8 mask_bound_reorg[16])
{
	load_and_reorg_part_new(buf_1, buf_id_1, buf_out_1, buf_out_2, buf_out_3, buf_out_4, 0, 0,mask_bound_reorg[0] + 1,mask_bound_reorg[1],mask_bound_reorg[2] + 1,mask_bound_reorg[3]);
	load_and_reorg_part_new(buf_2, buf_id_2, buf_out_1, buf_out_2, buf_out_3, buf_out_4, 0, 1, mask_bound_reorg[4] + 1, mask_bound_reorg[5], mask_bound_reorg[6] + 1, mask_bound_reorg[7]);
	load_and_reorg_part_new(buf_3, buf_id_3, buf_out_1, buf_out_2, buf_out_3, buf_out_4, 1, 0, mask_bound_reorg[8] + 1, mask_bound_reorg[9], mask_bound_reorg[10] + 1, mask_bound_reorg[11]);
	load_and_reorg_part_new(buf_4, buf_id_4, buf_out_1, buf_out_2, buf_out_3, buf_out_4, 1, 1, mask_bound_reorg[12] + 1, mask_bound_reorg[13], mask_bound_reorg[14] + 1, mask_bound_reorg[15]);
}


void local_buf_copy( FIX_FM dest[32][44][84], FIX_FM_acc src[32][44][84])
{
	for(int h = 0; h < 44; h+=2) {
		for(int w = 0; w < 84; w++) {
//#pragma HLS pipeline
			for(int c = 0; c < 32; c++) {
				dest[c][h  ][w].range(FM_RG, 0) = src[c][h  ][w].range(FM_RG, 0);
				dest[c][h+1][w].range(FM_RG, 0) = src[c][h+1][w].range(FM_RG, 0);
			}
		}
	}
}

//buffer copy used in new branch
void local_buf_copy_new(FIX_FM dest[32][44][84], FIX_FM_acc src[32][44][84])
{
	for (int h = 0; h < 44; h += 2) {
		for (int w = 0; w < 84; w++) {
//#pragma HLS pipeline
			for (int c = 0; c < 32; c++) {
				dest[c][h][w] = src[c][h][w];
				dest[c][h + 1][w] = src[c][h + 1][w];
			}
		}
	}
}


void compute_generate_mask_fix(FIX_FM input[13][13], uint8 xmin[1], uint8 xmax[1], uint8 ymin[1], uint8 ymax[1], uint8 offsetx, uint8 offsety, uint8 index[1]) {
//#pragma HLS array_partition variable=input dim=1 complete
	for (int m = 0 ; m < 5 ; m++) {
//# pragma HLS pipeline
		for (int n = 0 ; n < 5 ; n++) {
//#pragma HLS pipeline
//#pragma HLS unroll
			if (input[m + 1 + 7 * offsetx][n + 1 + 7 * offsety] > 0.3 ) {
				index[0]++;
				if ((m + 5*offsetx) < xmin[0]) xmin[0] = m + 5 * offsetx;
				if ((n + 5*offsety) < ymin[0]) ymin[0] = n + 5 * offsety;
				if ((m + 5 * offsetx) > xmax[0]) xmax[0] = m + 5 * offsetx;
				if ((n + 5 * offsety) > ymax[0]) ymax[0] = n + 5 * offsety;
			}
		}
	}
}

void generate_mask_fix(uint8 xmin1[1], uint8 xmax1[1], uint8 ymin1[1], uint8 ymax1[1],
	uint8 xmin2[1], uint8 xmax2[1], uint8 ymin2[1], uint8 ymax2[1],
	uint8 xmin3[1], uint8 xmax3[1], uint8 ymin3[1], uint8 ymax3[1],
	uint8 xmin4[1], uint8 xmax4[1], uint8 ymin4[1], uint8 ymax4[1],
	FIX_FM temp[13][13], uint8 index1[1], uint8 index2[1], uint8 index3[1], uint8 index4[1])
{
	xmin1[0] = 4; ymin1[0] = 4; xmin2[0] = 4; ymin3[0] = 4;
	xmax1[0] = 0; ymax1[0] = 0; xmax2[0] = 0; ymax3[0] = 0;
	xmin3[0] = 9; xmin4[0] = 9; ymin2[0] = 9; ymin4[0] = 9;
	xmax3[0] = 5; xmax4[0] = 5; ymax2[0] = 5; ymax4[0] = 5;
	index1[0] = 0;
	index2[0] = 0;
	index3[0] = 0;
	index4[0] = 0;

//#pragma HLS array_partition variable=temp dim=1 complete
	compute_generate_mask_fix(temp, xmin1, xmax1, ymin1, ymax1, 0, 0, index1);
	compute_generate_mask_fix(temp, xmin2, xmax2, ymin2, ymax2, 0, 1, index2);
	compute_generate_mask_fix(temp, xmin3, xmax3, ymin3, ymax3, 1, 0, index3);
	compute_generate_mask_fix(temp, xmin4, xmax4, ymin4, ymax4, 1, 1, index4);



}






void MaskNet(	uint8 image_in_raw_pad[3*162*2*322*2],

				uint512 conv_weight_1x1_all[1000][32],
				uint512 conv_weight_3x3_all[1000][3][3],
				uint512 bias_all[500],
	            uint512 conv_weight_5x5_all[6][5][5],
				

				uint512 DDR_dw1_pool_out_PL_burst[64/32*82*2*162*2],
				uint512 DDR_dw2_pool_out_PL_burst[96/32*42*2*82*2],
				uint512 DDR_buf_burst[128*44*84],
	   
	            

				float debug[10][32][44][84],
				float predict_boxes[4][5],
				int constant[4][3]
)
{

#pragma HLS INTERFACE m_axi depth=3*162*322 	port=image_in_raw_pad			offset=slave	bundle=INPUT
#pragma HLS INTERFACE m_axi depth=306*32*32		port=conv_weight_1x1_all		offset=slave	bundle=INPUT
#pragma HLS INTERFACE m_axi depth=24*32*3*3		port=conv_weight_3x3_all		offset=slave	bundle=INPUT
#pragma HLS INTERFACE m_axi depth=63*32			port=bias_all					offset=slave	bundle=INPUT
#pragma HLS INTERFACE m_axi depth=6*5*5         port=conv_weight_5x5_all        offset=slave    bundle=INPUT



#pragma HLS INTERFACE m_axi depth=106272		port=DDR_dw1_pool_out_PL_burst	offset=slave	bundle=INPUT
#pragma HLS INTERFACE m_axi depth=41328			port=DDR_dw2_pool_out_PL_burst	offset=slave	bundle=INPUT
#pragma HLS INTERFACE m_axi depth=473088		port=DDR_buf_burst				offset=slave	bundle=INPUT


#pragma HLS INTERFACE m_axi depth=5				port=predict_boxes				offset=slave	bundle=OUTPUT
#pragma HLS INTERFACE m_axi depth=5				port=constant					offset=slave	bundle=OUTPUT
#pragma HLS INTERFACE m_axi depth=10*32*22*42	port=debug						offset=slave	bundle=OUTPUT

#pragma HLS INTERFACE s_axilite register	port=return


// #pragma HLS ALLOCATION instances=CONV_1x1_new			 	limit=1 function
// #pragma HLS ALLOCATION instances=DW_CONV_3x3_new	    	limit=1 function
// #pragma HLS ALLOCATION instances=Relu_Max_Pooling	    	limit=1 function
// #pragma HLS ALLOCATION instances=load_image_chunk_norm		limit=1 function
// #pragma HLS ALLOCATION instances=DW_CONV_5x5		        limit=1 function
// #pragma HLS ALLOCATION instances=Relu_Max_Pooling_1x2		limit=1 function	


	int CI_N, CO_N , CI_N_new_branch , CO_N_new_branch;
	int weight_3x3_index, weight_1x1_index, bias_3x3_index, bias_1x1_index;
	uint8 mask_bound[16];
	uint8 mask_bound_reorg[16];

	/////////////////////////////// DW1_CONV_3x3 + DW1_CONV_1x1 + POOL ////////////////////////////
	/// DW1_CONV_3x3
	weight_3x3_index = 0;
	bias_3x3_index = 0;
	weight_1x1_index = 0;
	bias_1x1_index = 1;

	CI_N = 32 / 32;
	CO_N = 64 / 32;

	load_weight_3x3_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[weight_3x3_index]);
	load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index]);
	load_bias_from_axi(bias_buf[1], bias_all[bias_1x1_index + 0]);
	load_bias_from_axi(bias_buf[2], bias_all[bias_1x1_index + 1]);
	load_weight_1x1_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1_index + 0]);
	load_weight_1x1_from_axi(weight_buf_1x1[1], conv_weight_1x1_all[weight_1x1_index + 1]);

	for(int row = 0; row < 8; row++) {

		load_image_chunk_norm(FM_buf1, image_in_raw_pad, 0, row, 0/4, row/4);
		for(int col = 0; col < 8; col++) {
			set_bias_3x3_new(FM_buf2, bias_buf[0],0,mask_bound);

			if( col % 2 == 0 ) {
				DW_CONV_3x3_new(FM_buf1, FM_buf2, weight_buf_3x3[0], bias_buf[0], 1,0,0,mask_bound);
				load_image_chunk_norm(FM_buf3, image_in_raw_pad, col+1, row, (col+1)/4, row/4);
			}
			else {
				DW_CONV_3x3_new(FM_buf3, FM_buf2, weight_buf_3x3[0], bias_buf[0], 1,0,0,mask_bound);
				load_image_chunk_norm(FM_buf1, image_in_raw_pad, col+1, row, (col+1)/4, row/4);
			}

			for(int co = 0; co < CO_N; co++) {
				set_bias_1x1_new(FM_buf_acc, bias_buf[1 + co],0,mask_bound);
				CONV_1x1_new(FM_buf2, FM_buf_acc, weight_buf_1x1[co],0,mask_bound);
				Relu_Max_Pooling(FM_buf_acc, DDR_dw1_pool_out_PL_burst, DDR_dw2_pool_out_PL_burst, DDR_buf_burst, 0, co, col, row, col/4, row/4, 1);

			}
		}
	}


	printf("DW1 Done\n");


	/////////////////////////////// DW2_CONV_3x3 + DW2_CONV_1x1 + POOL ////////////////////////////
	/// DW2_CONV_3x3
	weight_3x3_index += CI_N;
	bias_3x3_index += CI_N + CO_N;
	weight_1x1_index += CO_N * CI_N;
	bias_1x1_index += CO_N + CO_N;

	load_weight_3x3_from_axi(weight_buf_3x3[1], conv_weight_3x3_all[weight_3x3_index + 0]);
	load_weight_3x3_from_axi(weight_buf_3x3[2], conv_weight_3x3_all[weight_3x3_index + 1]);
	load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + 0]);
	load_bias_from_axi(bias_buf[1], bias_all[bias_3x3_index + 1]);

	CI_N = 64 / 32;
	CO_N = 96 / 32;

	for(int row = 0; row < 4; row++) {
		for(int col = 0; col < 4; col++) {

			load_dw1_pool_from_DDR(DDR_dw1_pool_out_PL_burst, FM_buf1, 0, col, row, col/2, row/2);
			set_bias_3x3_new(FM_buf2, bias_buf[0 + 0],0,mask_bound);
			DW_CONV_3x3_new(FM_buf1, FM_buf2, weight_buf_3x3[1 + 0], bias_buf[0 + 0], 1,0,0,mask_bound);

			load_dw1_pool_from_DDR(DDR_dw1_pool_out_PL_burst, FM_buf4, 1, col, row, col/2, row/2);
			set_bias_3x3_new(FM_buf3, bias_buf[0 + 1],0,mask_bound);
			DW_CONV_3x3_new(FM_buf4, FM_buf3, weight_buf_3x3[1 + 1], bias_buf[0 + 1], 1,0,0,mask_bound);

			for(int co = 0; co < CO_N; co++) {
				load_bias_from_axi(bias_buf[2], bias_all[bias_1x1_index + co]);
				set_bias_1x1_new(FM_buf_acc, bias_buf[2],0,mask_bound);

				load_weight_1x1_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1_index + 0 + co * CI_N]);
				CONV_1x1_new(FM_buf2, FM_buf_acc, weight_buf_1x1[0],0,mask_bound);

				load_weight_1x1_from_axi(weight_buf_1x1[1], conv_weight_1x1_all[weight_1x1_index + 1 + co * CI_N]);
				CONV_1x1_new(FM_buf3, FM_buf_acc, weight_buf_1x1[1],0,mask_bound);

				Relu_Max_Pooling(FM_buf_acc, DDR_dw1_pool_out_PL_burst, DDR_dw2_pool_out_PL_burst, DDR_buf_burst, 0, co, col, row, col/2, row/2, 2);
			}
		}
	}


	printf("DW2 Done\n");


	/////////////////////////////// DW3_CONV_3x3 + DW3_CONV_1x1 + POOL ////////////////////////////


	/// DW3_CONV_3x3
	// output in DDR_buf[0] - DDR_buf[2]

	weight_3x3_index += CI_N;
	bias_3x3_index += CI_N + CO_N;
	weight_1x1_index += CO_N * CI_N;
	bias_1x1_index += CO_N + CO_N;

	load_weight_3x3_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[weight_3x3_index + 0]);
	load_weight_3x3_from_axi(weight_buf_3x3[1], conv_weight_3x3_all[weight_3x3_index + 1]);
	load_weight_3x3_from_axi(weight_buf_3x3[2], conv_weight_3x3_all[weight_3x3_index + 2]);
	load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + 0]);
	load_bias_from_axi(bias_buf[1], bias_all[bias_3x3_index + 1]);
	load_bias_from_axi(bias_buf[2], bias_all[bias_3x3_index + 2]);

	CI_N = 96 / 32;
	CO_N = 192 / 32;
	for(int row = 0; row < 2; row++) {
		for(int col = 0; col < 2; col++) {

			load_dw2_pool_from_DDR(DDR_dw2_pool_out_PL_burst, FM_buf1, 0, col, row, col/1, row/1 );
			set_bias_3x3_new(FM_buf2, bias_buf[0 + 0],0,mask_bound);
			DW_CONV_3x3_new(FM_buf1, FM_buf2, weight_buf_3x3[0 + 0], bias_buf[0 + 0], 1,0,0,mask_bound);

			load_dw2_pool_from_DDR(DDR_dw2_pool_out_PL_burst, FM_buf1, 1, col, row, col/1, row/1 );
			set_bias_3x3_new(FM_buf3, bias_buf[0 + 1],0,mask_bound);
			DW_CONV_3x3_new(FM_buf1, FM_buf3, weight_buf_3x3[0 + 1], bias_buf[0 + 1], 1,0,0,mask_bound);

			load_dw2_pool_from_DDR(DDR_dw2_pool_out_PL_burst, FM_buf1, 2, col, row, col/1, row/1 );
			set_bias_3x3_new(FM_buf4, bias_buf[0 + 2],0,mask_bound);
			DW_CONV_3x3_new(FM_buf1, FM_buf4, weight_buf_3x3[0 + 2], bias_buf[0 + 2], 1,0,0,mask_bound);

			/// DW3_CONV_1x1
			for(int co = 0; co < CO_N; co++) {

				load_bias_from_axi(bias_buf[3], bias_all[bias_1x1_index + co]);
				set_bias_1x1_new(FM_buf_acc, bias_buf[3],0,mask_bound);

				load_weight_1x1_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1_index + 0 + co * CI_N]);
				CONV_1x1_new(FM_buf2, FM_buf_acc, weight_buf_1x1[0],0,mask_bound);

				load_weight_1x1_from_axi(weight_buf_1x1[1], conv_weight_1x1_all[weight_1x1_index + 1 + co * CI_N]);
				CONV_1x1_new(FM_buf3, FM_buf_acc, weight_buf_1x1[1],0,mask_bound);

				load_weight_1x1_from_axi(weight_buf_1x1[2], conv_weight_1x1_all[weight_1x1_index + 2 + co * CI_N]);
				CONV_1x1_new(FM_buf4, FM_buf_acc, weight_buf_1x1[2],0,mask_bound);

				relu_copy_buf_to_DDR_acc_new(DDR_buf_burst, 100 + co + (col*2 + row) * CO_N, FM_buf_acc, 0, 0,0, mask_bound);
				Relu_Max_Pooling(FM_buf_acc, DDR_dw1_pool_out_PL_burst, DDR_dw2_pool_out_PL_burst, DDR_buf_burst, 6 + co, co, col, row, col, row, 3 );
			}
		}
	}


	printf("DW3 Done\n");
	
	/////////////////////////////////////////New branch group Conv5x5 group size = 32//////////////////////////////////////////////



	load_bias_from_axi(bias_buf[0], bias_all[106]);
	set_bias_5x5(FM_buf_acc, bias_buf[0]);
	load_weight_5x5_from_axi(weight_buf_5x5[0], conv_weight_5x5_all[0]);
	compute_load_buf_from_DDR_new_branch_channel(FM_buf1, DDR_buf_burst, 6, 0, 0, 0); //channel shuffle in group convolution
	DW_CONV_5x5(FM_buf1, FM_buf_acc, weight_buf_5x5[0], bias_buf[0],0,0);
	DW_CONV_5x5(FM_buf1, FM_buf_acc, weight_buf_5x5[0], bias_buf[0], 0, 1);
	DW_CONV_5x5(FM_buf1, FM_buf_acc, weight_buf_5x5[0], bias_buf[0], 1, 0);
	DW_CONV_5x5(FM_buf1, FM_buf_acc, weight_buf_5x5[0], bias_buf[0], 1, 1);


	load_weight_5x5_from_axi(weight_buf_5x5[1], conv_weight_5x5_all[1]);
	compute_load_buf_from_DDR_new_branch_channel(FM_buf2, DDR_buf_burst, 6, 1, 0, 0);//channel shuffle in group convolution
	DW_CONV_5x5(FM_buf2, FM_buf_acc, weight_buf_5x5[1], bias_buf[0], 0, 0);
	DW_CONV_5x5(FM_buf2, FM_buf_acc, weight_buf_5x5[1], bias_buf[0], 0, 1);
	DW_CONV_5x5(FM_buf2, FM_buf_acc, weight_buf_5x5[1], bias_buf[0], 1, 0);
	DW_CONV_5x5(FM_buf2, FM_buf_acc, weight_buf_5x5[1], bias_buf[0], 1, 1);



	load_weight_5x5_from_axi(weight_buf_5x5[2], conv_weight_5x5_all[2]);
	compute_load_buf_from_DDR_new_branch_channel(FM_buf3, DDR_buf_burst, 6, 2, 1, 0);//channel shuffle in group convolution
	DW_CONV_5x5(FM_buf3, FM_buf_acc, weight_buf_5x5[2], bias_buf[0], 0, 0);
	DW_CONV_5x5(FM_buf3, FM_buf_acc, weight_buf_5x5[2], bias_buf[0], 0, 1);
	DW_CONV_5x5(FM_buf3, FM_buf_acc, weight_buf_5x5[2], bias_buf[0], 1, 0);
	DW_CONV_5x5(FM_buf3, FM_buf_acc, weight_buf_5x5[2], bias_buf[0], 1, 1);



	load_weight_5x5_from_axi(weight_buf_5x5[0], conv_weight_5x5_all[3]);
	compute_load_buf_from_DDR_new_branch_channel(FM_buf1, DDR_buf_burst, 6, 3, 1, 0);//channel shuffle in group convolution
	DW_CONV_5x5(FM_buf1, FM_buf_acc, weight_buf_5x5[0], bias_buf[0], 0, 0);
	DW_CONV_5x5(FM_buf1, FM_buf_acc, weight_buf_5x5[0], bias_buf[0], 0, 1);
	DW_CONV_5x5(FM_buf1, FM_buf_acc, weight_buf_5x5[0], bias_buf[0], 1, 0);
	DW_CONV_5x5(FM_buf1, FM_buf_acc, weight_buf_5x5[0], bias_buf[0], 1, 1);



	load_weight_5x5_from_axi(weight_buf_5x5[1], conv_weight_5x5_all[4]);
	compute_load_buf_from_DDR_new_branch_channel(FM_buf2, DDR_buf_burst, 6, 4, 1, 1);//channel shuffle in group convolution
	DW_CONV_5x5(FM_buf2, FM_buf_acc, weight_buf_5x5[1], bias_buf[0], 0, 0);
	DW_CONV_5x5(FM_buf2, FM_buf_acc, weight_buf_5x5[1], bias_buf[0], 0, 1);
	DW_CONV_5x5(FM_buf2, FM_buf_acc, weight_buf_5x5[1], bias_buf[0], 1, 0);
	DW_CONV_5x5(FM_buf2, FM_buf_acc, weight_buf_5x5[1], bias_buf[0], 1, 1);



	load_weight_5x5_from_axi(weight_buf_5x5[2], conv_weight_5x5_all[5]);
	compute_load_buf_from_DDR_new_branch_channel(FM_buf3, DDR_buf_burst, 6, 5, 1, 1);//channel shuffle in group convolution
	DW_CONV_5x5(FM_buf3, FM_buf_acc, weight_buf_5x5[2], bias_buf[0], 0, 0);
	DW_CONV_5x5(FM_buf3, FM_buf_acc, weight_buf_5x5[2], bias_buf[0], 0, 1);
	DW_CONV_5x5(FM_buf3, FM_buf_acc, weight_buf_5x5[2], bias_buf[0], 1, 0);
	DW_CONV_5x5(FM_buf3, FM_buf_acc, weight_buf_5x5[2], bias_buf[0], 1, 1);

	/////////////////////////////////////////New branch group Conv5x5 end//////////////////////////////////////////////


	/////////////////////////////////////////prepare buffers which can be used later//////////////////////////////////////////////
	clear_buffer_new(FM_buf2);
	clear_buffer_new(FM_buf4);
	clear_buffer_new(FM_buf3);
	local_buf_copy_new(FM_buf1, FM_buf_acc);
	/////////////////////////////////////////prepare buffers end//////////////////////////////////////////////

	
	/////////////////////////////////////////Relu and Maxpooling 5x5 //////////////////////////////////////////////
	//We separate maxpooling 5x5 into three parts: the first part is relu_maxpooling 2x2, the second part is relu_maxpooling 2x2, the final part is relu_maxpooling 1x2

	compute_Relu_Max_Pooling_2x2(FM_buf1, FM_buf2, 1, 10, 1, 20, 0, 0);
	compute_Relu_Max_Pooling_2x2(FM_buf1, FM_buf2, 1, 10, 23, 42, 0, 1);
	compute_Relu_Max_Pooling_2x2(FM_buf1, FM_buf2, 13, 22, 1, 20, 1, 0);
	compute_Relu_Max_Pooling_2x2(FM_buf1, FM_buf2, 13, 22, 23, 42, 1, 1);

	compute_Relu_Max_Pooling_2x2(FM_buf2, FM_buf3, 1, 5, 1, 10, 0, 0);
	compute_Relu_Max_Pooling_2x2(FM_buf2, FM_buf3, 1, 5, 13, 22, 0, 1);
	compute_Relu_Max_Pooling_2x2(FM_buf2, FM_buf3, 8, 12, 1, 10, 1, 0);
	compute_Relu_Max_Pooling_2x2(FM_buf2, FM_buf3, 8, 12, 13, 22, 1, 1);

	Relu_Max_Pooling_1x2(FM_buf3, FM_buf4);
	/////////////////////////////////////////Relu and Maxpooling 5x5 end//////////////////////////////////////////////




	/////////////////////////////////////////Conv3x3 + sigmoid in new branch//////////////////////////////////////////////
	load_weight_3x3_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[64]);
	clear_buffer_new(FM_buf1);
	DW_CONV_3x3_new(FM_buf4, FM_buf1, weight_buf_3x3[0], bias_buf[0],0,0,1, mask_bound);
	accumulate_sigmoid(FM_buf1, mask);
	/////////////////////////////////////////Conv3x3 + sigmoid in new branch end//////////////////////////////////////////////



	/////////////////////////////////////////generate confidence mask//////////////////////////////////////////////

	uint8 xmin1[1], ymin1[1], xmin2[1], ymin3[1];
	uint8 xmax1[1], ymax1[1], xmax2[1],  ymax3[1];
	uint8 xmin3[1], xmin4[1], ymin2[1], ymin4[1];
	uint8 xmax3[1], xmax4[1], ymax2[1], ymax4[1];
	uint8 index1[1], index2[1], index3[1], index4[1];



	generate_mask_fix(xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2, xmin3, xmax3, ymin3, ymax3, xmin4, xmax4, ymin4, ymax4, mask, index1,index2,index3,index4);
	if (index1[0] == 0) {
		xmin1[0] = 0;
		xmax1[0] = 4;
		ymin1[0] = 0;
		ymax1[0] = 4;

	}
	if (index2[0] == 0) {
		xmin2[0] = 0;
		xmax2[0] = 4;
		ymin2[0] = 5;
		ymax2[0] = 9;

	}
	if (index3[0] == 0) {
		xmin3[0] = 5;
		xmax3[0] = 9;
		ymin3[0] = 0;
		ymax3[0] = 4;

	}
	if (index4[0] == 0) {
		xmin4[0] = 5;
		xmax4[0] = 9;
		ymin4[0] = 5;
		ymax4[0] = 9;

	}
	/////////////////////////////////////////generate confidence mask end//////////////////////////////////////////////

	/////////////////////////////////////////upsample confidence mask/////////////////////////////////////////////////
	
	mask_bound[0] = xmin1[0] * 4;
	mask_bound[1] = (xmax1[0] + 1) * 4;
	mask_bound[2] = ymin1[0] * 8;
	mask_bound[3] = (ymax1[0] + 1) * 8;

	mask_bound[4] = xmin2[0] * 4;
	mask_bound[5] = (xmax2[0] + 1) * 4;
	mask_bound[6] = ymin2[0] * 8;
	mask_bound[7] = (ymax2[0] + 1) * 8;

	mask_bound[8] = xmin3[0] * 4;
	mask_bound[9] = (xmax3[0] + 1) * 4;
	mask_bound[10] = ymin3[0] * 8;
	mask_bound[11] = (ymax3[0] + 1) * 8;

	mask_bound[12] = xmin4[0] * 4;
	mask_bound[13] = (xmax4[0] + 1) * 4;
	mask_bound[14] = ymin4[0] * 8;
	mask_bound[15] = (ymax4[0] + 1) * 8;

	mask_bound_reorg[0] = xmin1[0] * 8;
	mask_bound_reorg[1] = (xmax1[0] + 1) * 8;
	mask_bound_reorg[2] = ymin1[0] * 16;
	mask_bound_reorg[3] = (ymax1[0] + 1) * 16;

	mask_bound_reorg[4] = xmin2[0] * 8;
	mask_bound_reorg[5] = (xmax2[0] + 1) * 8;
	mask_bound_reorg[6] = (ymin2[0] - 5) * 16;
	mask_bound_reorg[7] = (ymax2[0] - 4) * 16;

	mask_bound_reorg[8] = (xmin3[0] - 5) * 8;
	mask_bound_reorg[9] = (xmax3[0] - 4) * 8;
	mask_bound_reorg[10] = ymin3[0] * 16;
	mask_bound_reorg[11] = (ymax3[0] + 1) * 16;

	mask_bound_reorg[12] = (xmin4[0] - 5) * 8;
	mask_bound_reorg[13] = (xmax4[0] - 4) * 8;
	mask_bound_reorg[14] = (ymin4[0] - 5) * 16;
	mask_bound_reorg[15] = (ymax4[0] - 4) * 16;


	printf("new branch done\n");


	/////////////////////////////// DW4_CONV_3x3 + DW4_CONV_1x1 ////////////////////////////

	/// DW4_CONV_3x3
	// input in DDR_buf[6] - DDR_buf[11]
	// output in DDR_buf[12] - DDR_buf[17]

	clear_buffer_new(FM_buf1);
	clear_buffer_new(FM_buf2);
	clear_buffer_new(FM_buf3);
	clear_buffer_new(FM_buf4);
	clear_buffer(FM_buf_acc);

	weight_3x3_index += CI_N;
	bias_3x3_index += CI_N + CO_N;
	weight_1x1_index += CO_N * CI_N;
	bias_1x1_index += CO_N + CO_N;

	CI_N = 192 / 32;

	/// conv3x3: ping-pong in: FM_buf1 and FM_buf3
	/// conv3x3: out: FM_buf2
	load_buf_from_DDR_new(FM_buf1, DDR_buf_burst, 6 + 0,1, mask_bound);
	for(int c = 0; c < CI_N; c++) {
		load_weight_3x3_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[weight_3x3_index + c]);
		load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + c]);
		set_bias_3x3_new(FM_buf2, bias_buf[0],1,mask_bound);

		if( c % 2 == 0 ) {
			load_buf_from_DDR_new(FM_buf3, DDR_buf_burst, 6 + c+1,1,mask_bound);
			DW_CONV_3x3_new(FM_buf1, FM_buf2, weight_buf_3x3[0], bias_buf[0], 0,1,0,mask_bound);
		}
		else {
			load_buf_from_DDR_new(FM_buf1, DDR_buf_burst, 6 + c+1,1,mask_bound);
			DW_CONV_3x3_new(FM_buf3, FM_buf2, weight_buf_3x3[0], bias_buf[0], 0,1,0,mask_bound);
		}

		relu_copy_buf_to_DDR_new(DDR_buf_burst, 12 + c, FM_buf2, 0, 0,1, mask_bound);
	}

	/// DW4_CONV_1x1
	// input in DDR_buf[12] - DDR_buf[17]
	// output in DDR_buf[18] - DDR_buf[29]

	CO_N = 384 / 32;
	for(int co = 0; co < CO_N; co++) {

		load_bias_from_axi(bias_buf[0], bias_all[bias_1x1_index + co]);
		set_bias_1x1_new(FM_buf_acc, bias_buf[0], 1,mask_bound);

		load_buf_from_DDR_new(FM_buf1, DDR_buf_burst, 12 + 0, 1,mask_bound);
		for(int ci = 0; ci < CI_N; ci++) {
			load_weight_1x1_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1_index + ci + co * CI_N]);

			if( ci % 2 == 0) {
				load_buf_from_DDR_new(FM_buf2, DDR_buf_burst, 12 + ci+1, 1,mask_bound);
				CONV_1x1_new(FM_buf1, FM_buf_acc, weight_buf_1x1[0], 1,mask_bound);
			}
			else {
				load_buf_from_DDR_new(FM_buf1, DDR_buf_burst, 12 + ci+1, 1,mask_bound);
				CONV_1x1_new(FM_buf2, FM_buf_acc, weight_buf_1x1[0], 1,mask_bound);
			}
		}
		relu_copy_buf_to_DDR_acc_new(DDR_buf_burst, 18 + co, FM_buf_acc, 0, 0, 1,mask_bound);
	}


	printf("DW4 Done\n");


	/////////////////////////////// DW5_CONV_3x3 + DW5_CONV_1x1 ////////////////////////////

	/// DW5_CONV_3x3
	// input in DDR_buf[18] - DDR_buf[29]
	// output in DDR_buf[30] - DDR_buf[41]

	weight_3x3_index += CI_N;
	bias_3x3_index += CI_N + CO_N;
	weight_1x1_index += CO_N * CI_N;
	bias_1x1_index += CO_N + CO_N;

	CI_N = 384 / 32;
	load_buf_from_DDR_new(FM_buf1, DDR_buf_burst, 18 + 0, 1,mask_bound);
	for(int c = 0; c < CI_N; c++) {
		load_weight_3x3_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[weight_3x3_index + c]);
		load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + c]);
		set_bias_3x3_new(FM_buf2, bias_buf[0], 1,mask_bound);

		if( c % 2 == 0) {
			load_buf_from_DDR_new(FM_buf3, DDR_buf_burst, 18 + c+1, 1,mask_bound);
			DW_CONV_3x3_new(FM_buf1, FM_buf2, weight_buf_3x3[0], bias_buf[0], 0, 1,0,mask_bound);
		}
		else {
			load_buf_from_DDR_new(FM_buf1, DDR_buf_burst, 18 + c+1, 1,mask_bound);
			DW_CONV_3x3_new(FM_buf3, FM_buf2, weight_buf_3x3[0], bias_buf[0], 0, 1,0,mask_bound);
		}


		relu_copy_buf_to_DDR_new(DDR_buf_burst, 30 + c, FM_buf2, 0, 0, 1,mask_bound);
	}

	/// DW5_CONV_1x1
	// input in DDR_buf[30] - DDR_buf[41]
	// output in DDR_buf[42] - DDR_buf[57]

	CO_N = 512 / 32;
	for(int co = 0; co < CO_N; co++) {

		load_bias_from_axi(bias_buf[0], bias_all[bias_1x1_index + co]);
		set_bias_1x1_new(FM_buf_acc, bias_buf[0], 1,mask_bound);

		load_buf_from_DDR_new(FM_buf2, DDR_buf_burst, 30 + 0, 1,mask_bound);
		for(int ci = 0; ci < CI_N; ci++) {
			load_weight_1x1_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1_index + ci + co * CI_N]);

			if( ci % 2 == 0) {
				load_buf_from_DDR_new(FM_buf1, DDR_buf_burst, 30 + ci+1, 1,mask_bound);
				CONV_1x1_new(FM_buf2, FM_buf_acc, weight_buf_1x1[0], 1,mask_bound);
			}
			else {
				load_buf_from_DDR_new(FM_buf2, DDR_buf_burst, 30 + ci+1, 1,mask_bound);
				CONV_1x1_new(FM_buf1, FM_buf_acc, weight_buf_1x1[0], 1,mask_bound);
			}
		}
		relu_copy_buf_to_DDR_acc_new(DDR_buf_burst, 42 + co, FM_buf_acc, 0, 0, 1,mask_bound);
	}


	printf("DW5 Done\n");

	/////////////////////////////// CONCAT DW3_CONV_1x1_OUT & DW5_CONV_1x1_OUT /////////////
	/////////////////////////////// DW6_CONV_3x3 + DW6_CONV_1x1 ////////////////////////////

	//// DW6_CONV_3x3
	weight_3x3_index += CI_N;
	bias_3x3_index += CI_N + CO_N;
	weight_1x1_index += CO_N * CI_N;
	bias_1x1_index += CO_N + CO_N;


	// First half: 0~767 channels from DW3_conv_1x1_output
	// Output of DW3_CONV_1x1_OUT are stored in DDR_buf[100] to DDR_buf[123]
	// Output of first half are stored in DDR_buf[58] to DDR_buf[81]
	CI_N = 768 / 32;
	for(int c = 0; c < CI_N; c+=4) {

		// From DDR_buf[100] to DDR_buf[123]
		// Img 0: 100 ~ 105
		// Img 1: 106 ~ 111
		// Img 2: 112 ~ 117
		// Img 3: 118 ~ 123

		load_and_reorg_new(DDR_buf_burst, 100 + c/4,         DDR_buf_burst, 100 + c/4 + 6,
					   DDR_buf_burst, 100 + c/4 + 6 * 2, DDR_buf_burst, 100 + c/4 + 6 * 3,
                       FM_buf1, FM_buf3, FM_buf4, FM_buf_acc,mask_bound_reorg);

		//// 1/4 channels
		load_weight_3x3_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[weight_3x3_index + c + 0]);
		load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + c + 0]);
		set_bias_3x3_new(FM_buf2, bias_buf[0],1,mask_bound);
		DW_CONV_3x3_new(FM_buf1, FM_buf2, weight_buf_3x3[0], bias_buf[0], 0,1, 0,mask_bound);
		local_buf_copy(FM_buf1, FM_buf_acc);
		relu_copy_buf_to_DDR_new(DDR_buf_burst, 58 + c + 0, FM_buf2, 0, 0, 1,mask_bound);

		//// 2/4 channels
		load_weight_3x3_from_axi(weight_buf_3x3[1], conv_weight_3x3_all[weight_3x3_index + c + 1]);
		load_bias_from_axi(bias_buf[1], bias_all[bias_3x3_index + c + 1]);
		set_bias_3x3_new(FM_buf2, bias_buf[1], 1,mask_bound);
		DW_CONV_3x3_new(FM_buf3, FM_buf2, weight_buf_3x3[1], bias_buf[1], 0, 1,0,mask_bound);
		relu_copy_buf_to_DDR_new(DDR_buf_burst, 58 + c + 1, FM_buf2, 0, 0,1, mask_bound);

		//// 3/4 channels
		load_weight_3x3_from_axi(weight_buf_3x3[2], conv_weight_3x3_all[weight_3x3_index + c + 2]);
		load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + c + 2]);
		set_bias_3x3_new(FM_buf2, bias_buf[0], 1,mask_bound);
		DW_CONV_3x3_new(FM_buf4, FM_buf2, weight_buf_3x3[2], bias_buf[0], 0, 1,0,mask_bound);
		relu_copy_buf_to_DDR_new(DDR_buf_burst, 58 + c + 2, FM_buf2, 0, 0,1, mask_bound);

		//// 4/4 channels
		load_weight_3x3_from_axi(weight_buf_3x3[3], conv_weight_3x3_all[weight_3x3_index + c + 3]);
		load_bias_from_axi(bias_buf[1], bias_all[bias_3x3_index + c + 3]);
		set_bias_3x3_new(FM_buf2, bias_buf[1], 1,mask_bound);
		DW_CONV_3x3_new(FM_buf1, FM_buf2, weight_buf_3x3[3], bias_buf[1], 0,1, 0,mask_bound);
		relu_copy_buf_to_DDR_new(DDR_buf_burst, 58 + c + 3, FM_buf2, 0, 0, 1,mask_bound);
	}

	// Second half: 768~1280 channels from DW5_conv_1x1_output
	// Output of DW5_CONV_1x1_OUT are stored in DDR_buf[42] - DDR_buf[57]
	// Output of first half are stored in DDR_buf[82] to DDR_buf[97]
	CI_N = 512 / 32;

	load_buf_from_DDR_new(FM_buf1, DDR_buf_burst, 42 + 0, 1,mask_bound);
	for(int c = 0; c < CI_N; c++) {
		load_weight_3x3_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[weight_3x3_index + c + 24]);
		load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + c + 24]);
		set_bias_3x3_new(FM_buf2, bias_buf[0], 1,mask_bound);

		if( c % 2 == 0) {
			load_buf_from_DDR_new(FM_buf3, DDR_buf_burst, 42 + c+1, 1,mask_bound);
			DW_CONV_3x3_new(FM_buf1, FM_buf2, weight_buf_3x3[0], bias_buf[0], 0, 1,0,mask_bound);
		}
		else {
			load_buf_from_DDR_new(FM_buf1, DDR_buf_burst, 42 + c+1, 1,mask_bound);
			DW_CONV_3x3_new(FM_buf3, FM_buf2, weight_buf_3x3[0], bias_buf[0], 0, 1,0,mask_bound);
		}
		relu_copy_buf_to_DDR_new(DDR_buf_burst, 82 + c, FM_buf2, 0, 0, 1,mask_bound);
	}

	/// DW6_CONV_1x1
	// input in DDR_buf[58] - DDR_buf[97]
	// output in DDR_buf[98] - DDR_buf[100]

	bias_1x1_index += 24;
	CO_N = 96 / 32;
	CI_N = 1280 / 32;
	for(int co = 0; co < CO_N; co++) {

		load_bias_from_axi(bias_buf[0], bias_all[bias_1x1_index + co]);
		set_bias_1x1_new(FM_buf_acc, bias_buf[0], 1,mask_bound);

		load_buf_from_DDR_new(FM_buf2, DDR_buf_burst, 58 + 0, 1,mask_bound);
		for(int ci = 0; ci < CI_N; ci++) {
			load_weight_1x1_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1_index + ci + co * CI_N]);

			if( ci % 2 == 0) {
				load_buf_from_DDR_new(FM_buf1, DDR_buf_burst, 58 + ci+1, 1,mask_bound);
				CONV_1x1_new(FM_buf2, FM_buf_acc, weight_buf_1x1[0], 1,mask_bound);
			}
			else {
				load_buf_from_DDR_new(FM_buf2, DDR_buf_burst, 58 + ci+1, 1,mask_bound);
				CONV_1x1_new(FM_buf1, FM_buf_acc, weight_buf_1x1[0], 1,mask_bound);
			}
		}

		relu_copy_buf_to_DDR_acc_new(DDR_buf_burst, 98 + co, FM_buf_acc, 0, 0, 1,mask_bound);
	}


	printf("DW6 Done\n");


	/////////////////////////////// PW7_CONV_1x1 ////////////////////////////
	weight_1x1_index += CO_N * CI_N;
	// input in DDR_buf[98] - DDR_buf[100]
	// output in FM_buf_acc

	CO_N = 32 / 32;
	CI_N = 96 / 32;
	for(int co = 0; co < CO_N; co++) {
		clear_buffer(FM_buf_acc);
		for(int ci = 0; ci < CI_N; ci++) {
			load_weight_1x1_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1_index + ci + co * CI_N]);
			load_buf_from_DDR_new(FM_buf2, DDR_buf_burst, 98 + ci, 1,mask_bound);
			CONV_1x1_new(FM_buf2, FM_buf_acc, weight_buf_1x1[0], 1,mask_bound);
		}
	}

	printf("PW7 Done\n");


	compute_bounding_box(predict_boxes, constant,mask_bound);

	return;



}
