#include <cstddef>
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include "hls_stream.h"
#include <iostream>
#include <fstream>
#include <cmath>



//#define CSIM_DEBUG
//#define CSIM_CMP_OUTPUT


// for .range(Hi, Lo)
#define FM_RG			8
#define FM_ACC_RG		11
#define WT_RG			10

 //#define FM_RG			15
 //#define FM_ACC_RG		15
// #define WT_RG			15



#ifdef CSIM_DEBUG
	typedef float FIX_32_4;	//fix point
	typedef float FIX_32_25;	//fix point
	typedef float FIX_FM;	//fix point for feature map
	typedef float FIX_FM_acc;	//fix point for feature map
	typedef float FIX_FM_last;
	typedef float FIX_WT;	//fix point for weights
	typedef float FIX_32_16;
	typedef float FIX_32_10;
	typedef float FIX_32_12;
	typedef float FIX_16_6;
	typedef float FIX_16_5;
	typedef float FIX_16_4;
	typedef float FIX_16_10;

	typedef float uint8;
	typedef float uint16;
	typedef float uint128;
	typedef float uint256;
	typedef float uint512;

#else

	typedef ap_fixed<9,  3, AP_RND, AP_SAT> FIX_FM;	//fix point for feature map
	typedef ap_fixed<12, 4, AP_RND, AP_SAT> FIX_FM_acc;	//fix point for accumulation
	typedef ap_fixed<11, 4, AP_RND, AP_SAT> FIX_WT;	//fix point for weights
    //typedef ap_fixed<16,  8, AP_RND, AP_SAT> FIX_FM;	//fix point for feature map
	//typedef ap_fixed<16, 8, AP_RND, AP_SAT> FIX_FM_acc;	//fix point for accumulation
	//typedef ap_fixed<16, 8, AP_RND, AP_SAT> FIX_WT;	//fix point for weights




	typedef ap_fixed<16, 8, AP_RND, AP_SAT> FIX_16_8;
	typedef ap_fixed<16, 6, AP_RND, AP_SAT> FIX_16_6;
	typedef ap_fixed<16, 5, AP_RND, AP_SAT> FIX_16_5;
	typedef ap_fixed<16, 4, AP_RND, AP_SAT> FIX_16_4;
	typedef ap_fixed<16, 3, AP_RND, AP_SAT> FIX_16_3;
	typedef ap_fixed<16, 10, AP_RND, AP_SAT> FIX_16_10;
	typedef ap_fixed<32,16, AP_RND, AP_SAT> FIX_32_16;
	typedef ap_fixed<32,12, AP_RND, AP_SAT> FIX_32_12;
	typedef ap_fixed<32,10, AP_RND, AP_SAT> FIX_32_10;
	typedef ap_fixed<32, 4, AP_RND, AP_SAT> FIX_32_4;
	typedef ap_fixed<32, 7, AP_RND, AP_SAT> FIX_32_7;
	typedef ap_fixed<32,25, AP_RND, AP_SAT> FIX_32_25;

	typedef ap_uint<2> uint2;
	typedef ap_uint<4> uint4;
	typedef ap_uint<8> uint8;
	typedef ap_uint<16> uint16;
	typedef ap_uint<256> uint256;
	typedef ap_uint<512> uint512;


#endif
//int golden_model();

void MaskNet(uint8 image_in_raw_pad[3*162*2*322*2],

				uint512 fix_conv_weight_1x1_all_512bit[1000][32],
				uint512 fix_conv_weight_3x3_all_512bit[1000][3][3],
				uint512 fix_bias_all_512bit[500],
	            uint512 fix_conv_weight_5x5_all_512bit[6][5][5],
				

				uint512 DDR_dw1_pool_out_PL[64/32*82*2*162*2],
				uint512 DDR_dw2_pool_out_PL[96/32*42*2*82*2],

				uint512 DDR_buf[128*44*84],
	            //uint512 DDR_buf_burst_new_branch1[6 * 88 * 88],
	            //uint512 DDR_buf_burst_new_branch2[3*44*84],

				float debug[10][32][44][84],
				float predict_box[4][5],
				int constant[4][3]
);

void DW_CONV_3x3(FIX_FM bottom[32][44][84],
					FIX_FM top[32][44][84],
					FIX_WT weight[32][3][3],
					FIX_WT bias[32],
					int relu
);

void CONV_1x1(FIX_FM bottom[32][44][84],
			  FIX_FM_acc top[32][44][84],
			  FIX_WT weights[32][32]
);

void DW_CONV_5x5(FIX_FM bottom[32][44][84],
	FIX_FM_acc top[32][44][84],
	FIX_WT weights[32][5][5],
	FIX_WT bias[32],
	int offsetx,
	int offsety
);


void DW_CONV_3x3_new(FIX_FM bottom[32][44][84],
	FIX_FM top[32][44][84],
	FIX_WT weights[32][3][3],
	FIX_WT bias[32],
	int relu,  int index,int sigmoid, uint8 mask_bound[16]);

void CONV_1x1_new(FIX_FM bottom[32][44][84],
	FIX_FM_acc top[32][44][84],
	FIX_WT weights[32][32],
	int index, uint8 mask_bound[16]);

float max_2(float a1, float a2);
float max_5(float a1, float a2, float a3, float a4, float a5);


void fill_output_dw1_conv3x3( float buf[32][44][84], int ch, int col, int row);
void fill_output_dw1_conv1x1( float buf[32][44][84], int ch, int col, int row);
void fill_output_dw1_pool( float buf[32][44][84], int ch, int col, int row);

void fill_output_dw2_conv3x3( float buf[32][44][84], int ch, int col, int row);
void fill_output_dw2_conv1x1( float buf[32][44][84], int ch, int col, int row);
void fill_output_dw2_pool( float buf[32][44][84], int ch, int col, int row);

void fill_output_dw3_conv3x3( float buf[32][44][84], int ch, int col, int row);
void fill_output_dw3_conv1x1( float buf[32][44][84], int ch, int col, int row);
void fill_output_dw3_pool( float buf[32][44][84], int ch, int col, int row);

void fill_output_dw4_conv3x3( float buf[32][44][84], int ch, int col, int row);
void fill_output_dw4_conv1x1( float buf[32][44][84], int ch, int col, int row);

void fill_output_dw5_conv3x3( float buf[32][44][84], int ch, int col, int row);
void fill_output_dw5_conv1x1( float buf[32][44][84], int ch, int col, int row);

void fill_output_dw6_conv3x3( float buf[32][44][84], int ch, int col, int row);
void fill_output_dw6_conv1x1( float buf[32][44][84], int ch, int col, int row);

void fill_output_pw7_conv1x1( float buf[32][44][84], int ch, int col, int row);

void PL_golden_compare_dw1_conv3x3();
void PL_golden_compare_dw1_conv1x1();
void PL_golden_compare_dw1_pool();

void PL_golden_compare_dw2_conv3x3();
void PL_golden_compare_dw2_conv1x1();
void PL_golden_compare_dw2_pool();

void PL_golden_compare_dw3_conv3x3();
void PL_golden_compare_dw3_conv1x1();
void PL_golden_compare_dw3_pool();

void PL_golden_compare_dw4_conv3x3();
void PL_golden_compare_dw4_conv1x1();

void PL_golden_compare_dw5_conv3x3();
void PL_golden_compare_dw5_conv1x1();

void PL_golden_compare_dw6_conv3x3();
void PL_golden_compare_dw6_conv1x1();

void PL_golden_compare_pw7_conv1x1();
