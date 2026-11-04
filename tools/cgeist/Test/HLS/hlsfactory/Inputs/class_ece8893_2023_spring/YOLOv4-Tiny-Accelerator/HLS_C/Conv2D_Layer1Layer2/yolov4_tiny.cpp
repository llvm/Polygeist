#include "conv.h"


void yolov4_tiny (
		fm_t input_image[3][416][416],
		wt_t conv_layer_1_weights[1024][1024][3][3],
		wt_t conv_layer_1_bias[1024],
		wt_t conv_layer_2_weights[1024][1024][3][3],
		wt_t conv_layer_2_bias[1024],
		fm_t output_feature_map[1024][416][416],
		fm_t output_L12_feature_map[64][104][104],
		fm_t input_feature_map[1024][416][416]
		)
{

	#pragma HLS INTERFACE m_axi depth=3*416*416 port=input_image bundle=fm
	#pragma HLS INTERFACE m_axi depth=32*3*3*3 port=conv_layer_1_weights bundle=wt
	#pragma HLS INTERFACE m_axi depth=32 port=conv_layer_1_bias bundle=wt
	#pragma HLS INTERFACE m_axi depth=32*64*3*3 port=conv_layer_2_weights bundle=wt
	#pragma HLS INTERFACE m_axi depth=64 port=conv_layer_2_bias bundle=wt
	#pragma HLS INTERFACE m_axi depth=1024*416*416 port=output_feature_map bundle=fm	
	#pragma HLS INTERFACE m_axi depth=64*104*104 port=output_L12_feature_map bundle=fm	
	#pragma HLS INTERFACE m_axi depth=1024*416*416 port=input_feature_map bundle=fm	
	#pragma HLS INTERFACE s_axilite register port=return

	for (int c = 0; c < IN_FM_DEPTH1; c++)
	{
		for (int i = 0; i < IN_FM_HEIGHT1; i++)
		{
			for (int j = 0; j < IN_FM_WIDTH1; j++)
			{
				input_feature_map[c][i][j] = input_image[c][i][j];
			}
		}
	}


	tiled_conv_L1 (input_image, conv_layer_1_weights, conv_layer_1_bias, output_feature_map);
	copy_output_to_input_L2(input_feature_map,output_feature_map);
	tiled_conv_L2 (input_feature_map, conv_layer_2_weights, conv_layer_2_bias, output_feature_map);

	for (int c = 0; c < OUT_FM_DEPTH2; c++)
	{
		for (int i = 0; i < OUT_FM_HEIGHT2; i++)
		{
			for (int j = 0; j < OUT_FM_WIDTH2; j++)
			{
				output_L12_feature_map[c][i][j] = output_feature_map[c][i][j];			
			}
		}
	}
}
