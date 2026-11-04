/*
Conv1d_1 adds bias and ReLU inside
Maxpooling1 only downsample
Conv1d_2 adds bias
Maxpooling2 add ReLu and downsample
Conv1d_3 adds bias
Maxpooling3 add ReLU and downsample
Conv1d_4 adds bias
Maxpooling4 add ReLU and downsample
Maxpooling 4 only downsample
*/

#include "conv.h"
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

fm_t fixp_conv_layer_output_feature_map1[1][32][187];
fm_t fixp_conv_layer_output_feature_map1_max[1][32][92];
fm_t fixp_conv_layer_output_feature_map2[1][32][92];
fm_t fixp_conv_layer_output_feature_map2_max[1][32][44];
fm_t fixp_conv_layer_output_feature_map3[1][32][44];
fm_t fixp_conv_layer_output_feature_map4[1][32][20];
fm_t fixp_conv_layer_output_feature_map3_max[1][32][20];
fm_t fixp_conv_layer_output_feature_map4_max[1][32][8];
fm_t fixp_conv_layer_output_feature_map5_max[1][32][2];
fm_t fixp_dense1_output[1][32];
fm_t fixp_conv_layer_output_feature_flat[1][64];


void conv1d_1 ( //conv1 code
    fm_t Y_buf[1][32][187], //ignore first dimension, 32 output channels, 187 signal length
    fm_t X_buf[1][1][187], //ignore first dimension, 1 input channel, signal length
    wt_t W_buf[32][1][5], //32 out channels, kernel size:5
    wt_t B_buf[32] //32 outchannels
)
{
    //#pragma HLS inline off
    //#pragma HLS array_partition variable=Y_buf dim=3 complete
    #pragma HLS array_partition variable=X_buf dim=3 cyclic factor=5
    #pragma HLS array_partition variable=W_buf dim=3 cyclic factor=5
    //better performance/resource utilization when partioning cyclicly by kernel size(5) rather than doing complete
    // expect input of size 1, 1, 187
    // padding 2 for kernel size 5

    conv1d_1a: 
    for (int c = 0; c < 32; c++) 
    { 
        //#pragma HLS pipeline 
        conv1d_1b:
        for (int l = 0; l < 187; l++)
        { 
            #pragma HLS pipeline 
            conv1d_1c:
            for (int k = 0; k < 5; k++)
            {
                if (l + k < 2 || l + k > 188)
                {
                    Y_buf[0][c][l] += 0;
                }
                else
                {
                    Y_buf[0][c][l] += X_buf[0][0][l + k - 2] * W_buf[c][0][k];
                }
            }

            // Add ReLU
            Y_buf[0][c][l] += B_buf[c]; //adds bias

            if (Y_buf[0][c][l] < 0)
            {
                Y_buf[0][c][l] = 0; // Set to zero negative values
            }
        }
    }
}

void conv1d_2 ( //conv2 code
    fm_t Y_buf[1][32][92], //32 output channels, 187 signal length
    fm_t X_buf[1][32][92], //input channels, signal length
    wt_t W_buf[32][32][5], //32 out channels, kernel size:5
    wt_t B_buf[32] //32 outchannels
)
{
    #pragma HLS array_partition variable=W_buf dim=3 cyclic factor=5
    #pragma HLS array_partition variable=X_buf dim=3 cyclic factor=5
    //better performance/resource utilization when partioning cyclicly by kernel size(5) rather than doing complete
    //expects input of size 1, 32, 92
    //padding 2 for kernel size 5   
    conv1d_2a:
    for(int x = 0; x < 32; x++) { //input channels
        conv1d_2c:
        for(int l = 0; l < 92; l++) { //signal length
            Y_buf[0][x][l] += B_buf[x];
            conv1d_2b:
            for(int c = 0; c < 32; c++) { //output channels
                // conv1d_2c:
                // for(int l = 0; l < 92; l++) { //signal length
                #pragma HLS pipeline
                conv1d_2d:
                for(int k = 0; k < 5; k++) {
                    if(l+k < 2 || l+k > 93) {
                        Y_buf[0][c][l] += 0;
                    } else {
                        Y_buf[0][c][l] += X_buf[0][x][l+k-2] * W_buf[c][x][k];
                    }
                }
            }
        }
    }

}

void conv1d_3 ( //conv3+4 code
    fm_t Y_buf[1][32][44], //32 output channels, 187 signal length
    fm_t X_buf[1][32][44], //input channels, signal length
    wt_t W_buf[32][32][5], //32 out channels, kernel size:5
    wt_t B_buf[32] //32 outchannels
)
{
    //array partioning X_buf causes timing issues with negative slack so lines below are commented out
    //could be because we call this function twice back to back?
    #pragma HLS array_partition variable=W_buf dim=3 cyclic factor=5
    //#pragma HLS array_partition variable=X_buf dim=3 cyclic factor=5

    //expects input of size 1, 32, 92
    //padding 2 for kernel size 5   
    conv1d_3a:
    for(int x = 0; x < 32; x++) { //input channels
        conv1d_3c:
        for(int l = 0; l < 44; l++) { //signal length
            Y_buf[0][x][l] += B_buf[x];
            conv1d_3b:
            for(int c = 0; c < 32; c++) { //output channels
                // conv1d_3c:
                // for(int l = 0; l < 44; l++) { //signal length
                #pragma HLS pipeline
                conv1d_3d:
                for(int k = 0; k < 5; k++) {
                    if(l+k < 2 || l+k > 45) {
                        Y_buf[0][c][l] += 0;
                    } else {
                        Y_buf[0][c][l] += X_buf[0][x][l+k-2] * W_buf[c][x][k];
                    }
                }

            }
        }
    }

}
void conv1d_4 ( //conv3+4 code
    fm_t Y_buf[1][32][20], //32 output channels, 187 signal length
    fm_t X_buf[1][32][20], //input channels, signal length
    wt_t W_buf[32][32][5], //32 out channels, kernel size:5
    wt_t B_buf[32] //32 outchannels
)
{
    //array partioning X_buf causes timing issues with negative slack so lines below are commented out
    //could be because we call this function twice back to back?
    #pragma HLS array_partition variable=W_buf dim=3 cyclic factor=5
    //#pragma HLS array_partition variable=X_buf dim=3 cyclic factor=5

    //expects input of size 1, 32, 92
    //padding 2 for kernel size 5   
    conv1d_3a:
    for(int x = 0; x < 32; x++) { //input channels
        conv1d_3c:
        for(int l = 0; l < 20; l++) { //signal length
            Y_buf[0][x][l] += B_buf[x];
            conv1d_3b:
            for(int c = 0; c < 32; c++) { //output channels
                // conv1d_3c:
                // for(int l = 0; l < 44; l++) { //signal length
                #pragma HLS pipeline
                conv1d_3d:
                for(int k = 0; k < 5; k++) {
                    if(l+k < 2 || l+k > 21) {
                        Y_buf[0][c][l] += 0;
                    } else {
                        Y_buf[0][c][l] += X_buf[0][x][l+k-2] * W_buf[c][x][k];
                    }
                }
            }
        }
    }

}

void max_pooling1(
    fm_t Y_buf[1][32][187],
    fm_t Y_maxpool_buf[1][32][92])
{
    //#pragma HLS array_partition variable=Y_buf dim=3 cylic factor=5
    // Perform maxpooling
    int output_size = (187 - KERNEL_SIZE) / STRIDE + 1;
    fm_t max_val = -10000000;
    //std::cout << "Max value:  " << max_val << std::endl;
    for (int j = 0; j < 32; j++) //32
    { // loop over channels
        for (int k = 0; k < output_size; k++)
        { // loop over output width
            //#pragma HLS pipeline 
            max_val = -10000000;
            // Compute max value in kernel window
            for (int l = 0; l < KERNEL_SIZE; l++)
            { // loop over kernel size
                int idx = k * STRIDE + l;
                fm_t val = Y_buf[0][j][idx];
                //std::cout << "Value:  " << val << std::endl;
                if (val > max_val)
                {
                    max_val = val;
                    //std::cout << "Max value:  " << max_val << std::endl;
                }
            }
            // Store max value in output tensor
            //std::cout << "Stored value in array:  " << max_val << std::endl;
            Y_maxpool_buf[0][j][k] = max_val;
        }
    }
}

void max_pooling2(
    fm_t Y_buf[1][32][92],
    fm_t Y_maxpool_buf[1][32][44])
{
    // Add ReLU
    for (int x = 0; x < 32; x++)
    { // input channels
        for (int l = 0; l < 92; l++)
        { // signal length
            if (Y_buf[0][x][l] < 0)
            {
                Y_buf[0][x][l] = 0;
            }
        }
    }

    // Perform maxpooling
    int output_size = (92 - KERNEL_SIZE) / STRIDE + 1;
    float max_val = -10000000;
    //std::cout << "Max value:  " << max_val << std::endl;
    for (int j = 0; j < 32; j++) //32
    { // loop over channels
        for (int k = 0; k < output_size; k++)
        { // loop over output width
            //std::cout << "Window:  " << k << std::endl;
            max_val = -10000000;
            // Compute max value in kernel window
            for (int l = 0; l < KERNEL_SIZE; l++)
            { // loop over kernel size
                int idx = k * STRIDE + l;
                float val = Y_buf[0][j][idx];
                //std::cout << "Value:  " << val << std::endl;
                if (val > max_val)
                {
                    max_val = val;
                    //std::cout << "Max value:  " << max_val << std::endl;
                }
            }
            // Store max value in output tensor
            //std::cout << "Stored value in array:  " << max_val << std::endl;
            Y_maxpool_buf[0][j][k] = max_val;
        }
    }
}

void max_pooling3(
    fm_t Y_buf[1][32][44],
    fm_t Y_maxpool_buf[1][32][20])
{
    // Add ReLU
    for (int x = 0; x < 32; x++)
    { // input channels
        for (int l = 0; l < 44; l++)
        { // signal length
            if (Y_buf[0][x][l] < 0)
            {
                Y_buf[0][x][l] = 0; // Set to zero negative values
            }
        }
    }
    // Perform maxpooling
    int output_size = (44 - KERNEL_SIZE) / STRIDE + 1;
    float max_val = -10000000;
    //std::cout << "Max value:  " << max_val << std::endl;
    for (int j = 0; j < 32; j++) //32
    { // loop over channels
        for (int k = 0; k < output_size; k++)
        { // loop over output width
            //std::cout << "Window:  " << k << std::endl;
            max_val = -10000000;
            // Compute max value in kernel window
            for (int l = 0; l < KERNEL_SIZE; l++)
            { // loop over kernel size
                int idx = k * STRIDE + l;
                float val = Y_buf[0][j][idx];
                //std::cout << "Value:  " << val << std::endl;
                if (val > max_val)
                {
                    max_val = val;
                    //std::cout << "Max value:  " << max_val << std::endl;
                }
            }
            // Store max value in output tensor
            //std::cout << "Stored value in array:  " << max_val << std::endl;
            Y_maxpool_buf[0][j][k] = max_val;
        }
    }
}

void max_pooling4(
    fm_t Y_buf[1][32][20],
    fm_t Y_maxpool_buf[1][32][8])
{
    //Add ReLU
    for (int x = 0; x < 32; x++)
    { // input channels
        for (int l = 0; l < 20; l++)
        { // signal length
            if (Y_buf[0][x][l] < 0)
            {
                Y_buf[0][x][l] = 0; // Set to zero negative values
            }
        }
    }
    // Perform maxpooling
    int output_size = (20 - KERNEL_SIZE) / STRIDE + 1;
    float max_val = -10000000;
    //std::cout << "Max value:  " << max_val << std::endl;
    for (int j = 0; j < 32; j++) //32
    { // loop over channels
        for (int k = 0; k < output_size; k++)
        { // loop over output width
            //std::cout << "Window:  " << k << std::endl;
            max_val = -10000000;
            // Compute max value in kernel window
            for (int l = 0; l < KERNEL_SIZE; l++)
            { // loop over kernel size
                int idx = k * STRIDE + l;
                float val = Y_buf[0][j][idx];
                //std::cout << "Value:  " << val << std::endl;
                if (val > max_val)
                {
                    max_val = val;
                    //std::cout << "Max value:  " << max_val << std::endl;
                }
            }
            // Store max value in output tensor
            //std::cout << "Stored value in array:  " << max_val << std::endl;
            Y_maxpool_buf[0][j][k] = max_val;
        }
    }
}

void max_pooling5(
    fm_t Y_buf[1][32][8],
    fm_t Y_maxpool_buf[1][32][2])
{
    // Perform maxpooling
    int output_size = (8 - KERNEL_SIZE) / STRIDE + 1;
    float max_val = -10000000;
    //std::cout << "Max value:  " << max_val << std::endl;
    for (int j = 0; j < 32; j++) //32
    { // loop over channels
        for (int k = 0; k < output_size; k++)
        { // loop over output width
            //std::cout << "Window:  " << k << std::endl;
            max_val = -10000000;
            // Compute max value in kernel window
            for (int l = 0; l < KERNEL_SIZE; l++)
            { // loop over kernel size
                int idx = k * STRIDE + l;
                float val = Y_buf[0][j][idx];
                //std::cout << "Value:  " << val << std::endl;
                if (val > max_val)
                {
                    max_val = val;
                    //std::cout << "Max value:  " << max_val << std::endl;
                }
            }
            // Store max value in output tensor
            //std::cout << "Stored value in array:  " << max_val << std::endl;
            Y_maxpool_buf[0][j][k] = max_val;
        }
    }
}

void flatten(
    fm_t in[1][32][2],
    fm_t out[1][64])
{
    #pragma HLS array_partition variable=in dim=3 complete
    #pragma HLS array_partition variable=in dim=2 complete
    #pragma HLS array_partition variable=out dim=2 complete

    #pragma HLS pipeline
    flatten_a:
    for(int i = 0; i < 32; i++){
        flatten_b:
        for(int j = 0; j < 2; j++){
            out[0][i*2+j] = in[0][i][j];
        }
    }
}



// pass flattened feature map through dense1
void dense1(
    fm_t fixp_conv_layer_output_feature_flat[1][64],
    fm_t fixp_dense1_bias[32],
    fm_t fixp_dense1_weights[32][64],
    fm_t fixp_dense1_output[1][32])
{
    #pragma HLS pipeline off
    dense1_a:
    for (int i = 0; i < 32; i++) {
        fixp_dense1_output[0][i] = fixp_dense1_bias[i];
        //#pragma HLS pipeline
        dense1_b:
        for (int j = 0; j < 64; j++) {
            fixp_dense1_output[0][i] += fixp_conv_layer_output_feature_flat[0][j] * fixp_dense1_weights[i][j];
        }
    }

    #pragma HLS unroll
    // apply ReLU activation function
    last_acc:
    for (int i = 0; i < 32; i++) {
        if (fixp_dense1_output[0][i] < 0) {
            fixp_dense1_output[0][i] = 0;
        }
    }
}

void dense2(
    fm_t fixp_dense1_output[1][32],
    fm_t fixp_dense2_bias[5],
    fm_t fixp_dense2_weights[5][32],
    fm_t output_feature_map[5])
{
    #pragma HLS pipeline off
    //#pragma HLS array_partition variable=output_feature_map dim=2 complete
    //#pragma HLS array_partition variable=fixp_dense2_weights dim=1 complete

    dense2_a:
    for(int i = 0; i < 5; i++) {
        output_feature_map[i] = fixp_dense2_bias[i];
        //#pragma HLS pipeline 
        dense2_b:
        for (int j = 0; j < 32; j++) {
            output_feature_map[i] += fixp_dense1_output[0][j] * fixp_dense2_weights[i][j];
        }
    }
}

void tiled_conv (
    fm_t input_feature_map[1][1][187],
    wt_t fixp_conv_layer_weights1[32][1][5],
    wt_t fixp_conv_layer_bias1[32],
    wt_t fixp_conv_layer_weights2[32][32][5],
    wt_t fixp_conv_layer_bias2[32],
    wt_t fixp_conv_layer_weights3[32][32][5],
    wt_t fixp_conv_layer_bias3[32],
    wt_t fixp_conv_layer_weights4[32][32][5],
    wt_t fixp_conv_layer_bias4[32],

    // declare weights and biases for dense1 and dense2
    wt_t fixp_dense1_weights[32][64], 
    wt_t fixp_dense1_bias[32],
    wt_t fixp_dense2_weights[5][32],
    wt_t fixp_dense2_bias[5],
    fm_t output_feature_map[5]
)
{
    #pragma HLS INTERFACE m_axi depth=1  port=input_feature_map   bundle=fm
    #pragma HLS INTERFACE m_axi depth=1  port=output_feature_map  bundle=fm

    #pragma HLS INTERFACE m_axi depth=1  port=fixp_conv_layer_weights1  bundle=c_weight
    #pragma HLS INTERFACE m_axi depth=1  port=fixp_conv_layer_weights2  bundle=c_weight
    #pragma HLS INTERFACE m_axi depth=1  port=fixp_conv_layer_weights3  bundle=c_weight
    #pragma HLS INTERFACE m_axi depth=1  port=fixp_conv_layer_weights4  bundle=c_weight
    #pragma HLS INTERFACE m_axi depth=1  port=fixp_conv_layer_bias1  bundle=c_bias
    #pragma HLS INTERFACE m_axi depth=1  port=fixp_conv_layer_bias2  bundle=c_bias
    #pragma HLS INTERFACE m_axi depth=1  port=fixp_conv_layer_bias3  bundle=c_bias
    #pragma HLS INTERFACE m_axi depth=1  port=fixp_conv_layer_bias4  bundle=c_bias

    #pragma HLS INTERFACE m_axi depth=1  port=fixp_dense1_weights   bundle=dense
    #pragma HLS INTERFACE m_axi depth=1  port=fixp_dense1_bias   bundle=dense
    #pragma HLS INTERFACE m_axi depth=1  port=fixp_dense2_weights   bundle=dense
    #pragma HLS INTERFACE m_axi depth=1  port=fixp_dense2_bias   bundle=dense
    
    #pragma HLS INTERFACE s_axilite register	port=return

    wt_t temp_fixp_conv_layer_weights1[32][1][5];
    wt_t temp_fixp_conv_layer_weights2[32][32][5];
    wt_t temp_fixp_conv_layer_weights3[32][32][5];
    wt_t temp_fixp_conv_layer_weights4[32][32][5];
    wt_t temp_fixp_dense1_weights[32][64];
    wt_t temp_fixp_dense2_weights[5][32];
    fm_t temp_input[1][1][187];
    fm_t temp_output_feature_map[5];

    #pragma HLS array_partition variable=temp_fixp_conv_layer_weights1 dim=3 complete
    #pragma HLS array_partition variable=temp_fixp_conv_layer_weights1 dim=2 complete
    #pragma HLS array_partition variable=temp_fixp_conv_layer_weights2 dim=3 complete
    #pragma HLS array_partition variable=temp_fixp_conv_layer_weights3 dim=3 complete
    #pragma HLS array_partition variable=temp_fixp_conv_layer_weights4 dim=3 complete

    #pragma HLS array_partition variable=temp_input dim=3 cyclic factor=5
    #pragma HLS array_partition variable=temp_input dim=2 complete
    #pragma HLS array_partition variable=temp_input dim=1 complete

    //moves values from DRAM to BRAM only for arrays that get partitioned to reduce communication overhead
    load_input:
    for(int i = 0; i < 187; i++){
        temp_input[0][0][i] = input_feature_map[0][0][i];
    }

    load_conv_weights1:
    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 5; j++) {
            temp_fixp_conv_layer_weights1[i][0][j] = fixp_conv_layer_weights1[i][0][j];
        }
    }

    load_conv_weights2:
    for(int i = 0; i < 32; i++) {
        for(int k = 0; k < 32; k++) {
            for(int j = 0; j < 5; j++) {
                temp_fixp_conv_layer_weights2[i][k][j] = fixp_conv_layer_weights2[i][k][j];
            }
        }
    }

    load_conv_weights3:
    for(int i = 0; i < 32; i++) {
        for(int k = 0; k < 32; k++) {
            for(int j = 0; j < 5; j++) {
                temp_fixp_conv_layer_weights3[i][k][j] = fixp_conv_layer_weights3[i][k][j];
            }
        }
    }

    load_conv_weights4:
    for(int i = 0; i < 32; i++) {
        for(int k = 0; k < 32; k++) {
            for(int j = 0; j < 5; j++) {
                temp_fixp_conv_layer_weights4[i][k][j] = fixp_conv_layer_weights4[i][k][j];
            }
        }
    }

    load_dense1_weights:
    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 64; j++) {
            temp_fixp_dense1_weights[i][j] = fixp_dense1_weights[i][j];
        }
    }

    load_dense2_weights:
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 32; j++) {
            temp_fixp_dense2_weights[i][j] = fixp_dense2_weights[i][j];
        }
    }

    load_output_zeros:
    for(int i = 0; i < 5; i++){
        temp_output_feature_map[i]=output_feature_map[i];
    }

 
    //begin CNN code 
    conv1:
    conv1d_1 ( //conv1 code
        fixp_conv_layer_output_feature_map1, //ignore first dimension, 32 output channels, 187 signal length
        temp_input, //ignore first dimension, 1 input channel, signal length
        temp_fixp_conv_layer_weights1, //32 out channels, kernel size:5
        fixp_conv_layer_bias1 //32 outchannels
    );

    maxpool1:
    max_pooling1(fixp_conv_layer_output_feature_map1, fixp_conv_layer_output_feature_map1_max);
    
    conv2:
    conv1d_2 ( //conv2
        fixp_conv_layer_output_feature_map2, 
        fixp_conv_layer_output_feature_map1_max, 
        temp_fixp_conv_layer_weights2, //32 out channels, kernel size:5
        fixp_conv_layer_bias2 //32 outchannels
    );

    maxpool2:
    max_pooling2(fixp_conv_layer_output_feature_map2, fixp_conv_layer_output_feature_map2_max);

    conv3:
    conv1d_3 ( //conv3
        fixp_conv_layer_output_feature_map3, 
        fixp_conv_layer_output_feature_map2_max, 
        temp_fixp_conv_layer_weights3, //32 out channels, kernel size:5
        fixp_conv_layer_bias3 //32 outchannels
    );

    maxpool3:
    max_pooling3(fixp_conv_layer_output_feature_map3, fixp_conv_layer_output_feature_map3_max);

    conv4:
    conv1d_4 ( //conv3
        fixp_conv_layer_output_feature_map4, 
        fixp_conv_layer_output_feature_map3_max, 
        temp_fixp_conv_layer_weights4, //32 out channels, kernel size:5
        fixp_conv_layer_bias4 //32 outchannels
    );

    maxpool4:
    max_pooling4(fixp_conv_layer_output_feature_map4, fixp_conv_layer_output_feature_map4_max);

    maxpool5:
    max_pooling5(fixp_conv_layer_output_feature_map4_max, fixp_conv_layer_output_feature_map5_max);

    flatten:
    flatten(fixp_conv_layer_output_feature_map5_max, fixp_conv_layer_output_feature_flat); //makes array into (1,64)

    dense1:
    dense1(fixp_conv_layer_output_feature_flat, fixp_dense1_bias, temp_fixp_dense1_weights,  fixp_dense1_output);


    dense2:
    dense2(fixp_dense1_output, fixp_dense2_bias, temp_fixp_dense2_weights, temp_output_feature_map);
    
    //Write back to DRAM
    load_output_toDRAM:
    for(int i = 0; i < 5; i++){
        output_feature_map[i] = temp_output_feature_map[i];
    }
}