#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <ap_float.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_vector.h>
#include <hls_stream.h>
#include <cmath>

#define BATCH 2//original 256, for testing 2
#define IN_CHANNELS 256
#define OUT_CHANNELS 512
#define HEIGHT 40
#define WIDTH 40
#define KERNEL_HEIGHT 3
#define KERNEL_WIDTH 3
#define PADDING 1
#define STRIDE 2
#define DS_KERNEL_HEIGHT 1
#define DS_KERNEL_WIDTH 1
#define UNROLL_FACTOR 8



typedef float data_t; //Don't change this, it is being used to read the input data.
typedef ap_fixed<16, 8> data_t_fixed; //Fixed point data type for internal calculations.
//typedef ap_fixed<32, 16> data_t; //Fixed point data type for internal calculations.
///TODO: Need to create a new data type to be used in the kernel function. Also might need to change all the 
//data types to fixed point.
//FIFO Definations

void ResNet(	
	data_t input[BATCH][IN_CHANNELS][HEIGHT][WIDTH],
    data_t block0_weights_conv1[OUT_CHANNELS][IN_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t block0_gamma1[OUT_CHANNELS],
    data_t block0_beta1[OUT_CHANNELS],
    data_t block0_mean1[OUT_CHANNELS],
    data_t block0_var1[OUT_CHANNELS],
	data_t block0_output_bn1[BATCH][OUT_CHANNELS][HEIGHT/2][WIDTH/2],
	data_t block0_weights_conv2[OUT_CHANNELS][OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
	data_t block0_gamma2[OUT_CHANNELS],
    data_t block0_beta2[OUT_CHANNELS],
    data_t block0_mean2[OUT_CHANNELS],
    data_t block0_var2[OUT_CHANNELS],
    hls::vector<data_t,WIDTH/2> block0_output_bn2[BATCH][OUT_CHANNELS][HEIGHT / 2],
    hls::vector<data_t,WIDTH/2> output_local[BATCH][OUT_CHANNELS][HEIGHT / 2],
	data_t block0_weights_ds_conv1[OUT_CHANNELS][IN_CHANNELS][DS_KERNEL_HEIGHT][DS_KERNEL_WIDTH],
    data_t block0_ds_gamma1[OUT_CHANNELS],
    data_t block0_ds_beta1[OUT_CHANNELS],
    data_t block0_ds_mean1[OUT_CHANNELS],
    data_t block0_ds_var1[OUT_CHANNELS],
	data_t output_block0[BATCH][OUT_CHANNELS][HEIGHT/2][WIDTH/2],
	data_t block1_weights_conv1[OUT_CHANNELS][OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t block1_gamma1[OUT_CHANNELS],
    data_t block1_beta1[OUT_CHANNELS],
    data_t block1_mean1[OUT_CHANNELS],
    data_t block1_var1[OUT_CHANNELS],
	data_t block1_output_bn1[BATCH][OUT_CHANNELS][HEIGHT/2][WIDTH/2],
	data_t block1_weights_conv2[OUT_CHANNELS][OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
	data_t block1_gamma2[OUT_CHANNELS],
    data_t block1_beta2[OUT_CHANNELS],
    data_t block1_mean2[OUT_CHANNELS],
    data_t block1_var2[OUT_CHANNELS],
	data_t block1_output_bn2[BATCH][OUT_CHANNELS][HEIGHT/2][WIDTH/2],
    data_t fc_weights[OUT_CHANNELS][OUT_CHANNELS],
    data_t fc_bias[OUT_CHANNELS],
	data_t tanh_output[BATCH][OUT_CHANNELS]
);

void conv2d_stride(
    data_t input[BATCH][IN_CHANNELS][HEIGHT][WIDTH],
    data_t weights[OUT_CHANNELS][IN_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH], // 🔥 Correct
    data_t output[BATCH][OUT_CHANNELS][HEIGHT/2][WIDTH/2],
    data_t gamma[OUT_CHANNELS],
    data_t beta[OUT_CHANNELS],
    data_t mean[OUT_CHANNELS],
    data_t var[OUT_CHANNELS],
    bool relu = true // ReLU activation flag
);

void load_weights_ds(data_t weights[OUT_CHANNELS][IN_CHANNELS][DS_KERNEL_HEIGHT][DS_KERNEL_WIDTH],
    data_t local_weights[OUT_CHANNELS][IN_CHANNELS]);
    
void conv2d_stride_ds(
    data_t input[BATCH][IN_CHANNELS][HEIGHT][WIDTH],
    data_t weights[OUT_CHANNELS][IN_CHANNELS], 
    hls::vector<data_t,WIDTH/2> output[BATCH][OUT_CHANNELS][HEIGHT/2],
    data_t gamma[OUT_CHANNELS],
    data_t beta[OUT_CHANNELS],
    data_t mean[OUT_CHANNELS],
    data_t var[OUT_CHANNELS]);
	
void conv2d(
    data_t input[BATCH][OUT_CHANNELS][HEIGHT/2][WIDTH/2],
    data_t weights[OUT_CHANNELS][OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t output[BATCH][OUT_CHANNELS][HEIGHT/2][WIDTH/2],
    data_t gamma[OUT_CHANNELS],
    data_t beta[OUT_CHANNELS],
    data_t mean[OUT_CHANNELS],
    data_t var[OUT_CHANNELS],
    bool relu = true // ReLU activation flag);
);

void average_pooling(
    data_t input[BATCH][OUT_CHANNELS][HEIGHT/2][WIDTH/2],
    data_t output[BATCH][OUT_CHANNELS]);
	
void fully_connected(
    data_t input[BATCH][OUT_CHANNELS],
    data_t weights[OUT_CHANNELS][OUT_CHANNELS],
    data_t bias[OUT_CHANNELS],
    data_t output[BATCH][OUT_CHANNELS]
);


void Add_4D(
    hls::vector<data_t,WIDTH/2> input1[BATCH][OUT_CHANNELS][HEIGHT/2],
    hls::vector<data_t,WIDTH/2> input2[BATCH][OUT_CHANNELS][HEIGHT/2],
    data_t output[BATCH][OUT_CHANNELS][HEIGHT/2][WIDTH/2]);

//void load_input_to_stream(data_t input[BATCH][IN_CHANNELS][HEIGHT][WIDTH], data_stream &conv1_in_stream);

void load_weights_per_out_channel_OUT(data_t weights[OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH], data_t local_weights[OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH]);