/*
Layer 4:
block 0:
Conv1: input shape(256,256,40,40), output shape(256,512,20,20); stride =2
bn1: input shape(256, 512, 20, 20), output shape(256, 512, 20, 20)
relu: skip
conv2: input shape(256, 512,20, 20), output shape(256, 512, 20, 20); stride =1
bn2: input shape(256, 512, 20, 20), output_shape (256, 512, 20, 20)

Downsample:
ds_conv1: input shape(256,256,40,40), output shape(256,512,20,20); stride =2
ds_bn1: input shape(256, 512, 20, 20), output shape(256, 512, 20, 20)

output of block 0: (256, 512, 20, 20)

block 1:
conv1: input shape(256, 512,20, 20), output shape(256, 512, 20, 20); stride =1
bn1: input shape(256, 512, 20, 20), output_shape (256, 512, 20, 20)
conv2: input shape(256, 512,20, 20), output shape(256, 512, 20, 20); stride =1
bn2: input shape(256, 512, 20, 20), output_shape (256, 512, 20, 20)

Avgpool: input shape (256, 512, 20, 20), output shape(256, 512, 1, 1) or (256, 512)
//Flatten: input shape (256, 512, 1, 1), output shape (256, 512)

Linear: input shape (256, 512), output shape (256, 512)
Tanh: input shape(256, 512), output shape(256, 512)

*/

#include "top.h"
// conv for stride =1
void conv2d_vector(
    data_t input[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t weights[OUT_CHANNELS][OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    hls::vector<data_t,WIDTH/2> output[BATCH][OUT_CHANNELS][HEIGHT / 2],
    data_t gamma[OUT_CHANNELS],
    data_t beta[OUT_CHANNELS],
    data_t mean[OUT_CHANNELS],
    data_t var[OUT_CHANNELS],
    bool relu) // ReLU activation flag
{
    const data_t eps = 1e-5;
    for (int b = 0; b < BATCH; ++b)
    {
        for (int oc = 0; oc < OUT_CHANNELS; ++oc)
        {
            data_t inv_std = data_t(1.0) / std::sqrt(var[oc] + eps); //(gamma/sqrt(var[c] + eps))
            data_t b_local = beta[oc];
            data_t gamma_local = gamma[oc] * inv_std;
            data_t mean_local = mean[oc];

            for (int i = 0; i < HEIGHT / 2; ++i)
            {
                for (int j = 0; j < WIDTH / 2; ++j)
                {
                    data_t sum = 0;
                    for (int ic = 0; ic < OUT_CHANNELS; ++ic)
                    {
                        for (int kh = 0; kh < KERNEL_HEIGHT; ++kh)
                        {
                            for (int kw = 0; kw < KERNEL_WIDTH; ++kw)
                            {
                                #pragma HLS UNROLL factor=UNROLL_FACTOR
                                int in_h = i + kh - PADDING;
                                int in_w = j + kw - PADDING;
                                if (in_h >= 0 && in_h < HEIGHT / 2 && in_w >= 0 && in_w < WIDTH / 2)
                                {
                                    sum += input[b][ic][in_h][in_w] * weights[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                    sum = gamma_local * (sum - mean_local) + b_local;
                    if (relu)
                    {
                        sum = (sum > 0) ? sum : data_t(0); // ReLU activation
                    }
                    output[b][oc][i][j] = sum;
                }
            }
        }
    }
}
void ResNet(
    data_t input[BATCH][IN_CHANNELS][HEIGHT][WIDTH],
    data_t block0_weights_conv1[OUT_CHANNELS][IN_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t block0_gamma1[OUT_CHANNELS],
    data_t block0_beta1[OUT_CHANNELS],
    data_t block0_mean1[OUT_CHANNELS],
    data_t block0_var1[OUT_CHANNELS],
    data_t block0_output_bn1[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
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
    data_t output_block0[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t block1_weights_conv1[OUT_CHANNELS][OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t block1_gamma1[OUT_CHANNELS],
    data_t block1_beta1[OUT_CHANNELS],
    data_t block1_mean1[OUT_CHANNELS],
    data_t block1_var1[OUT_CHANNELS],
    data_t block1_output_bn1[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t block1_weights_conv2[OUT_CHANNELS][OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t block1_gamma2[OUT_CHANNELS],
    data_t block1_beta2[OUT_CHANNELS],
    data_t block1_mean2[OUT_CHANNELS],
    data_t block1_var2[OUT_CHANNELS],
    data_t block1_output_bn2[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t fc_weights[OUT_CHANNELS][OUT_CHANNELS],
    data_t fc_bias[OUT_CHANNELS],
    data_t tanh_output[BATCH][OUT_CHANNELS])
{
    #pragma HLS INTERFACE m_axi port=input bundle=gmem0 depth=4194304 offset=slave
#pragma HLS INTERFACE m_axi port=block0_weights_conv1 bundle=gmem1 depth=524288 offset=slave
#pragma HLS INTERFACE m_axi port=block0_output_conv1 bundle=gmem2 depth=524288 offset=slave
#pragma HLS INTERFACE m_axi port=block0_gamma1 bundle=gmem3 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block0_beta1 bundle=gmem3 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block0_mean1 bundle=gmem3 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block0_var1 bundle=gmem3 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block0_output_bn1 bundle=gmem4 depth=524288 offset=slave
#pragma HLS INTERFACE m_axi port=block0_weights_conv2 bundle=gmem5 depth=524288 offset=slave
#pragma HLS INTERFACE m_axi port=block0_output_conv2 bundle=gmem6 depth=524288 offset=slave
#pragma HLS INTERFACE m_axi port=block0_gamma2 bundle=gmem7 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block0_beta2 bundle=gmem7 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block0_mean2 bundle=gmem7 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block0_var2 bundle=gmem7 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block0_output_bn2 bundle=gmem8 depth=524288 offset=slave
#pragma HLS INTERFACE m_axi port=input_local bundle=gmem9 depth=4194304 offset=slave
#pragma HLS INTERFACE m_axi port=output_local bundle=gmem10 depth=524288 offset=slave
#pragma HLS INTERFACE m_axi port=block0_weights_ds_conv1 bundle=gmem11 depth=524288 offset=slave
#pragma HLS INTERFACE m_axi port=block0_output_ds_conv1 bundle=gmem12 depth=524288 offset=slave
#pragma HLS INTERFACE m_axi port=block0_ds_gamma1 bundle=gmem13 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block0_ds_beta1 bundle=gmem13 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block0_ds_mean1 bundle=gmem13 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block0_ds_var1 bundle=gmem13 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=output_block0 bundle=gmem14 depth=524288 offset=slave
#pragma HLS INTERFACE m_axi port=block1_weights_conv1 bundle=gmem15 depth=524288 offset=slave
#pragma HLS INTERFACE m_axi port=block1_output_conv1 bundle=gmem16 depth=524288 offset=slave
#pragma HLS INTERFACE m_axi port=block1_gamma1 bundle=gmem17 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block1_beta1 bundle=gmem17 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block1_mean1 bundle=gmem17 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block1_var1 bundle=gmem17 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block1_output_bn1 bundle=gmem18 depth=524288 offset=slave
#pragma HLS INTERFACE m_axi port=block1_weights_conv2 bundle=gmem19 depth=524288 offset=slave
#pragma HLS INTERFACE m_axi port=block1_output_conv2 bundle=gmem20 depth=524288 offset=slave
#pragma HLS INTERFACE m_axi port=block1_gamma2 bundle=gmem21 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block1_beta2 bundle=gmem21 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block1_mean2 bundle=gmem21 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block1_var2 bundle=gmem21 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=block1_output_bn2 bundle=gmem22 depth=524288 offset=slave
#pragma HLS INTERFACE m_axi port=output_avgpooling bundle=gmem23 depth=131072 offset=slave
#pragma HLS INTERFACE m_axi port=fc_weights bundle=gmem24 depth=262144 offset=slave
#pragma HLS INTERFACE m_axi port=fc_bias bundle=gmem25 depth=512 offset=slave
#pragma HLS INTERFACE m_axi port=fc_output bundle=gmem26 depth=131072 offset=slave
#pragma HLS INTERFACE m_axi port=tanh_output bundle=gmem27 depth=131072 offset=slave

#pragma HLS INTERFACE s_axilite port=return bundle=control

data_t avg_pooling_local_output[BATCH][OUT_CHANNELS];
#pragma HLS ARRAY_PARTITION variable = avg_pooling_local_output complete dim = 1

data_t local_weights_ds[OUT_CHANNELS][IN_CHANNELS];
#pragma HLS ARRAY_PARTITION variable = local_weights_ds complete dim = 1

//data_stream conv1_in_stream;


    load_weights_ds(block0_weights_ds_conv1, local_weights_ds); // Load weights for downsample conv
  //  load_input_to_stream(input, conv1_in_stream);
    conv2d_stride(input, block0_weights_conv1, block0_output_bn1, block0_gamma1, block0_beta1, block0_mean1, block0_var1, true);
    // 1b. Conv2+BatchNorm+ReLU
    conv2d_vector(block0_output_bn1, block0_weights_conv2, block0_output_bn2, block0_gamma2, block0_beta2, block0_mean2, block0_var2, true);
    // 1c. Downsample Conv1+BatchNorm
    conv2d_stride_ds(input, local_weights_ds, output_local, block0_ds_gamma1, block0_ds_beta1, block0_ds_mean1, block0_ds_var1); // ReLU activation
    Add_4D(block0_output_bn2, output_local, output_block0);
    // 2. Block 1
    // 2a. Conv1+BatchNorm+ReLU
    conv2d(output_block0, block1_weights_conv1, block1_output_bn1, block1_gamma1, block1_beta1, block1_mean1, block1_var1, true);
    // 2b. Conv2+BatchNorm+ReLU
    conv2d(block1_output_bn1, block1_weights_conv2, block1_output_bn2, block1_gamma2, block1_beta2, block1_mean2, block1_var2, true);
    // 3. Average Pooling
    average_pooling(block1_output_bn2, avg_pooling_local_output);
    // 4. Fully Connected Layer+Tanh
    fully_connected(avg_pooling_local_output, fc_weights, fc_bias, tanh_output);
}


void load_weights_per_out_channel(data_t weights[IN_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH], data_t local_weights[IN_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH])
{
    for (int ic = 0; ic < IN_CHANNELS; ++ic)
    {
        for (int kh = 0; kh < KERNEL_HEIGHT; ++kh)
        {
            for (int kw = 0; kw < KERNEL_WIDTH; ++kw)
            {
#pragma HLS pipeline II = 1
                local_weights[ic][kh][kw] = weights[ic][kh][kw];
            }  
        }
    }
}



void load_weights_ds(data_t weights[OUT_CHANNELS][IN_CHANNELS][DS_KERNEL_HEIGHT][DS_KERNEL_WIDTH], data_t local_weights[OUT_CHANNELS][IN_CHANNELS])
{
    for (int oc = 0; oc < OUT_CHANNELS; ++oc)
    {
        for (int ic = 0; ic < IN_CHANNELS; ++ic)
        {
#pragma HLS pipeline II = 1
            local_weights[oc][ic] = weights[oc][ic][0][0];
        }
    }
}
// Simple Addition of 4D tensors
//  4D tensor: input[BATCH][IN_CHANNELS][HEIGHT][WIDTH]
void Add_4D(
    hls::vector<data_t,WIDTH / 2> input1[BATCH][OUT_CHANNELS][HEIGHT / 2],
    hls::vector<data_t,WIDTH / 2> input2[BATCH][OUT_CHANNELS][HEIGHT / 2],
    data_t output[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2])
{
    //Cam use input1 and input2 as vectors
    hls::vector<data_t, WIDTH / 2> local_y_vector;
    for (int b = 0; b < BATCH; ++b)
    {
        for (int ic = 0; ic < OUT_CHANNELS; ++ic)
        {
            for (int i = 0; i < HEIGHT / 2; ++i)
            {
#pragma HLS pipeline II = 1
                local_y_vector= input1[b][ic][i]+ input2[b][ic][i];
                for (int j = 0; j < WIDTH / 2; ++j)
                {
                    output[b][ic][i][j] = (local_y_vector[j] > 0) ?local_y_vector[j] : data_t(0); // ReLU activation
                }
            }
        }
    }
}

void load_4D_Inputs_to_2D(data_t input[HEIGHT][WIDTH],data_t input_local[HEIGHT][WIDTH])
{
    for (int i = 0; i < HEIGHT; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
#pragma HLS pipeline II = 1
            input_local[i][j] = input[i][j];
        }
    }
}

void conv2d_stride(
    data_t input[BATCH][IN_CHANNELS][HEIGHT][WIDTH],
    data_t weights[OUT_CHANNELS][IN_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t output[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t gamma[OUT_CHANNELS],
    data_t beta[OUT_CHANNELS],
    data_t mean[OUT_CHANNELS],
    data_t var[OUT_CHANNELS],
    bool relu) 
{
    const data_t eps = 1e-5;
    for (int b = 0; b < BATCH; ++b)
    {
        for (int oc = 0; oc < OUT_CHANNELS; ++oc)
        {
            data_t inv_std = data_t(1.0) / std::sqrt(var[oc] + eps); 
            data_t b_local = beta[oc];
            data_t gamma_local = gamma[oc] * inv_std;
            data_t mean_local = mean[oc];
           
            data_t local_weights[IN_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH];
            #pragma HLS ARRAY_PARTITION variable=local_weights complete
            load_weights_per_out_channel(weights[oc], local_weights);

            for (int i = 0; i < HEIGHT; i += 2)
            {
                for (int j = 0; j < WIDTH; j += 2)
                {
                    data_t sum = 0;
                    for (int ic = 0; ic < IN_CHANNELS; ++ic)
                    { 
                        for (int kh = 0; kh < KERNEL_HEIGHT; ++kh)
                        {
                            for (int kw = 0; kw < KERNEL_WIDTH; ++kw)
                            {   
                                
                                int in_h = i + kh - PADDING;
                                int in_w = j + kw - PADDING;
                                if (in_h >= 0 && in_h < HEIGHT && in_w >= 0 && in_w < WIDTH)
                                {   
                                    sum += input[b][ic][in_h][in_w] * local_weights[ic][kh][kw];
                                }
                            }
                        }
                    }
                    sum = gamma_local * (sum - mean_local) + b_local;
                    if (relu)
                    {
                        sum = (sum > 0) ? sum : data_t(0); 
                    }
                    output[b][oc][i / 2][j / 2] = sum;
                }
            }
        }
    }
}

void conv2d_stride_ds(
    data_t input[BATCH][IN_CHANNELS][HEIGHT][WIDTH],
    data_t weights[OUT_CHANNELS][IN_CHANNELS],
    hls::vector<data_t, WIDTH / 2> output[BATCH][OUT_CHANNELS][HEIGHT / 2],
    data_t gamma[OUT_CHANNELS],
    data_t beta[OUT_CHANNELS],
    data_t mean[OUT_CHANNELS],
    data_t var[OUT_CHANNELS]) 
{
    const data_t eps = 1e-5;
    for (int b = 0; b < BATCH; ++b)
    {
        for (int oc = 0; oc < OUT_CHANNELS; ++oc)
        {
            data_t inv_std = data_t(1.0) / std::sqrt(var[oc] + eps);
            data_t b_local = beta[oc];
            data_t gamma_local = gamma[oc] * inv_std;
            data_t mean_local = mean[oc];

            for (int i = 0; i < HEIGHT; i += 2)
            { 
                for (int j = 0; j < WIDTH; j += 2)
                {
                    data_t sum = 0;

                    for (int ic = 0; ic < IN_CHANNELS; ++ic)
                    {
                        sum += input[b][ic][i][j] * weights[oc][ic];
                    }
                    sum = gamma_local * (sum - mean_local) + b_local;
                    output[b][oc][i / 2][j / 2] = sum;
                }
            }
        }
    }
}


void conv2d(
    data_t input[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t weights[OUT_CHANNELS][OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t output[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t gamma[OUT_CHANNELS],
    data_t beta[OUT_CHANNELS],
    data_t mean[OUT_CHANNELS],
    data_t var[OUT_CHANNELS],
    bool relu) 
{
    const data_t eps = 1e-5;
    for (int b = 0; b < BATCH; ++b)
    {
        for (int oc = 0; oc < OUT_CHANNELS; ++oc)
        {
            data_t inv_std = data_t(1.0) / std::sqrt(var[oc] + eps); 
            data_t b_local = beta[oc];
            data_t gamma_local = gamma[oc] * inv_std;
            data_t mean_local = mean[oc];

            for (int i = 0; i < HEIGHT / 2; ++i)
            {
                for (int j = 0; j < WIDTH / 2; ++j)
                {
                    data_t sum = 0;
                    for (int ic = 0; ic < OUT_CHANNELS; ++ic)
                    {
                        for (int kh = 0; kh < KERNEL_HEIGHT; ++kh)
                        {
                            for (int kw = 0; kw < KERNEL_WIDTH; ++kw)
                            {
                                #pragma HLS UNROLL factor=UNROLL_FACTOR
                                int in_h = i + kh - PADDING;
                                int in_w = j + kw - PADDING;
                                if (in_h >= 0 && in_h < HEIGHT / 2 && in_w >= 0 && in_w < WIDTH / 2)
                                {
                                    sum += input[b][ic][in_h][in_w] * weights[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                    sum = gamma_local * (sum - mean_local) + b_local;
                    if (relu)
                    {
                        sum = (sum > 0) ? sum : data_t(0); 
                    }
                    output[b][oc][i][j] = sum;
                }
            }
        }
    }
}




void average_pooling(
    data_t input[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t output[BATCH][OUT_CHANNELS])
{ 
    for (int b = 0; b < BATCH; ++b)
    {
        for (int c = 0; c < OUT_CHANNELS; ++c)
        {
            data_t sum = 0;
            for (int h = 0; h < HEIGHT / 2; ++h)
            {
            #pragma HLS pipeline II = 1
                for (int w = 0; w < WIDTH / 2; ++w)
                {
                    sum += (input[b][c][h][w]);
                }
            }
            output[b][c] = (sum / ((HEIGHT / 2) * (WIDTH / 2))); 
        }
    }
}
void fully_connected(
    data_t input[BATCH][OUT_CHANNELS],
    data_t weights[OUT_CHANNELS][OUT_CHANNELS],
    data_t bias[OUT_CHANNELS],
    data_t output[BATCH][OUT_CHANNELS])
{
    
	#pragma HLS BIND_STORAGE variable=weights type=RAM_1P impl=URAM
	
	data_t input_local[BATCH][OUT_CHANNELS];
	
	#pragma HLS ARRAY_PARTITION variable=input_local cyclic factor=16 dim=2
	

	for (int b = 0; b < BATCH; b++) {
        for (int k = 0; k < OUT_CHANNELS; k++) {
            #pragma HLS UNROLL factor=UNROLL_FACTOR
            input_local[b][k] = input[b][k];
        }
    }


    for (int b = 0; b < BATCH; b++)
    {

        for (int j = 0; j < OUT_CHANNELS; j++)
        {
            data_t sum=0;
            for (int k = 0; k < OUT_CHANNELS; k++)
            {   
                #pragma HLS UNROLL factor=UNROLL_FACTOR
                sum+= input_local[b][k] * weights[j][k]; 
            }
            sum += bias[j]; 
                            
            data_t sum_sq = sum * sum;
            data_t numerator = sum * (27 + sum_sq);
            data_t denominator = 27 + 9 * sum_sq;
            output[b][j] = numerator / denominator;
        }
    }
}


