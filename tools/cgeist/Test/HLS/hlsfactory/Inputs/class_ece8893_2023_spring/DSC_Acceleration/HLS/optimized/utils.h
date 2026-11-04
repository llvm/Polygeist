///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    utils.h
// Description: Header file for utility functions.
//
// You are free to modify the existing functions or add new ones.
// https://stackoverflow.com/questions/10632251/undefined-reference-to-template-function
// Reason for moving implementation to utils.h instead of cpp
///////////////////////////////////////////////////////////////////////////////
#include "conv.h"

//--------------------------------------------------------------------------
// Function Declarations
//--------------------------------------------------------------------------
//TODO optimize later to handle last loop case where buf sizze == actual left to compute
void conv_3x3 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t B_buf[OUT_BUF_DEPTH],
    int b,
    int d,
    int stride, 
    int inp_channel, 
    int out_channel
)
{
    //Handle output buf not matching
    //Handle depths not matching

    int f_m = (out_channel > OUT_BUF_DEPTH) ? OUT_BUF_DEPTH : out_channel;
    int c_m = (inp_channel > IN_BUF_DEPTH)  ? IN_BUF_DEPTH  : inp_channel;

    if (b)
    {
        if ((b + 1) * OUT_BUF_DEPTH > out_channel) f_m =  (out_channel - b * OUT_BUF_DEPTH);
    }
    if (d)
    {
        if ((d + 1) * IN_BUF_DEPTH > inp_channel) c_m =  (inp_channel - d * IN_BUF_DEPTH);
    }
    int OUT_BUF_HEIGHT_T =  OUT_BUF_HEIGHT/stride;
    int OUT_BUF_WIDTH_T = OUT_BUF_WIDTH/stride;

    if (d==0)
    {
        INITIALIZE_OUTPUT_CONV2D:
        for (int f = 0; f < f_m; f++)
        {
            for (int i = 0; i < OUT_BUF_HEIGHT_T; i++)
            {
                for (int j = 0; j < OUT_BUF_WIDTH_T; j++)
                {
                    Y_buf[f][i][j] = B_buf[f];
                }
            }
        }
    }

    KERNEL_FILTER_HEIGHT:
    for (int kh = 0; kh < KERNEL_HEIGHT; kh++)
    {
        KERNEL_FILTER_WIDTH:
        for (int kw = 0; kw < KERNEL_WIDTH; kw++)       
        {
            INPUT_IMAGE_DEPTH:
            for (int c = 0; c < c_m; c++)
            {
                FILTER_OUTPUT_DEPTH:
                for (int f = 0; f < f_m; f++)
                {
                    INPUT_IMAGE_HEIGHT:
                    for (int i = 0; i < OUT_BUF_HEIGHT/stride; i++)
                    {
                        #pragma HLS pipeline
                        INPUT_IMAGE_WIDTH:
                        for (int j = 0; j < OUT_BUF_WIDTH/stride; j++)
                        {
                            int ki = (i * stride) + kh;
                            int kj = (j * stride) + kw;
							Y_buf[f][i][j] += X_buf[c][ki][kj] * W_buf[f][c][kh][kw];
                        }
                    }
                }
            }
        }
    }
}

void dconv_3x3 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t B_buf[OUT_BUF_DEPTH],
    int d,
    int stride, int inp_channel
)
{
    //Handle output buf not matching
    //Handle depths not matching
    int c_m = (inp_channel > IN_BUF_DEPTH)  ? IN_BUF_DEPTH  : inp_channel;

    INITIALIZE_OUTPUT_DEPTHWISE_CONV2D:
    for (int c = 0; c < IN_BUF_DEPTH; c++)
    {
        for (int i = 0; i < OUT_BUF_HEIGHT/stride; i++)
        {
            for (int j = 0; j < OUT_BUF_WIDTH/stride; j++)
            {
                Y_buf[c][i][j] = B_buf[c];
            }
        }
    }

    if (d)
    {
        if ((d + 1) * OUT_BUF_DEPTH > inp_channel) c_m =  (inp_channel - d * OUT_BUF_DEPTH);
    }

    int OUT_BUF_HEIGHT_T =  OUT_BUF_HEIGHT/stride;
    int OUT_BUF_WIDTH_T = OUT_BUF_WIDTH/stride;

    INPUT_IMAGE_DEPTH:
    for (int c = 0; c < c_m; c++)
    {
        KERNEL_FILTER_HEIGHT:
        for (int kh = 0; kh < KERNEL_HEIGHT; kh++)
        {
            KERNEL_FILTER_WIDTH:
            for (int kw = 0; kw < KERNEL_WIDTH; kw++)       
            {
                INPUT_IMAGE_HEIGHT:
                for (int i = 0; i < OUT_BUF_HEIGHT_T; i++)
                {
                    #pragma HLS pipeline
                    INPUT_IMAGE_WIDTH:
                    for (int j = 0; j < OUT_BUF_WIDTH_T; j++)
                    {
                        int ki = (i * stride) + kh;
                        int kj = (j * stride) + kw;
                        Y_buf[c][i][j] += X_buf[c][ki][kj] * W_buf[c][0][kh][kw];
                        // if ((c==2 or c==18) and i==8 and j==8 and print_out)
                        //     std::cout << "conv_out_buf[1][8][8]" << Y_buf[c][i][j] << " " <<  X_buf[c][ki][kj] << " * " <<  W_buf[c][0][kh][kw] <<std::endl;
                    }
                }
            }
        }
    }
}

void d2conv_3x3 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t B_buf[OUT_BUF_DEPTH],
    int d, 
    int stride, 
    int inp_channel
)
{
    //Handle output buf not matching
    //Handle depths not matching
    int c_m = (inp_channel > IN_BUF_DEPTH)  ? IN_BUF_DEPTH  : inp_channel;

    INITIALIZE_OUTPUT_DEPTHWISE_CONV2D:    
    for (int c = 0; c < c_m; c++)
    {
        for (int i = 0; i < OUT_BUF_HEIGHT/stride; i++)
        {
            for (int j = 0; j < OUT_BUF_WIDTH/stride; j++)
            {
                Y_buf[c][i][j] = B_buf[c];
            }
        }
    }

    if (d)
    {
        if ((d + 1) * OUT_BUF_DEPTH > inp_channel) c_m =  (inp_channel - d * OUT_BUF_DEPTH);
    }

    int OUT_BUF_HEIGHT_T =  OUT_BUF_HEIGHT/stride;
    int OUT_BUF_WIDTH_T = OUT_BUF_WIDTH/stride;

    INPUT_IMAGE_DEPTH:
    for (int c = 0; c < c_m; c++)
    {
        KERNEL_FILTER_HEIGHT:
        for (int kh = 0; kh < KERNEL_HEIGHT; kh++)
        {
            KERNEL_FILTER_WIDTH:
            for (int kw = 0; kw < KERNEL_WIDTH; kw++)       
            {
                INPUT_IMAGE_HEIGHT:
                for (int i = 0; i < OUT_BUF_HEIGHT_T; i++)
                {
                    #pragma HLS pipeline
                    INPUT_IMAGE_WIDTH:
                    for (int j = 0; j < OUT_BUF_WIDTH_T; j++)
                    {
                        int ki = (i * stride) + kh;
                        int kj = (j * stride) + kw;
                        Y_buf[c][i][j] += X_buf[c][ki][kj] * W_buf[0][c][kh][kw];
                        // if ((c==2 or c==18) and i==8 and j==8 and print_out)
                        //     std::cout << "conv_out_buf[1][8][8]" << Y_buf[c][i][j] << " " <<  X_buf[c][ki][kj] << " * " <<  W_buf[c][0][kh][kw] <<std::endl;
                    }
                }
            }
        }
    }
}

// template <int out_channel>
void pointwise_conv_3x3 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t B_buf[OUT_BUF_DEPTH],
    int b,
    int d,
    int inp_channel, 
    int out_channel
)
{
    //Handle output buf not matching
    //Handle depths not matching

    if (d==0)
    {
        INITIALIZE_OUTPUT_POINTWISE_CONV2D:
        for (int f = 0; f < OUT_BUF_DEPTH; f++)
        {
            for (int i = 0; i < OUT_BUF_HEIGHT; i++)
            {
                for (int j = 0; j < OUT_BUF_WIDTH; j++)
                {
                    Y_buf[f][i][j] = B_buf[f];
                }
            }
        }
    }

    int f_m;
    int c_m;
    if (b)
    {
        if ((b + 1) * OUT_BUF_DEPTH > out_channel) f_m =  (out_channel - b * OUT_BUF_DEPTH);
    }
    if (d)
    {
        if ((d + 1) * IN_BUF_DEPTH > inp_channel) c_m =  (inp_channel - d * IN_BUF_DEPTH);
    }

    f_m = (out_channel > OUT_BUF_DEPTH) ? OUT_BUF_DEPTH : out_channel;
    c_m = (inp_channel > IN_BUF_DEPTH)  ? IN_BUF_DEPTH  : inp_channel;

    FILTER_OUTPUT_DEPTH:
    for (int f = 0; f < f_m; f+=1)
    {
        INPUT_IMAGE_DEPTH:
        for (int c = 0; c < c_m; c+=1)
        {
            INPUT_IMAGE_HEIGHT:
            for (int i = 0; i < OUT_BUF_HEIGHT; i++)
            {
                #pragma HLS pipeline 
                INPUT_IMAGE_WIDTH:
                for (int j = 0; j < OUT_BUF_WIDTH; j++)
                {
                    Y_buf[f][i][j] += X_buf[c][i][j] * W_buf[f][c][0][0];
                }
            }
        }
    }
}

//--------------------------------------------------------------------------
// Function to perform reLU6 on output BRAM.
//--------------------------------------------------------------------------
// template <>
void reLU6_1 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH]
)
{
    RELU_F:
    for (int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        RELU_H:
        for (int i = 0; i < OUT_BUF_HEIGHT; i++)
        {
            #pragma HLS pipeline
            RELU_W:
            for (int j = 0; j < OUT_BUF_WIDTH; j++)
            {
                if(Y_buf[f][i][j] < (fm_t) 0)
                {
                    Y_buf[f][i][j] = (fm_t) 0;
                }
                else if(Y_buf[f][i][j] > (fm_t) 6)
                {
                    Y_buf[f][i][j] = (fm_t) 6;
                }
            }
        }
    } 
}

void reLU6_2 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH]
)
{
    int OUT_BUF_HEIGHT_T =  OUT_BUF_HEIGHT/2;
    int OUT_BUF_WIDTH_T = OUT_BUF_WIDTH/2;

    RELU_F:
    for (int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        RELU_H:
        for (int i = 0; i < OUT_BUF_HEIGHT_T; i++)
        {
            #pragma HLS pipeline
            RELU_W:
            for (int j = 0; j < OUT_BUF_WIDTH_T; j++)
            {
                if(Y_buf[f][i][j] < (fm_t) 0)
                {
                    Y_buf[f][i][j] = (fm_t) 0;
                }
                else if(Y_buf[f][i][j] > (fm_t) 6)
                {
                    Y_buf[f][i][j] = (fm_t) 6;
                }
            }
        }
    } 
}

//--------------------------------------------------------------------------
// Function to load input feature map tile block from DRAM to on-chip BRAM.
//--------------------------------------------------------------------------
template <int inp_channel, int inp_height, int inp_width>
void load_input_tile_block_from_DRAM (
    fm_t in_fm_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH], 
    fm_t in_fm[inp_channel][inp_height][inp_width], 
    int  ti, 
    int  tj, 
    int  d,
    int padding
)
{
    #pragma HLS inline off

    int c_m = (inp_channel > IN_BUF_DEPTH)  ? IN_BUF_DEPTH  : inp_channel;

    if (d)
    {
        if ((d + 1) * IN_BUF_DEPTH > inp_channel) c_m =  (inp_channel - d * IN_BUF_DEPTH);
    }

    const int depth_offset  =  d * IN_BUF_DEPTH;
    const int height_offset = ti * OUT_BUF_HEIGHT; // OUT_BUF is intended, not a typo. 
    const int width_offset  = tj * OUT_BUF_WIDTH;
        
    INPUT_BUFFER_DEPTH:
    for(int c = 0; c < c_m; c++)
    {
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < IN_BUF_HEIGHT; i++)
        {
            INPUT_BUFFER_WIDTH:
            for(int j = 0; j < IN_BUF_WIDTH; j++)
            {
                // TODO: Handle border features here
                if ( height_offset+i-padding >= 0 && height_offset+i-padding < inp_height    &&    width_offset + j -padding >=0 && width_offset + j -padding < inp_width )
                {
                    // if ((c==0 or c==16) and i==8 and j==8 and print_out)
                    //     std::cout <<"c: " << depth_offset + c <<  " " << "h: " << height_offset + i - padding << "w: " <<  width_offset + j - padding<< " "<< in_fm[depth_offset + c][height_offset + i - padding][width_offset + j - padding]<< std::endl;
                    in_fm_buf[c][i][j] = in_fm[depth_offset + c][height_offset + i - padding][width_offset + j - padding];
                }
                else
                    in_fm_buf[c][i][j]  = 0;
            }
        }
    }
}

//--------------------------------------------------------------------------
// Function to load weights and bias from DRAM to on-chip BRAM.
//--------------------------------------------------------------------------
template <int out_channel, int inp_channel, int kernel_height, int kernel_width>
void load_conv_layer_params_from_DRAM(
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t bias_buf[OUT_BUF_DEPTH],
    wt_t weights[out_channel][inp_channel][kernel_height][kernel_width],
    wt_t bias[out_channel],
    int b,
    int d
)
{
    #pragma HLS inline off

    const int kernel_offset  = b * OUT_BUF_DEPTH;
    const int channel_offset = d * IN_BUF_DEPTH;

    int f_m = (out_channel > OUT_BUF_DEPTH) ? OUT_BUF_DEPTH : out_channel;
    int c_m = (inp_channel > IN_BUF_DEPTH)  ? IN_BUF_DEPTH  : inp_channel;

    if (b)
    {
        if ((b + 1) * OUT_BUF_DEPTH > out_channel) f_m =  (out_channel - b * OUT_BUF_DEPTH);
    }
    if (d)
    {
        if ((d + 1) * IN_BUF_DEPTH > inp_channel) c_m =  (inp_channel - d * IN_BUF_DEPTH);
    }

    WEIGHT_KERNEL_NUM:
    for(int f = 0; f < f_m; f++)
    {
        WEIGHT_KERNEL_DEPTH:
        for(int c = 0; c < c_m; c++)
        {
            WEIGHT_KERNEL_HEIGHT:
            for(int kh = 0; kh < kernel_height; kh++)
	        {
                WEIGHT_KERNEL_WIDTH:
	            for(int kw = 0; kw < kernel_width; kw++)
	            {
	                weight_buf[f][c][kh][kw] = weights[kernel_offset + f][channel_offset + c][kh][kw];
                }
            }
        }
    }
    
    BIAS:
    for(int f = 0; f < f_m; f++)
    {
        bias_buf[f] = bias[(kernel_offset + channel_offset) + f]; //TODO :: BIG ASSUMPTION that either b or d will be 0 else this wont work
    }
}

//--------------------------------------------------------------------------
// Function to store output feature map tile block from BRAM to DRAM.
//--------------------------------------------------------------------------
template <int out_channel, int inp_height, int inp_width>
void store_output_tile_to_DRAM (
    fm_t out_fm[out_channel][inp_height][inp_width], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    int  ti,
    int  tj,
    int  b,
    int stride
)
{
    #pragma HLS inline
    const int OUT_BUF_HEIGHT_T = OUT_BUF_HEIGHT/stride;
    const int OUT_BUF_WIDTH_T  = OUT_BUF_WIDTH/stride;

    const int depth_offset  =  b * OUT_BUF_DEPTH;
    const int height_offset = ti * OUT_BUF_HEIGHT_T;
    const int width_offset  = tj * OUT_BUF_WIDTH_T;

    int h_m = (inp_height > OUT_BUF_HEIGHT_T)  ? OUT_BUF_HEIGHT_T : inp_height;
    int w_m = (inp_width > OUT_BUF_WIDTH_T)    ? OUT_BUF_WIDTH_T  : inp_width;
    int d_m = (out_channel > OUT_BUF_DEPTH)    ? OUT_BUF_DEPTH    : out_channel;

    if (ti)
    {
        if ((ti + 1) * OUT_BUF_HEIGHT_T > inp_height) h_m =  (inp_height - ti * OUT_BUF_HEIGHT_T);
    }
    if (tj)
    {
        if ((tj + 1) * OUT_BUF_WIDTH_T > inp_width) w_m =  (inp_width - tj * OUT_BUF_WIDTH_T);
    }
    if (b)
    {
        if ((b + 1) * OUT_BUF_DEPTH > out_channel) d_m =  (out_channel - b * OUT_BUF_DEPTH);
    }

    OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < d_m; f++)
    {
        OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < h_m; i++)
        {
            OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < w_m; j++)
            {
                out_fm[depth_offset + f][height_offset + i][width_offset + j] = out_fm_buf[f][i][j];
            }
        }
    }
}
