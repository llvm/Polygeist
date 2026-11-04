#include "gradient.h"

//--------------------------------------------------------------------------
// Function to load input feature map tile block from DRAM to on-chip BRAM.
//--------------------------------------------------------------------------
template <int inp_channel, int inp_height, int inp_width>
void load_input_tile_block_from_DRAM (
    fm_t in_fm_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH], 
    fm_t in_fm[inp_channel][inp_height][inp_width], 
    int  ti, 
    int  tj, 
    int  d
)
{
    #pragma HLS inline off
    
    int IN_BUF_DEPTH_MUX;
    if (inp_channel < IN_BUF_DEPTH)
        IN_BUF_DEPTH_MUX = inp_channel;
    else
        IN_BUF_DEPTH_MUX = IN_BUF_DEPTH;

    const int depth_offset  =  d * IN_BUF_DEPTH;
    const int height_offset = ti * OUT_BUF_HEIGHT; // OUT_BUF is intended, not a typo. 
    const int width_offset  = tj * OUT_BUF_WIDTH;
        
    INPUT_BUFFER_DEPTH:
    for(int c = 0; c < IN_BUF_DEPTH_MUX; c++)
    {
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < IN_BUF_HEIGHT; i++)
        {
            INPUT_BUFFER_WIDTH:
            for(int j = 0; j < IN_BUF_WIDTH; j++)
            {
                // TODO: Handle border features here
		        if ( height_offset+i-1 >= 0 && height_offset+i-1 < inp_height    &&    width_offset + j -1 >=0 && width_offset + j -1 < inp_width )
                {
                    in_fm_buf[c][i][j] = in_fm[depth_offset + c][height_offset + i - 1][width_offset + j - 1];
                }
                else
                    in_fm_buf[c][i][j]  = 0;
            }
        }
    }
}

//--------------------------------------------------------------------------
// Function to load layer parameters (weights and bias) for convolution.
//--------------------------------------------------------------------------
template <int out_channel, int inp_channel>
void load_conv_layer_params_from_DRAM(
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][3][3],
    wt_t bias_buf[OUT_BUF_DEPTH],
    wt_t weights[out_channel][inp_channel][3][3],
    wt_t bias[out_channel],
    int b,
    int d
)
{
    #pragma HLS inline off

    int IN_BUF_DEPTH_MUX;
    if (inp_channel < IN_BUF_DEPTH)
        IN_BUF_DEPTH_MUX = inp_channel;
    else
        IN_BUF_DEPTH_MUX = IN_BUF_DEPTH;

    const int kernel_offset  = b * OUT_BUF_DEPTH;
    const int channel_offset = d * IN_BUF_DEPTH;

    WEIGHT_KERNEL_NUM:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        WEIGHT_KERNEL_DEPTH:
        for(int c = 0; c < IN_BUF_DEPTH_MUX; c++)
        {
            WEIGHT_KERNEL_HEIGHT:
            for(int kh = 0; kh < 3; kh++)
	        {
                WEIGHT_KERNEL_WIDTH:
	            for(int kw = 0; kw < 3; kw++)
	            {
	                weight_buf[f][c][kh][kw] = weights[kernel_offset + f][channel_offset + c][kh][kw];
                }
            }
        }
    }
    
    BIAS:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        bias_buf[f] = bias[kernel_offset + f];
    }
}

template <int out_channel, int inp_channel>
void load_transpose_flipped_conv_layer_params_from_DRAM(
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][3][3],
    wt_t bias_buf[OUT_BUF_DEPTH],
    wt_t weights[inp_channel][out_channel][3][3],
    int b,
    int d
)
{
    #pragma HLS inline off

    int OUT_BUF_DEPTH_MUX;
    if (out_channel < IN_BUF_DEPTH)
        OUT_BUF_DEPTH_MUX = out_channel;
    else
        OUT_BUF_DEPTH_MUX = OUT_BUF_DEPTH;

    const int kernel_offset  = b * OUT_BUF_DEPTH;
    const int channel_offset = d * IN_BUF_DEPTH;

    WEIGHT_KERNEL_NUM:
    for(int f = 0; f < OUT_BUF_DEPTH_MUX; f++)
    {
        WEIGHT_KERNEL_DEPTH:
        for(int c = 0; c < IN_BUF_DEPTH; c++)
        {
            WEIGHT_KERNEL_HEIGHT:
            for(int kh = 0; kh < 3; kh++)
	        {
                WEIGHT_KERNEL_WIDTH:
	            for(int kw = 0; kw < 3; kw++)
	            {
	                weight_buf[f][c][kh][kw] = weights[channel_offset + c][kernel_offset + f][2-kh][2-kw];
                }
            }
        }
    }

    BIAS:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        bias_buf[f] = 0;
    }
    
}



//---------------------------------------------------------------------------
// Perform synthesizable tiling-based convolution for a single tile.
//---------------------------------------------------------------------------

void conv_3x3 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][3][3],
    wt_t bias_buf[OUT_BUF_DEPTH],
    int  d,
    int inp_channel,
    int out_channel
)
{
    if(d==0){ 
        init:
        for(int oh=0; oh<OUT_BUF_HEIGHT; oh++){
            // #pragma HLS pipeline
                
            for(int ow=0; ow<OUT_BUF_WIDTH; ow++){
                // #pragma HLS unroll
        
                for(int oc=0; oc<OUT_BUF_DEPTH; oc++){
                    // #pragma HLS unroll
                    Y_buf[oc][oh][ow]   = bias_buf[oc];
                }
            }
        }
    }

    int IN_BUF_DEPTH_MUX;
    if (inp_channel < IN_BUF_DEPTH)
        IN_BUF_DEPTH_MUX = inp_channel;
    else
        IN_BUF_DEPTH_MUX = IN_BUF_DEPTH;

    int OUT_BUF_DEPTH_MUX;
    if (out_channel < OUT_BUF_DEPTH)
        OUT_BUF_DEPTH_MUX = out_channel;
    else
        OUT_BUF_DEPTH_MUX = OUT_BUF_DEPTH;

    ip_chan:
    for(int ic=0; ic<IN_BUF_DEPTH_MUX; ic++){
        fh:
        for(int fh=0; fh<3; fh++){
            fw:
            for(int fw=0; fw<3; fw++){
                
                op_chan:
                for(int oc=0; oc<OUT_BUF_DEPTH_MUX; oc++){
                    // #pragma HLS pipeline
                    
                    op_h:
                    for(int oh=0; oh<OUT_BUF_HEIGHT; oh++){
                        // #pragma HLS unroll  
                        #pragma HLS pipeline

                        op_w:
                        for(int ow=0; ow<OUT_BUF_WIDTH; ow++){
                            // #pragma HLS unroll
                            Y_buf[oc][oh][ow]   += X_buf[ic][oh+1+fh-1][ow+1+fw-1] * W_buf[oc][ic][fh][fw];
                        }
                    }
                }
            }
        }
    }

}

//--------------------------------------------------------------------------
// Function to store complete output tile block from BRAM to DRAM.
//--------------------------------------------------------------------------
template <int out_channel, int inp_height, int inp_width>
void store_output_tile_to_DRAM (
    fm_t out_fm[out_channel][inp_height][inp_width], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    int  ti,
    int  tj,
    int  b,
    bool relu_enable
)
{
    #pragma HLS inline
    const int depth_offset  =  b * OUT_BUF_DEPTH;
    const int height_offset = ti * OUT_BUF_HEIGHT;
    const int width_offset  = tj * OUT_BUF_WIDTH;

    int OUT_BUF_DEPTH_MUX = OUT_BUF_DEPTH;
    
    if(out_channel < OUT_BUF_DEPTH)
        OUT_BUF_DEPTH_MUX = out_channel;

    OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < OUT_BUF_DEPTH_MUX; f++)
    {
        OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < OUT_BUF_HEIGHT; i++)
        {
            OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < OUT_BUF_WIDTH; j++)
            {
                // ReLU in-place
                if(out_fm_buf[f][i][j] < (fm_t) 0 && relu_enable)
                {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = (fm_t) 0;
                }
                else
                {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = out_fm_buf[f][i][j];
                }
            }
        }
    }
}

//--------------------------------------------------------------------------
// Function to store maxpooled subsampled output tile block from BRAM to DRAM.
//--------------------------------------------------------------------------
template <int out_channel, int inp_height, int inp_width>
void store_maxpool_output_tile_to_DRAM (
    fm_t out_fm[out_channel][inp_height][inp_width],
    mk_t mask_fm[out_channel][inp_height][inp_width], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    int  ti,
    int  tj,
    int  b
)
{
    #pragma HLS inline
    const int depth_offset  =  b * OUT_BUF_DEPTH;
    const int height_offset = ti * OUT_BUF_HEIGHT;
    const int width_offset  = tj * OUT_BUF_WIDTH;

    fm_t max;
    mk_t pos;

    OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < OUT_BUF_HEIGHT; i+=2)
        {
            OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < OUT_BUF_WIDTH; j+=2)
            {
                max = out_fm_buf[f][i][j];
                pos = (mk_t) 0;

                if(max < out_fm_buf[f][i][j+1]){
                    max = out_fm_buf[f][i][j+1];
                    pos = (mk_t) 1;
                }

                if(max < out_fm_buf[f][i+1][j]){
                    max = out_fm_buf[f][i+1][j];
                    pos = (mk_t) 2;
                }

                if(max < out_fm_buf[f][i+1][j+1]){
                    max = out_fm_buf[f][i+1][j+1];
                    pos = (mk_t) 3;
                }

                out_fm[depth_offset + f][(height_offset + i)/2][(width_offset + j)/2] = max;
                mask_fm[depth_offset + f][(height_offset + i)/2][(width_offset + j)/2] = pos;
            }
        }
    }
}


///////////////
//--------------------------------------------------------------------------
// Function to load Matrix and Vector tile block from DRAM to on-chip BRAM
// for FC layer convolution.
//--------------------------------------------------------------------------
template<int inp_length, int out_length >
void load_mat_vec_tile_block_from_DRAM(
    fm_t VecB_DRAM[inp_length],                 // inp_feature_map
    wt_t MatA_DRAM[out_length][inp_length],     // Layer weigths 
    wt_t layer_bias[out_length],                // bias 
    fm_t VecB_buf[BLOCK_SIZE_N],                
    wt_t MatA_buf[BLOCK_SIZE_M][BLOCK_SIZE_N], 
    wt_t layer_bias_buf[BLOCK_SIZE_M],
    int tm,
    int tn
)
{
    #pragma HLS inline off

    const int kernel_offset = tm * BLOCK_SIZE_M;
    const int channel_offset = tn * BLOCK_SIZE_N;

    // Read in the data from DRAM to BRAM
    Read_A: for(int i = 0; i < BLOCK_SIZE_M; i++)
                for(int j = 0; j < BLOCK_SIZE_N; j++) 
                    MatA_buf[i][j] = MatA_DRAM[kernel_offset + i][channel_offset + j];

    Read_B: for(int i = 0; i < BLOCK_SIZE_N; i++)
                VecB_buf[i] = VecB_DRAM[channel_offset + i];
    
    Read_bias: for(int i = 0; i < BLOCK_SIZE_M; i++)
                layer_bias_buf[i] = layer_bias[kernel_offset + i];
}

template<int inp_length, int out_length >
void load_transpose_mat_vec_tile_block_from_DRAM(
    fm_t VecB_DRAM[inp_length],                 // inp_feature_map
    wt_t MatA_DRAM[inp_length][out_length],     // Layer weigths 
    fm_t VecB_buf[BLOCK_SIZE_N],                
    wt_t MatA_buf[BLOCK_SIZE_M][BLOCK_SIZE_N],
    wt_t layer_bias_buf[BLOCK_SIZE_M],
    int tm,
    int tn
)
{
    #pragma HLS inline off

    const int kernel_offset = tm * BLOCK_SIZE_M;
    const int channel_offset = tn * BLOCK_SIZE_N;

    // Read in the data from DRAM to BRAM
    Read_A: for(int i = 0; i < BLOCK_SIZE_M; i++)
                for(int j = 0; j < BLOCK_SIZE_N; j++) 
                    MatA_buf[i][j] = MatA_DRAM[channel_offset + j][kernel_offset + i];

    Read_B: for(int i = 0; i < BLOCK_SIZE_N; i++)
                VecB_buf[i] = VecB_DRAM[channel_offset + i];
    
    Read_bias: for(int i = 0; i < BLOCK_SIZE_M; i++)
                layer_bias_buf[i] = 0;

}

template<int inp_length, int out_length, int inp_channel, int inp_height, int inp_width >
void load_flattened_mat_vec_tile_block_from_DRAM(
    fm_t in_fm[inp_channel][inp_height][inp_width],
    wt_t MatA_DRAM[out_length][inp_length],     // Layer weigths 
    wt_t layer_bias[out_length],                // bias 
    fm_t VecB_buf[BLOCK_SIZE_N],                
    wt_t MatA_buf[BLOCK_SIZE_M][BLOCK_SIZE_N], 
    wt_t layer_bias_buf[BLOCK_SIZE_M],
    int tm,
    int tn
)
{
    #pragma HLS inline off

    const int kernel_offset = tm * BLOCK_SIZE_M;
    const int channel_offset = tn * BLOCK_SIZE_N;

    int ic, ih, iw, index;

    // Read in the data from DRAM to BRAM
    Read_A: for(int i = 0; i < BLOCK_SIZE_M; i++)
                for(int j = 0; j < BLOCK_SIZE_N; j++) 
                    MatA_buf[i][j] = MatA_DRAM[kernel_offset + i][channel_offset + j];

    Read_B: for(int i = 0; i < BLOCK_SIZE_N; i++){
                index = channel_offset + i;
                ic = (int) index/inp_channel;
                index = index - ic*inp_channel;
                ih = (int) index/inp_height;
                iw = index - ih*inp_height;
                VecB_buf[i] = in_fm[ic][ih][iw];
    }
    
    Read_bias: for(int i = 0; i < BLOCK_SIZE_M; i++)
                layer_bias_buf[i] = layer_bias[kernel_offset + i];
}

//--------------------------------------------------------------------------
// Function to perform matrix-vec multiplication for FC layer
//--------------------------------------------------------------------------
void mat_vec_mul(
    fm_t VecB_buf[BLOCK_SIZE_N],                //inp_feature_map
    wt_t MatA_buf[BLOCK_SIZE_M][BLOCK_SIZE_N], //weights
    wt_t layer_bias_buf[BLOCK_SIZE_M],         //bias
    fm_t VecC_buf[BLOCK_SIZE_M],               //output_feature_map
    int tn
)
{
    #pragma HLS inline off
    // initialization of output vector
    if (tn == 0) {
        Crow_buf_init: for(int m = 0; m < BLOCK_SIZE_M; m++)
            #pragma HLS unroll
            VecC_buf[m] = layer_bias_buf[m];
    }

    // Compute the matrix_vec multiplication
    mat_vec_row: for(int n = 0; n < BLOCK_SIZE_N; n++) {
        // #pragma HLS pipeline
        mat_vec_col: for(int m = 0; m < BLOCK_SIZE_M; m++) { 
            // #pragma HLS unroll 
            VecC_buf[m] += MatA_buf[m][n] * VecB_buf[n];
            }
        }

}


//--------------------------------------------------------------------------
// Function to store complete output tile block from BRAM to DRAM for FC layer.
//--------------------------------------------------------------------------
template <int out_length>
void store_output_vec_tile_to_DRAM(
    fm_t VecC_DRAM[out_length],
    mk_t relu_mask[out_length],
    fm_t VecC_buf[BLOCK_SIZE_M],
    int tm,
    bool relu_enable
)
{
    #pragma HLS inline off

    const int kernel_offset = tm * BLOCK_SIZE_M;

    int BLOCK_SIZE_M_MUX;
    if (out_length < BLOCK_SIZE_M)
        BLOCK_SIZE_M_MUX = out_length;
    else
        BLOCK_SIZE_M_MUX = BLOCK_SIZE_M;

    Crow_dram: for(int m = 0; m < BLOCK_SIZE_M_MUX; m++)
        if(relu_enable && VecC_buf[m] < 0){
            VecC_DRAM[kernel_offset + m] = 0;
            relu_mask[kernel_offset + m] = (mk_t) 1;
        }
        else {
            VecC_DRAM[kernel_offset + m] = VecC_buf[m]; 
            relu_mask[kernel_offset + m] = (mk_t) 0;
        }
}