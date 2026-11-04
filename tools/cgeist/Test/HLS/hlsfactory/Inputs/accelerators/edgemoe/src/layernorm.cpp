#include "../include/layernorm.hpp"
#include <hls_math.h>

wt_norm_t norm1_weights[FEATURE_DIM];
wt_bias_t norm1_bias[FEATURE_DIM];
wt_norm_t norm2_weights[FEATURE_DIM];
wt_bias_t norm2_bias[FEATURE_DIM];
fm_t norm_eps;

void load_norms(
    wt_norm_t norm_weights[NUM_LAYER_NORMS][FEATURE_DIM],
    wt_bias_t norm_bias[NUM_LAYER_NORMS][FEATURE_DIM]
)
{
    #pragma HLS inline off

    {
        hls::vector<wt_bias_t, FEATURE_BLOCK_SIZE>* bias_blocks = reinterpret_cast<hls::vector<wt_bias_t, FEATURE_BLOCK_SIZE>*>(norm_bias[NORM_1]);
        FOR_BLOCK(dim_out, FEATURE_DIM, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS pipeline

            hls::vector<wt_bias_t, FEATURE_BLOCK_SIZE> bias_block = bias_blocks[dim_out_block];
            FOR_OFFSET(dim_out)
            {
                norm1_bias[dim_out] = bias_block[dim_out_offset];
            }
        }
    }

    {
        hls::vector<wt_norm_t, FEATURE_BLOCK_SIZE>* weights_blocks = reinterpret_cast<hls::vector<wt_norm_t, FEATURE_BLOCK_SIZE>*>(norm_weights[NORM_1]);
        FOR_BLOCK(dim_out, FEATURE_DIM, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS pipeline

            hls::vector<wt_norm_t, FEATURE_BLOCK_SIZE> weights_block = weights_blocks[dim_out_block];
            FOR_OFFSET(dim_out)
            {
                norm1_weights[dim_out] = weights_block[dim_out_offset];
            }
        }
    }

    {
        hls::vector<wt_bias_t, FEATURE_BLOCK_SIZE>* bias_blocks = reinterpret_cast<hls::vector<wt_bias_t, FEATURE_BLOCK_SIZE>*>(norm_bias[NORM_2]);
        FOR_BLOCK(dim_out, FEATURE_DIM, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS pipeline

            hls::vector<wt_bias_t, FEATURE_BLOCK_SIZE> bias_block = bias_blocks[dim_out_block];
            FOR_OFFSET(dim_out)
            {
                norm2_bias[dim_out] = bias_block[dim_out_offset];
            }
        }
    }

    {
        hls::vector<wt_norm_t, FEATURE_BLOCK_SIZE>* weights_blocks = reinterpret_cast<hls::vector<wt_norm_t, FEATURE_BLOCK_SIZE>*>(norm_weights[NORM_2]);
        FOR_BLOCK(dim_out, FEATURE_DIM, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS pipeline

            hls::vector<wt_norm_t, FEATURE_BLOCK_SIZE> weights_block = weights_blocks[dim_out_block];
            FOR_OFFSET(dim_out)
            {
                norm2_weights[dim_out] = weights_block[dim_out_offset];
            }
        }
    }
}

void layernorm_accumulate(fm_blocks_t& x, fm_blocks_t& x_patch, fm_t& mean, fm_t& mean_sq)
{
    #pragma HLS inline off

    mean = 0.0;
    mean_sq = 0.0;

    FOR_BLOCK(dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
    {
        #pragma HLS pipeline rewind

        fm_block_t x_block = x[dim_block];
        fm_t partial_mean = 0.0;
        fm_t partial_mean_sq = 0.0;

        FOR_OFFSET(dim)
        {
            fm_t x_dim = x_block[dim_offset];
            fm_t x_dim_mean_term = x_dim * fm_t(1.0 / FEATURE_DIM);
            partial_mean += x_dim_mean_term;
            partial_mean_sq += x_dim * x_dim_mean_term;
        }

        x_patch[dim_block] = x_block;
        mean += partial_mean;
        mean_sq += partial_mean_sq;
    }
}

void layernorm_output(
    fm_blocks_t& out,
    fm_blocks_t& x_patch,
    fm_t& mean,
    fm_t& mean_sq,
    wt_norm_t weights[FEATURE_DIM],
    wt_bias_t bias[FEATURE_DIM]
)
{
    #pragma HLS inline off
    #pragma HLS array_reshape variable=weights cyclic factor=FEATURE_BLOCK_SIZE dim=1
    #pragma HLS array_reshape variable=bias cyclic factor=FEATURE_BLOCK_SIZE dim=1

    fm_t sq_mean = mean * mean;
    fm_t variance = mean_sq - sq_mean + norm_eps;
    fm_t rstddev = fm_t(1) / hls::sqrt(variance);

    FOR_BLOCK(dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
    {
        #pragma HLS pipeline rewind

        fm_block_t x_block = x_patch[dim_block];
        FOR_OFFSET(dim)
        {
            fm_t x_dim = x_block[dim_offset];
            x_dim -= mean;
            x_dim *= rstddev;
            x_dim *= weights[dim];
            x_dim += bias[dim];
            x_block[dim_offset] = x_dim;
        }
        out[dim_block] = x_block;
    }
}

void compute_norm(
    patch_blocks_t x,
    patch_blocks_t out,
    wt_norm_t weights[FEATURE_DIM],
    wt_bias_t bias[FEATURE_DIM]
)
{
    #pragma HLS inline off

    FOR_EACH(patch, NUM_PATCHES)
    {
        #pragma HLS dataflow

        fm_blocks_t x_patch;
        fm_t mean;
        fm_t mean_sq;

        layernorm_accumulate(x[patch], x_patch, mean, mean_sq);
        layernorm_output(out[patch], x_patch, mean, mean_sq, weights, bias);
    }
}

void compute_norm1(patch_blocks_t x, patch_blocks_t out)
{
    #pragma HLS inline

    compute_norm(x, out, norm1_weights, norm1_bias);
}

void compute_norm2(patch_blocks_t x, patch_blocks_t out)
{
    #pragma HLS inline

    compute_norm(x, out, norm2_weights, norm2_bias);
}
