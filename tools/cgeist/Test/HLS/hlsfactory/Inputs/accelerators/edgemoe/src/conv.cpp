#include "../include/conv.hpp"
#include <hls_stream.h>

constexpr unsigned int IMAGE_BLOCK_SIZE = (AXI_XFER_BIT_WIDTH / pixel_t::width);
typedef hls::vector<pixel_t, IMAGE_BLOCK_SIZE> image_block_t;

wt_patch_embed_t patch_embed_weights[FEATURE_DIM][INPUT_CHANNELS][PATCH_HEIGHT][PATCH_WIDTH];
wt_bias_t patch_embed_bias[FEATURE_DIM];

template<unsigned int y_step, unsigned int y_limit, unsigned int y_iters>
void patch_embed_accumulate_read(
    image_t image,
    hls::stream<image_block_t>& image_stream,
    unsigned int y_base
)
{
    #pragma HLS inline off

    image_block_t (*image_ptr)[INPUT_HEIGHT][INPUT_WIDTH / IMAGE_BLOCK_SIZE] = reinterpret_cast<image_block_t (*)[INPUT_HEIGHT][INPUT_WIDTH / IMAGE_BLOCK_SIZE]>(image);

    FOR_EACH(channel, INPUT_CHANNELS)
    {
        FOR_OFFSET(y)
        {
            FOR_BLOCK(patch_x, INPUT_WIDTH / PATCH_WIDTH, IMAGE_BLOCK_SIZE / PATCH_WIDTH)
            {
                image_stream << image_ptr[channel][y][patch_x_block];
            }
        }
    }
}

template<unsigned int y_step, unsigned int y_limit, unsigned int y_iters>
void patch_embed_accumulate_compute(
    hls::stream<image_block_t>& image_stream,
    fm_blocks_t patches[INPUT_WIDTH / PATCH_WIDTH],
    unsigned int y_base
)
{
    #pragma HLS inline off
    #pragma HLS array_reshape variable=patches cyclic factor=(IMAGE_BLOCK_SIZE / PATCH_WIDTH) dim=1
    #pragma HLS array_reshape variable=patch_embed_weights cyclic factor=FEATURE_BLOCK_SIZE dim=1
    #pragma HLS array_reshape variable=patch_embed_weights cyclic factor=PATCH_WIDTH dim=4
    #pragma HLS array_reshape variable=patch_embed_bias cyclic factor=FEATURE_BLOCK_SIZE dim=1

    image_block_t image_block;

    FOR_EACH(channel, INPUT_CHANNELS)
    {
        FOR_OFFSET(y)
        {
            FOR_BLOCK(patch_x, INPUT_WIDTH / PATCH_WIDTH, IMAGE_BLOCK_SIZE / PATCH_WIDTH)
            {
                FOR_BLOCK(dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
                {
                    #pragma HLS pipeline

                    if (dim_block == 0)
                    {
                        #pragma HLS occurrence cycle=dim_iters

                        image_stream >> image_block;
                    }

                    if (channel == 0 && y_offset == 0)
                    {
                        fm_block_t bias_block;
                        FOR_OFFSET(dim)
                        {
                            bias_block[dim_offset] = patch_embed_bias[dim];
                        }

                        FOR_BLOCK(x, IMAGE_BLOCK_SIZE, PATCH_WIDTH)
                        {
                            unsigned int patch_x = patch_x_base + x_block;
                            patches[patch_x][dim_block] = bias_block;
                        }
                    }

                    FOR_BLOCK(x, IMAGE_BLOCK_SIZE, PATCH_WIDTH)
                    {
                        unsigned int patch_x = patch_x_base + x_block;
                        fm_block_t addend;
                        FOR_OFFSET(dim)
                        {
                            fm_t addend_dim = 0.0;
                            FOR_OFFSET(x)
                            {
                                addend_dim += image_block[x] * patch_embed_weights[dim][channel][y_offset][x_offset];
                            }
                            addend[dim_offset] = addend_dim;
                        }
                        patches[patch_x][dim_block] += addend;
                    }
                }
            }
        }
    }
}

template<unsigned int y_step, unsigned int y_limit, unsigned int y_iters>
void patch_embed_accumulate(
    image_t image,
    fm_blocks_t patches[INPUT_WIDTH / PATCH_WIDTH],
    unsigned int y_block
)
{
    #pragma HLS inline off
    #pragma HLS dataflow

    unsigned int y_base = y_block * PATCH_HEIGHT;
    hls::stream<image_block_t> image_stream;

    patch_embed_accumulate_read<y_step, y_limit, y_iters>(image, image_stream, y_base);
    patch_embed_accumulate_compute<y_step, y_limit, y_iters>(image_stream, patches, y_base);
}

void patch_embed_output(
    fm_blocks_t patches[INPUT_WIDTH / PATCH_WIDTH],
    patch_blocks_t out,
    patch_blocks_t pos_embed,
    unsigned int y_block
)
{
    #pragma HLS inline off

    unsigned int patch_base = y_block * (INPUT_WIDTH / PATCH_WIDTH) + 1;
    // +1 because the first patch is the cls_tokens

    FOR_EACH(patch_x, INPUT_WIDTH / PATCH_WIDTH)
    {
        FOR_BLOCK(dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS pipeline

            unsigned int patch = patch_base + patch_x;

            out[patch][dim_block] = patches[patch_x][dim_block] + pos_embed[patch][dim_block];
        }
    }
}

void compute_patch_embed(image_t x, patch_blocks_t out, patch_blocks_t pos_embed)
{
    #pragma HLS inline off

    static_assert(INPUT_WIDTH % IMAGE_BLOCK_SIZE == 0, "INPUT_WIDTH must be a multiple of IMAGE_BLOCK_SIZE");
    static_assert(IMAGE_BLOCK_SIZE % PATCH_WIDTH == 0, "IMAGE_BLOCK_SIZE must be a multiple of PATCH_WIDTH");

    FOR_BLOCK(y, INPUT_HEIGHT, PATCH_HEIGHT)
    {
        #pragma HLS dataflow

        fm_blocks_t patches[INPUT_WIDTH / PATCH_WIDTH];

        patch_embed_accumulate<y_step, y_limit, y_iters>(x, patches, y_block);
        patch_embed_output(patches, out, pos_embed, y_block);
    }

    FOR_BLOCK(dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
    {
        #pragma HLS pipeline

        out[0][dim_block] = pos_embed[0][dim_block];
    }
}
