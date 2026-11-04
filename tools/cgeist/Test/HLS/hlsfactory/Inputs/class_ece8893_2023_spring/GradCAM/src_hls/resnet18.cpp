#include "util.h"
#include "maxpool/max_pool.cpp"
#include "avg_pool/avg_pool.hpp"
#include "linear_fc/linear_fc.hpp"
#include "conv1/conv1.hpp"
#include "conv_ds/conv_ds.hpp"
#include "conv_3x3_s1/conv_3x3_s1.hpp"
#include "cam/cam.cpp"
#include "resize.hpp"

#ifdef CSIM_DEBUG
#include "sim_util.hpp"
std::string root_dir = "out/";
#define WRITE_TO_FILE(var, dim0, dim1, dim2) \
{ \
    std::vector<int> dims(3); \
    dims[0] = dim0; \
    dims[1] = dim1; \
    dims[2] = dim2; \
    write_to_file(root_dir + VAR_NAME(var) + ".bin", dims, var); \
}
#define WRITE_TO_FILE_NAME(var, name, dim0, dim1, dim2) \
{ \
    std::vector<int> dims(3); \
    dims[0] = dim0; \
    dims[1] = dim1; \
    dims[2] = dim2; \
    write_to_file(root_dir + name + ".bin", dims, var); \
}
#else
#define WRITE_TO_FILE(var, dim0, dim1, dim2)
#define WRITE_TO_FILE_NAME(var, name, dim0, dim1, dim2)
#endif

#define INP_SIDE 224
#define INP_DEPTH 3

#define CONV1_SIDE (INP_SIDE/2)
#define CONV1_DEPTH 64
#define CONV1_SIZE (CONV1_DEPTH * CONV1_SIDE * CONV1_SIDE)
#define MAXPOOL_SIDE (CONV1_SIDE/2)
#define MAXPOOL_DEPTH (CONV1_DEPTH)
#define MAXPOOL_SIZE (MAXPOOL_DEPTH * MAXPOOL_SIDE * MAXPOOL_SIDE)

#define L1_SIDE (CONV1_SIDE/2)
#define L1_DEPTH (CONV1_DEPTH)
#define L1_SIZE (L1_DEPTH * L1_SIDE * L1_SIDE)

#define L2_SIDE (L1_SIDE/2)
#define L2_DEPTH (L1_DEPTH*2)
#define L2_SIZE (L2_DEPTH * L2_SIDE * L2_SIDE)

#define L3_SIDE (L2_SIDE/2)
#define L3_DEPTH (L2_DEPTH*2)
#define L3_SIZE (L3_DEPTH * L3_SIDE * L3_SIDE)

#define L4_SIDE (L3_SIDE/2)
#define L4_DEPTH (L3_DEPTH*2)
#define L4_SIZE (L4_DEPTH * L4_SIDE * L4_SIDE)

#define AVG_POOL_SIZE L4_DEPTH
#define OUTPUT_SIZE 1000
#define CAM_SIZE 49
#define RESIZE_SIZE 224*224
// FM_DRAM offsets
#define CONV1_FM_OFFSET 0
#define MAXPOOL_FM_OFFSET (CONV1_FM_OFFSET + CONV1_SIZE)
#define L1_FM_OFFSET (MAXPOOL_FM_OFFSET + MAXPOOL_SIZE)
#define L2_FM_OFFSET (L1_FM_OFFSET + 2*L1_SIZE)
#define L3_FM_OFFSET (L2_FM_OFFSET + 2*L2_SIZE)
#define L4_FM_OFFSET (L3_FM_OFFSET + 2*L3_SIZE)
#define AVG_POOL_OFFSET (L4_FM_OFFSET + 2*L4_SIZE)
#define OUTPUT_OFFSET (AVG_POOL_OFFSET + AVG_POOL_SIZE)
#define CAM_OFFSET (OUTPUT_OFFSET + OUTPUT_SIZE)
#define RESIZE_OFFSET (CAM_OFFSET + CAM_SIZE)

#define FM_DRAM_SIZE (RESIZE_OFFSET + RESIZE_SIZE)
void resnet18(
        fm_t input[INP_DEPTH][INP_SIDE][INP_SIDE],
        fm_t output[1000],
        fm_t fm_dram[],
        wt_t conv1_weight[64][3][7][7],
        wt_t conv1_bias[64],
        // layer 1
        wt_t l10_c1_weight[64][64][3][3],
        wt_t l10_c1_bias[64],
        wt_t l10_c2_weight[64][64][3][3],
        wt_t l10_c2_bias[64],
        wt_t l11_c1_weight[64][64][3][3],
        wt_t l11_c1_bias[64],
        wt_t l11_c2_weight[64][64][3][3],
        wt_t l11_c2_bias[64],
        // layer 2
        wt_t l2_ds_weight[128][64][1][1],
        wt_t l2_ds_bias[128],
        wt_t l20_c1_weight[128][64][3][3],
        wt_t l20_c1_bias[128],
        wt_t l20_c2_weight[128][128][3][3],
        wt_t l20_c2_bias[128],
        wt_t l21_c1_weight[128][128][3][3],
        wt_t l21_c1_bias[128],
        wt_t l21_c2_weight[128][128][3][3],
        wt_t l21_c2_bias[128],
        // layer 3
        wt_t l3_ds_weight[256][128][1][1],
        wt_t l3_ds_bias[256],
        wt_t l30_c1_weight[256][128][3][3],
        wt_t l30_c1_bias[256],
        wt_t l30_c2_weight[256][256][3][3],
        wt_t l30_c2_bias[256],
        wt_t l31_c1_weight[256][256][3][3],
        wt_t l31_c1_bias[256],
        wt_t l31_c2_weight[256][256][3][3],
        wt_t l31_c2_bias[256],
        // layer 4
        wt_t l4_ds_weight[512][256][1][1],
        wt_t l4_ds_bias[512],
        wt_t l40_c1_weight[512][256][3][3],
        wt_t l40_c1_bias[512],
        wt_t l40_c2_weight[512][512][3][3],
        wt_t l40_c2_bias[512],
        wt_t l41_c1_weight[512][512][3][3],
        wt_t l41_c1_bias[512],
        wt_t l41_c2_weight[512][512][3][3],
        wt_t l41_c2_bias[512],
        // fc
        wt_t fc_weight[1000][512],
        wt_t fc_bias[1000],
        // cam output
        fm_t cam_output[7][7]
        )
{

    
     
    WRITE_TO_FILE(input, INP_DEPTH, INP_SIDE, INP_SIDE);

    fm_t *conv1_out = fm_dram;
    fm_t *maxpool_out = fm_dram + MAXPOOL_FM_OFFSET;
    fm_t *l1_out0 = fm_dram + L1_FM_OFFSET;
    fm_t *l1_out1 = maxpool_out;
    fm_t *l2_out0 = fm_dram + L2_FM_OFFSET;
    fm_t *l2_out1 = l2_out0 + L2_SIZE;
    fm_t *l3_out0 = fm_dram + L3_FM_OFFSET;
    fm_t *l3_out1 = l3_out0 + L3_SIZE;
    fm_t *l4_out0 = fm_dram + L4_FM_OFFSET;
    fm_t *l4_out1 = l4_out0 + L4_SIZE;
    fm_t *avgpool_out = fm_dram + AVG_POOL_OFFSET;
    fm_t *resnet_out = fm_dram + OUTPUT_OFFSET;
    //fm_t *cam_output = fm_dram + CAM_OFFSET; 
    fm_t *resizedHeatmap = fm_dram + RESIZE_OFFSET;

    #include "bundles.hpp"

    // conv1
    conv1::tiled_conv((fm_t (*)[112][112]) conv1_out, input, conv1_weight, conv1_bias);
    WRITE_TO_FILE(conv1_out, CONV1_DEPTH, CONV1_SIDE, CONV1_SIDE);

    // maxpool
    maxpool::maxpool2d(maxpool_out, conv1_out);
    WRITE_TO_FILE(maxpool_out, MAXPOOL_DEPTH, MAXPOOL_SIDE, MAXPOOL_SIDE);

    // layer 1 
    // block 0
    conv_3x3_s1::tiled_conv<L1_DEPTH, MAXPOOL_DEPTH, L1_SIDE, L1_SIDE>(l1_out0, l1_out1, (wt_t *) l10_c1_weight, l10_c1_bias, false);
    conv_3x3_s1::tiled_conv<L1_DEPTH, L1_DEPTH, L1_SIDE, L1_SIDE>(l1_out1, l1_out0, (wt_t *) l10_c2_weight, l10_c2_bias, true);
    WRITE_TO_FILE_NAME(l1_out0, "l10_c1_out", L1_DEPTH, L1_SIDE, L1_SIDE);
    WRITE_TO_FILE_NAME(l1_out1, "l10_c2_out", L1_DEPTH, L1_SIDE, L1_SIDE);
    // block 1
    conv_3x3_s1::tiled_conv<L1_DEPTH, L1_DEPTH, L1_SIDE, L1_SIDE>(l1_out0, l1_out1, (wt_t *) l11_c1_weight, l11_c1_bias, false);
    conv_3x3_s1::tiled_conv<L1_DEPTH, L1_DEPTH, L1_SIDE, L1_SIDE>(l1_out1, l1_out0, (wt_t *) l11_c2_weight, l11_c2_bias, true);
    WRITE_TO_FILE_NAME(l1_out0, "l11_c1_out", L1_DEPTH, L1_SIDE, L1_SIDE);
    WRITE_TO_FILE_NAME(l1_out1, "l11_c2_out", L1_DEPTH, L1_SIDE, L1_SIDE);

    // layer 2
    // downsample
    conv_ds::tiled_conv_ds<L2_DEPTH, L1_DEPTH, L1_SIDE, L1_SIDE>
        (l2_out1, maxpool_out, (fm_t *)l2_ds_weight, l2_ds_bias);
    WRITE_TO_FILE_NAME(l2_out1, "l2_ds_out", L2_DEPTH, L2_SIDE, L2_SIDE);
    // block 0
    conv_3x3_s1::tiled_conv<L2_DEPTH, L1_DEPTH, L1_SIDE, L1_SIDE>(l2_out0, l1_out1, (wt_t *) l20_c1_weight, l20_c1_bias, false, true);
    conv_3x3_s1::tiled_conv<L2_DEPTH, L2_DEPTH, L2_SIDE, L2_SIDE>(l2_out1, l2_out0, (wt_t *) l20_c2_weight, l20_c2_bias, true);
    WRITE_TO_FILE_NAME(l2_out0, "l20_c1_out", L2_DEPTH, L2_SIDE, L2_SIDE);
    WRITE_TO_FILE_NAME(l2_out1, "l20_c2_out", L2_DEPTH, L2_SIDE, L2_SIDE);
    // block 1
    conv_3x3_s1::tiled_conv<L2_DEPTH, L2_DEPTH, L2_SIDE, L2_SIDE>(l2_out0, l2_out1, (wt_t *) l21_c1_weight, l21_c1_bias, false);
    conv_3x3_s1::tiled_conv<L2_DEPTH, L2_DEPTH, L2_SIDE, L2_SIDE>(l2_out1, l2_out0, (wt_t *) l21_c2_weight, l21_c2_bias, true);
    WRITE_TO_FILE_NAME(l2_out0, "l21_c1_out", L2_DEPTH, L2_SIDE, L2_SIDE);
    WRITE_TO_FILE_NAME(l2_out1, "l21_c2_out", L2_DEPTH, L2_SIDE, L2_SIDE);

    // layer 3
    // downsample
    conv_ds::tiled_conv_ds<L3_DEPTH, L2_DEPTH, L2_SIDE, L2_SIDE>
        (l3_out1, l2_out1, (fm_t *)l3_ds_weight, l3_ds_bias);
    WRITE_TO_FILE_NAME(l3_out1, "l3_ds_out", L3_DEPTH, L3_SIDE, L3_SIDE);
    // block 0
    conv_3x3_s1::tiled_conv<L3_DEPTH, L2_DEPTH, L2_SIDE, L2_SIDE>(l3_out0, l2_out1, (wt_t *) l30_c1_weight, l30_c1_bias, false, true);
    conv_3x3_s1::tiled_conv<L3_DEPTH, L3_DEPTH, L3_SIDE, L3_SIDE>(l3_out1, l3_out0, (wt_t *) l30_c2_weight, l30_c2_bias, true);
    WRITE_TO_FILE_NAME(l3_out0, "l30_c1_out", L3_DEPTH, L3_SIDE, L3_SIDE);
    WRITE_TO_FILE_NAME(l3_out1, "l30_c2_out", L3_DEPTH, L3_SIDE, L3_SIDE);
    // block 1
    conv_3x3_s1::tiled_conv<L3_DEPTH, L3_DEPTH, L3_SIDE, L3_SIDE>(l3_out0, l3_out1, (wt_t *) l31_c1_weight, l31_c1_bias, false);
    conv_3x3_s1::tiled_conv<L3_DEPTH, L3_DEPTH, L3_SIDE, L3_SIDE>(l3_out1, l3_out0, (wt_t *) l31_c2_weight, l31_c2_bias, true);
    WRITE_TO_FILE_NAME(l3_out0, "l31_c1_out", L3_DEPTH, L3_SIDE, L3_SIDE);
    WRITE_TO_FILE_NAME(l3_out1, "l31_c2_out", L3_DEPTH, L3_SIDE, L3_SIDE);

    // layer 4
    // downsample
    conv_ds::tiled_conv_ds<L4_DEPTH, L3_DEPTH, L3_SIDE, L3_SIDE>(l4_out1, l3_out1, (fm_t *)l4_ds_weight, l4_ds_bias);
    WRITE_TO_FILE_NAME(l4_out1, "l4_ds_out", L4_DEPTH, L4_SIDE, L4_SIDE);
    // block 0
    conv_3x3_s1::tiled_conv<L4_DEPTH, L3_DEPTH, L3_SIDE, L3_SIDE>(l4_out0, l3_out1, (wt_t *) l40_c1_weight, l40_c1_bias, false, true);
    conv_3x3_s1::tiled_conv<L4_DEPTH, L4_DEPTH, L4_SIDE, L4_SIDE>(l4_out1, l4_out0, (wt_t *) l40_c2_weight, l40_c2_bias, true);
    WRITE_TO_FILE_NAME(l4_out0, "l40_c1_out", L4_DEPTH, L4_SIDE, L4_SIDE);
    WRITE_TO_FILE_NAME(l4_out1, "l40_c2_out", L4_DEPTH, L4_SIDE, L4_SIDE);
    // block 1
    conv_3x3_s1::tiled_conv<L4_DEPTH, L4_DEPTH, L4_SIDE, L4_SIDE>(l4_out0, l4_out1, (wt_t *) l41_c1_weight, l41_c1_bias, false);
    conv_3x3_s1::tiled_conv<L4_DEPTH, L4_DEPTH, L4_SIDE, L4_SIDE>(l4_out1, l4_out0, (wt_t *) l41_c2_weight, l41_c2_bias, true);
    WRITE_TO_FILE_NAME(l4_out0, "l41_c1_out", L4_DEPTH, L4_SIDE, L4_SIDE);
    WRITE_TO_FILE_NAME(l4_out1, "l41_c2_out", L4_DEPTH, L4_SIDE, L4_SIDE);

    // avgpool
    static fm_t avgpool_out_buf[512];
    avg_pool::avg_pool((fm_t (*)[7][7])l4_out1, avgpool_out_buf);
    WRITE_TO_FILE(avgpool_out_buf, AVG_POOL_SIZE, 1, 1);
    // fc
    static fm_t output_buf[1000];
    linear_fc::linear_fc(avgpool_out_buf, output_buf, fc_weight, fc_bias);
    WRITE_TO_FILE(output_buf, 1000, 1, 1);
    //cam
    cam::cam((fm_t (*)[7])cam_output, l4_out1, fc_weight, output_buf);
    WRITE_TO_FILE(cam_output, 7, 7, 1);

    #ifdef CSIM_DEBUG
    //resize heatmap
    resize((fm_t (*)[224]) resizedHeatmap, (fm_t (*)[7])avgpool_out);
    WRITE_TO_FILE(resizedHeatmap, 224, 224, 1);
    #endif
}
