#include "gradient.h"
#include "utils.h"

// using namespace std;

void tiled_conv(
    fm_t fixp_layer1_ifmap[3][IM_SIZE][IM_SIZE],        // input image
    fm_t fixp_layer2_ifmap[32][IM_SIZE][IM_SIZE],       // output of conv1
    fm_t fixp_layer3_ifmap[32][IM_SIZE][IM_SIZE],       // output of conv2
    fm_t fixp_layer4_ifmap[32][IM_SIZE/2][IM_SIZE/2],   // output of max_pool1
    fm_t fixp_layer5_ifmap[64][IM_SIZE/2][IM_SIZE/2],   // output of conv3
    fm_t fixp_layer6_ifmap[64][IM_SIZE/2][IM_SIZE/2],   // output of conv4
    fm_t fixp_layer7_ifmap[64][IM_SIZE/4][IM_SIZE/4],   // output of max_pool2
    fm_t fixp_layer8_ifmap[128],                        // output of fc1
    fm_t fixp_layer9_ifmap[128],                        // output of ReLU1
    fm_t fixp_layer10_ifmap[10],                        // output of fc2

    wt_t fixp_conv1_weights[32][3][3][3],
    wt_t fixp_conv1_bias[32],
    wt_t fixp_conv2_weights[32][32][3][3],
    wt_t fixp_conv2_bias[32],
    wt_t fixp_conv3_weights[64][32][3][3],
    wt_t fixp_conv3_bias[64],
    wt_t fixp_conv4_weights[64][64][3][3],
    
    wt_t fixp_conv4_bias[64],
    wt_t fixp_fc1_weights[128][4096],
    wt_t fixp_fc1_bias[128],
    wt_t fixp_fc2_weights[10][128],
    wt_t fixp_fc2_bias[10],

    fm_t fixp_grad1[128],
    fm_t fixp_grad2[128],
    fm_t fixp_grad3[4096],
    fm_t fixp_grad4[64][IM_SIZE/4][IM_SIZE/4],
    fm_t fixp_grad5[64][IM_SIZE/2][IM_SIZE/2],
    fm_t fixp_grad6[64][IM_SIZE/2][IM_SIZE/2],
    fm_t fixp_grad7[32][IM_SIZE/2][IM_SIZE/2],
    fm_t fixp_grad8[32][IM_SIZE][IM_SIZE],
    fm_t fixp_grad9[32][IM_SIZE][IM_SIZE],
    fm_t fixp_grad10[3][IM_SIZE][IM_SIZE],

    mk_t fixp_mask_maxpool2[64][IM_SIZE/4][IM_SIZE/4],
    mk_t fixp_mask_maxpool1[32][IM_SIZE/2][IM_SIZE/2]

)
{
    //--------------------------------------------------------------------------
    // Defines interface IO ports for HLS. 
    // You should NOT modify these pragmas.
    //--------------------------------------------------------------------------
    #pragma HLS INTERFACE m_axi depth=3*32*32   port=fixp_layer1_ifmap   bundle=fm
    #pragma HLS INTERFACE m_axi depth=32*32*32  port=fixp_layer2_ifmap   bundle=fm
    #pragma HLS INTERFACE m_axi depth=32*32*32  port=fixp_layer3_ifmap   bundle=fm
    #pragma HLS INTERFACE m_axi depth=8*32*32   port=fixp_layer4_ifmap   bundle=fm
    #pragma HLS INTERFACE m_axi depth=16*32*32  port=fixp_layer5_ifmap   bundle=fm
    #pragma HLS INTERFACE m_axi depth=16*32*32  port=fixp_layer6_ifmap   bundle=fm
    #pragma HLS INTERFACE m_axi depth=4*32*32   port=fixp_layer7_ifmap   bundle=fm
    #pragma HLS INTERFACE m_axi depth=128       port=fixp_layer8_ifmap   bundle=fm
    #pragma HLS INTERFACE m_axi depth=128       port=fixp_layer9_ifmap   bundle=fm
    #pragma HLS INTERFACE m_axi depth=10        port=fixp_layer10_ifmap  bundle=fm
    
    #pragma HLS INTERFACE m_axi depth=32*3*3*3    port=fixp_conv1_weights  bundle=wt
    #pragma HLS INTERFACE m_axi depth=32*32*3*3   port=fixp_conv2_weights  bundle=wt
    #pragma HLS INTERFACE m_axi depth=64*32*3*3   port=fixp_conv3_weights  bundle=wt
    #pragma HLS INTERFACE m_axi depth=64*64*3*3   port=fixp_conv4_weights  bundle=wt
    #pragma HLS INTERFACE m_axi depth=128*4096    port=fixp_fc1_weights    bundle=wt
    #pragma HLS INTERFACE m_axi depth=128*10      port=fixp_fc2_weights    bundle=wt

    #pragma HLS INTERFACE m_axi depth=32    port=fixp_conv1_bias  bundle=wt
    #pragma HLS INTERFACE m_axi depth=32    port=fixp_conv2_bias  bundle=wt
    #pragma HLS INTERFACE m_axi depth=64    port=fixp_conv3_bias  bundle=wt
    #pragma HLS INTERFACE m_axi depth=64    port=fixp_conv4_bias  bundle=wt
    #pragma HLS INTERFACE m_axi depth=128   port=fixp_fc1_bias    bundle=wt
    #pragma HLS INTERFACE m_axi depth=10    port=fixp_fc2_bias    bundle=wt
    
    #pragma HLS INTERFACE m_axi depth=128       port=fixp_grad1   bundle=fm
    #pragma HLS INTERFACE m_axi depth=128       port=fixp_grad2   bundle=fm
    #pragma HLS INTERFACE m_axi depth=4096      port=fixp_grad3   bundle=fm
    #pragma HLS INTERFACE m_axi depth=4*32*32   port=fixp_grad4   bundle=fm
    #pragma HLS INTERFACE m_axi depth=16*32*32  port=fixp_grad5   bundle=fm
    #pragma HLS INTERFACE m_axi depth=16*32*32  port=fixp_grad6   bundle=fm
    #pragma HLS INTERFACE m_axi depth=8*32*32   port=fixp_grad7   bundle=fm
    #pragma HLS INTERFACE m_axi depth=32*32*32  port=fixp_grad8   bundle=fm
    #pragma HLS INTERFACE m_axi depth=32*32*32  port=fixp_grad9   bundle=fm
    #pragma HLS INTERFACE m_axi depth=3*32*32   port=fixp_grad10  bundle=fm

    #pragma HLS INTERFACE s_axilite register	port=return


    std::cout <<"------------------------------\n";
    std::cout << "Entered tiled_conv function\n" ;

    //--------------------------------------------------------------------------
    // On-chip buffers
    // You should NOT modify the buffer dimensions!
    //--------------------------------------------------------------------------
    fm_t conv_in_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    wt_t conv_wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][3][3];
    wt_t conv_bias_buf[OUT_BUF_DEPTH];
    fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];

    fm_t vecB_buf[BLOCK_SIZE_N];
    fm_t vecC_buf[BLOCK_SIZE_M];
    wt_t matA_buf[BLOCK_SIZE_M][BLOCK_SIZE_N];
    wt_t fc_bias_buf[BLOCK_SIZE_M];

    // ------------------------ PRAGMAS ------------------------------ 

    // #pragma HLS array_partition variable=conv_out_buf dim=1 complete
    // #pragma HLS array_partition variable=conv_wt_buf  dim=1 complete

    // #pragma HLS array_partition variable=conv_out_buf dim=2  complete
    // #pragma HLS array_partition variable=conv_in_buf  dim=2  complete

    #pragma HLS array_partition variable=conv_out_buf dim=3  complete
    #pragma HLS array_partition variable=conv_in_buf  dim=3  complete

    // #pragma HLS array_partition variable=conv_bias_buf  complete

    // #pragma HLS array_partition variable=vecC_buf complete
    // #pragma HLS array_partition variable=matA_buf dim=1 complete

    // ------------------------ LAYERS ------------------------------- 

    std::cout << "----------------FORWARD PASS---------------------\n";

    std::cout << "----------------Layer 1: Conv1---------------------\n";

    fp_conv1:
    for(int ti = 0; ti < IM_SIZE/OUT_BUF_HEIGHT; ti++)
    {
        for(int tj = 0; tj < IM_SIZE/OUT_BUF_WIDTH; tj++)
        {
            std::cout << "Processing Tile " << ti*IM_SIZE/OUT_BUF_WIDTH + tj + 1;
            std::cout << "/" << IM_SIZE/OUT_BUF_HEIGHT * IM_SIZE/OUT_BUF_WIDTH << std::endl;    

            for(int b=0; b<32/OUT_BUF_DEPTH; b++){

                for(int d=0; d<1; d++){

                    load_input_tile_block_from_DRAM <3,IM_SIZE,IM_SIZE>(
                        conv_in_buf,
                        fixp_layer1_ifmap,
                        ti,
                        tj,
                        d
                    );

                    load_conv_layer_params_from_DRAM <32,3> (
                        conv_wt_buf,
                        conv_bias_buf,
                        fixp_conv1_weights,
                        fixp_conv1_bias,
                        b,
                        d
                    );

                    conv_3x3(
                        conv_out_buf,
                        conv_in_buf,
                        conv_wt_buf,
                        conv_bias_buf,
                        d,
                        3,
                        32
                    );
                }

                store_output_tile_to_DRAM <32, IM_SIZE, IM_SIZE> (
                    fixp_layer2_ifmap,
                    conv_out_buf,
                    ti,
                    tj,
                    b,
                    0
                );
            }
        }
    }

    std::cout << "----------------Layer 2: Conv2---------------------\n";
    
    fp_conv2:
    for(int ti = 0; ti < IM_SIZE/OUT_BUF_HEIGHT; ti++)
    {
        for(int tj = 0; tj < IM_SIZE/OUT_BUF_WIDTH; tj++)
        {
            std::cout << "Processing Tile " << ti*IM_SIZE/OUT_BUF_WIDTH + tj + 1;
            std::cout << "/" << IM_SIZE/OUT_BUF_HEIGHT * IM_SIZE/OUT_BUF_WIDTH << std::endl;    

            for(int b=0; b<32/OUT_BUF_DEPTH; b++){

                for(int d=0; d<32/IN_BUF_DEPTH; d++){
                
                    fp_conv2_load_in: load_input_tile_block_from_DRAM <32,IM_SIZE,IM_SIZE>(
                        conv_in_buf,
                        fixp_layer2_ifmap,
                        ti,
                        tj,
                        d
                    );

                    fp_conv2_load_wt: load_conv_layer_params_from_DRAM <32,32> (
                        conv_wt_buf,
                        conv_bias_buf,
                        fixp_conv2_weights,
                        fixp_conv2_bias,
                        b,
                        d
                    );

                    fp_conv2_conv3x3: conv_3x3(
                        conv_out_buf,
                        conv_in_buf,
                        conv_wt_buf,
                        conv_bias_buf,
                        d,
                        32,
                        32
                    );
                }

                fp_conv2_store_out: store_output_tile_to_DRAM <32, IM_SIZE, IM_SIZE> (
                    fixp_layer3_ifmap,
                    conv_out_buf,
                    ti,
                    tj,
                    b,
                    0
                );

                // Layer 4 maxpooling -- can remove the above function to skip layer3 storing to DRAM
                store_maxpool_output_tile_to_DRAM <32, IM_SIZE/2, IM_SIZE/2> (
                    fixp_layer4_ifmap,
                    fixp_mask_maxpool1,
                    conv_out_buf,
                    ti,
                    tj,
                    b
                );
            }
        }
    }

    std::cout << "----------------Layer 4: Conv3---------------------\n";

    fp_conv3:
    for(int ti = 0; ti < IM_SIZE/2/OUT_BUF_HEIGHT; ti++)
    {
        for(int tj = 0; tj < IM_SIZE/2/OUT_BUF_WIDTH; tj++)
        {
            std::cout << "Processing Tile " << ti*IM_SIZE/2/OUT_BUF_WIDTH + tj + 1;
            std::cout << "/" << IM_SIZE/2/OUT_BUF_HEIGHT * IM_SIZE/2/OUT_BUF_WIDTH << std::endl;    

            for(int b=0; b<64/OUT_BUF_DEPTH; b++){

                for(int d=0; d<32/IN_BUF_DEPTH; d++){
                
                    load_input_tile_block_from_DRAM <32,IM_SIZE/2,IM_SIZE/2>(
                        conv_in_buf,
                        fixp_layer4_ifmap,
                        ti,
                        tj,
                        d
                    );

                    load_conv_layer_params_from_DRAM <64,32> (
                        conv_wt_buf,
                        conv_bias_buf,
                        fixp_conv3_weights,
                        fixp_conv3_bias,
                        b,
                        d
                    );

                    conv_3x3(
                        conv_out_buf,
                        conv_in_buf,
                        conv_wt_buf,
                        conv_bias_buf,
                        d,
                        32,
                        64
                    );
                }

                store_output_tile_to_DRAM <64, IM_SIZE/2, IM_SIZE/2> (
                    fixp_layer5_ifmap,
                    conv_out_buf,
                    ti,
                    tj,
                    b,
                    0
                );

            }
        }
    }

    std::cout << "----------------Layer 5: Conv4---------------------\n";

    fp_conv4:
    for(int ti = 0; ti < IM_SIZE/2/OUT_BUF_HEIGHT; ti++)
    {
        for(int tj = 0; tj < IM_SIZE/2/OUT_BUF_WIDTH; tj++)
        {
            std::cout << "Processing Tile " << ti*IM_SIZE/2/OUT_BUF_WIDTH + tj + 1;
            std::cout << "/" << IM_SIZE/2/OUT_BUF_HEIGHT * IM_SIZE/2/OUT_BUF_WIDTH << std::endl;    

            for(int b=0; b<64/OUT_BUF_DEPTH; b++){

                for(int d=0; d<64/IN_BUF_DEPTH; d++){
                
                    load_input_tile_block_from_DRAM <64,IM_SIZE/2,IM_SIZE/2>(
                        conv_in_buf,
                        fixp_layer5_ifmap,
                        ti,
                        tj,
                        d
                    );

                    load_conv_layer_params_from_DRAM <64,64> (
                        conv_wt_buf,
                        conv_bias_buf,
                        fixp_conv4_weights,
                        fixp_conv4_bias,
                        b,
                        d
                    );

                    conv_3x3(
                        conv_out_buf,
                        conv_in_buf,
                        conv_wt_buf,
                        conv_bias_buf,
                        d,
                        64,
                        64
                    );
                }

                store_output_tile_to_DRAM <64, IM_SIZE/2, IM_SIZE/2> (
                    fixp_layer6_ifmap,
                    conv_out_buf,
                    ti,
                    tj,
                    b,
                    0
                );

                store_maxpool_output_tile_to_DRAM <64, IM_SIZE/4, IM_SIZE/4> (
                    fixp_layer7_ifmap,
                    fixp_mask_maxpool2,
                    conv_out_buf,
                    ti,
                    tj,
                    b
                );

            }
        }
    }
    
    // for (int i=0; i<8; i++){
    //     cout << fixp_layer7_ifmap[0][0][i] << endl;
    // }

    std::cout << "----------------Layer 7: FC1---------------------\n";

    mk_t relu_mask_fc1[128];

    fp_fc1:
    for(int tm = 0; tm < 128/BLOCK_SIZE_M; tm++){
        for(int tn = 0; tn < 4096/BLOCK_SIZE_N; tn++){
            load_flattened_mat_vec_tile_block_from_DRAM <64*8*8, 128, 64, 8, 8> (
                fixp_layer7_ifmap,
                fixp_fc1_weights,
                fixp_fc1_bias,
                vecB_buf,
                matA_buf,
                fc_bias_buf,
                tm,
                tn
            );

            mat_vec_mul(
                vecB_buf, 
                matA_buf, 
                fc_bias_buf,
                vecC_buf,
                tn
            );
        }

        store_output_vec_tile_to_DRAM <128>(
            fixp_layer9_ifmap,
            relu_mask_fc1,
            vecC_buf,
            tm,
            1
        ); 
    }

    std::cout << "----------------Layer 9: FC2---------------------\n";

    mk_t relu_mask_fc2[10];

    fp_fc2:
    for(int tm = 0; tm < 1; tm++){
        for(int tn = 0; tn < 128/BLOCK_SIZE_N; tn++){
            load_mat_vec_tile_block_from_DRAM <128,10> (
                fixp_layer9_ifmap,
                fixp_fc2_weights,
                fixp_fc2_bias,
                vecB_buf,
                matA_buf,
                fc_bias_buf,
                tm,
                tn
            );

            mat_vec_mul(
                vecB_buf, 
                matA_buf, 
                fc_bias_buf,
                vecC_buf,
                tn
            );
        }

        store_output_vec_tile_to_DRAM <10>(
            fixp_layer10_ifmap,
            relu_mask_fc2,
            vecC_buf,
            tm,
            0
        ); 
    }

    // for (int i=0; i<10; i++){
    //     std::cout << fixp_layer10_ifmap[i] << std::endl;
    // }

    std::cout << "----------------BACKWARD PASS---------------------\n";
    
    std::cout << "----------------FC2---------------------\n";
    bp_fc2:
    for (int i=0; i<128; i++){
        fixp_grad1[i] = (fm_t) fixp_fc2_weights[0][i];
        fixp_grad2[i] = (fm_t) fixp_grad1[i] * (1-relu_mask_fc1[i]);
    };

    mk_t relu_mask_fc1_bp[4096];

    std::cout << "----------------FC1---------------------\n";

    bp_fc1:
    for(int tm = 0; tm < 4096/BLOCK_SIZE_M; tm++){
        for(int tn = 0; tn < 128/BLOCK_SIZE_N; tn++){
            load_transpose_mat_vec_tile_block_from_DRAM <128,4096> (
                fixp_grad2,
                fixp_fc1_weights,
                vecB_buf,
                matA_buf,
                fc_bias_buf,
                tm,
                tn
            );

            mat_vec_mul(
                vecB_buf, 
                matA_buf, 
                fc_bias_buf,
                vecC_buf,
                tn
            );
        }


        store_output_vec_tile_to_DRAM <4096>(
            fixp_grad3,
            relu_mask_fc1_bp,
            vecC_buf,
            tm,
            0
        ); 
    }

    std::cout << "----------------Maxpool2---------------------\n";

    bp_maxp2:
    fm_t temp;
    for(int ic=0; ic<64; ic++){
        for(int ih=0; ih<IM_SIZE/4; ih++){
            for(int iw=0; iw<IM_SIZE/4; iw++){
                temp = fixp_grad3[64*ic + IM_SIZE*ih/4 + iw];
                
                fixp_grad4[ic][ih][iw] = temp;

                if(fixp_mask_maxpool2[ic][ih][iw] == 0)
                    fixp_grad5[ic][2*ih][2*iw]  = temp;
                else
                    fixp_grad5[ic][2*ih][2*iw]  = 0;

                if(fixp_mask_maxpool2[ic][ih][iw] == 1)
                    fixp_grad5[ic][2*ih][2*iw+1]  = temp;
                else
                    fixp_grad5[ic][2*ih][2*iw+1]  = 0;

                if(fixp_mask_maxpool2[ic][ih][iw] == 2)
                    fixp_grad5[ic][2*ih+1][2*iw]  = temp;
                else
                    fixp_grad5[ic][2*ih+1][2*iw]  = 0;

                if(fixp_mask_maxpool2[ic][ih][iw] == 3)
                    fixp_grad5[ic][2*ih+1][2*iw+1]  = temp;
                else
                    fixp_grad5[ic][2*ih+1][2*iw+1]  = 0;

            }
        }
    }

    std::cout << "----------------Conv 4---------------------\n";

    bp_conv4:
    for(int ti = 0; ti < IM_SIZE/2/OUT_BUF_HEIGHT; ti++)
    {
        for(int tj = 0; tj < IM_SIZE/2/OUT_BUF_WIDTH; tj++)
        {
            std::cout << "Processing Tile " << ti*IM_SIZE/2/OUT_BUF_WIDTH + tj + 1;
            std::cout << "/" << IM_SIZE/2/OUT_BUF_HEIGHT * IM_SIZE/2/OUT_BUF_WIDTH << std::endl;    

            for(int b=0; b<64/OUT_BUF_DEPTH; b++){

                for(int d=0; d<64/IN_BUF_DEPTH; d++){
                
                    load_input_tile_block_from_DRAM <64,IM_SIZE/2,IM_SIZE/2>(
                        conv_in_buf,
                        fixp_grad5,
                        ti,
                        tj,
                        d
                    );

                    load_transpose_flipped_conv_layer_params_from_DRAM <64,64> (
                        conv_wt_buf,
                        conv_bias_buf,
                        fixp_conv4_weights,
                        b,
                        d
                    );

                    conv_3x3(
                        conv_out_buf,
                        conv_in_buf,
                        conv_wt_buf,
                        conv_bias_buf,
                        d,
                        64,
                        64
                    );
                }

                store_output_tile_to_DRAM <64, IM_SIZE/2, IM_SIZE/2> (
                    fixp_grad6,
                    conv_out_buf,
                    ti,
                    tj,
                    b,
                    0
                );
            }
        }
    }    


    std::cout << "----------------Conv 3---------------------\n";

    bp_conv3:
    for(int ti = 0; ti < IM_SIZE/2/OUT_BUF_HEIGHT; ti++)
    {
        for(int tj = 0; tj < IM_SIZE/2/OUT_BUF_WIDTH; tj++)
        {
            std::cout << "Processing Tile " << ti*IM_SIZE/2/OUT_BUF_WIDTH + tj + 1;
            std::cout << "/" << IM_SIZE/2/OUT_BUF_HEIGHT * IM_SIZE/2/OUT_BUF_WIDTH << std::endl;    

            for(int b=0; b<32/OUT_BUF_DEPTH; b++){

                for(int d=0; d<64/IN_BUF_DEPTH; d++){
                
                    load_input_tile_block_from_DRAM <64,IM_SIZE/2,IM_SIZE/2>(
                        conv_in_buf,
                        fixp_grad6,
                        ti,
                        tj,
                        d
                    );

                    load_transpose_flipped_conv_layer_params_from_DRAM <32,64> (
                        conv_wt_buf,
                        conv_bias_buf,
                        fixp_conv3_weights,
                        b,
                        d
                    );

                    conv_3x3(
                        conv_out_buf,
                        conv_in_buf,
                        conv_wt_buf,
                        conv_bias_buf,
                        d,
                        64,
                        32
                    );
                }

                store_output_tile_to_DRAM <32, IM_SIZE/2, IM_SIZE/2> (
                    fixp_grad7,
                    conv_out_buf,
                    ti,
                    tj,
                    b,
                    0
                );
            }
        }
    }

    std::cout << "----------------Maxpool1---------------------\n";

    bp_maxp1:
    for(int ic=0; ic<64; ic++){
        for(int ih=0; ih<IM_SIZE/2; ih++){
            for(int iw=0; iw<IM_SIZE/2; iw++){
                temp = fixp_grad7[ic][ih][iw];

                if(fixp_mask_maxpool1[ic][ih][iw] == 0)
                    fixp_grad8[ic][2*ih][2*iw]  = temp;
                else
                    fixp_grad8[ic][2*ih][2*iw]  = 0;

                if(fixp_mask_maxpool1[ic][ih][iw] == 1)
                    fixp_grad8[ic][2*ih][2*iw+1]  = temp;
                else
                    fixp_grad8[ic][2*ih][2*iw+1]  = 0;

                if(fixp_mask_maxpool1[ic][ih][iw] == 2)
                    fixp_grad8[ic][2*ih+1][2*iw]  = temp;
                else
                    fixp_grad8[ic][2*ih+1][2*iw]  = 0;

                if(fixp_mask_maxpool1[ic][ih][iw] == 3)
                    fixp_grad8[ic][2*ih+1][2*iw+1]  = temp;
                else
                    fixp_grad8[ic][2*ih+1][2*iw+1]  = 0;

            }
        }
    }

    std::cout << "----------------Conv 2---------------------\n";

    bp_conv2:
    for(int ti = 0; ti < IM_SIZE/OUT_BUF_HEIGHT; ti++)
    {
        for(int tj = 0; tj < IM_SIZE/OUT_BUF_WIDTH; tj++)
        {
            std::cout << "Processing Tile " << ti*IM_SIZE/OUT_BUF_WIDTH + tj + 1;
            std::cout << "/" << IM_SIZE/OUT_BUF_HEIGHT * IM_SIZE/OUT_BUF_WIDTH << std::endl;    

            for(int b=0; b<32/OUT_BUF_DEPTH; b++){

                for(int d=0; d<32/IN_BUF_DEPTH; d++){
                
                    load_input_tile_block_from_DRAM <32,IM_SIZE,IM_SIZE>(
                        conv_in_buf,
                        fixp_grad8,
                        ti,
                        tj,
                        d
                    );

                    load_transpose_flipped_conv_layer_params_from_DRAM <32,32> (
                        conv_wt_buf,
                        conv_bias_buf,
                        fixp_conv2_weights,
                        b,
                        d
                    );

                    conv_3x3(
                        conv_out_buf,
                        conv_in_buf,
                        conv_wt_buf,
                        conv_bias_buf,
                        d,
                        32,
                        32
                    );
                }

                store_output_tile_to_DRAM <32, IM_SIZE, IM_SIZE> (
                    fixp_grad9,
                    conv_out_buf,
                    ti,
                    tj,
                    b,
                    0
                );
            }
        }
    }

    std::cout << "----------------Conv 1---------------------\n";

    bp_conv1:
    for(int ti = 0; ti < IM_SIZE/OUT_BUF_HEIGHT; ti++)
    {
        for(int tj = 0; tj < IM_SIZE/OUT_BUF_WIDTH; tj++)
        {
            std::cout << "Processing Tile " << ti*IM_SIZE/OUT_BUF_WIDTH + tj + 1;
            std::cout << "/" << IM_SIZE/OUT_BUF_HEIGHT * IM_SIZE/OUT_BUF_WIDTH << std::endl;    

            for(int b=0; b<1; b++){

                for(int d=0; d<32/IN_BUF_DEPTH; d++){
                
                    load_input_tile_block_from_DRAM <32,IM_SIZE,IM_SIZE>(
                        conv_in_buf,
                        fixp_grad9,
                        ti,
                        tj,
                        d
                    );

                    load_transpose_flipped_conv_layer_params_from_DRAM <3,32> (
                        conv_wt_buf,
                        conv_bias_buf,
                        fixp_conv1_weights,
                        b,
                        d
                    );

                    conv_3x3(
                        conv_out_buf,
                        conv_in_buf,
                        conv_wt_buf,
                        conv_bias_buf,
                        d,
                        32,
                        3
                    );
                }

                store_output_tile_to_DRAM <3, IM_SIZE, IM_SIZE> (
                    fixp_grad10,
                    conv_out_buf,
                    ti,
                    tj,
                    b,
                    0
                );
            }
        }
    }

}