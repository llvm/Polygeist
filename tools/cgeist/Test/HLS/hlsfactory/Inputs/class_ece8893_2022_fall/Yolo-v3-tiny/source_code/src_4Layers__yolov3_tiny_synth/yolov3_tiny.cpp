#include "conv.h"

void tiled_conv_maxpool_id3_0 (
    fm_t input_feature_map[MAX_INPUT_DEPTH][416][416],
    wt_t layer_conv_weights[1024][1024][3][3],
    fm_t output_feature_map[1024][416][416],
    int id,
    int ih,
    int iw,
    int od
)
{    
    //--------------------------------------------------------------------------
    // On-chip buffers
    //--------------------------------------------------------------------------
    fm_t conv_in_buf[IN_BUF_DEPTH1][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    wt_t conv_wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH1][3][3];
    fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];
    
    // Partial output storage buffer   
    fm_t partial_out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];

    // Maxpool output storage buffer
    fm_t maxpool_out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT/2][OUT_BUF_WIDTH/2];

    // Process each tile iteratively
    for(int ti = 0; ti < (416/OUT_BUF_HEIGHT); ti++)
    {
        for(int tj = 0; tj < (416/OUT_BUF_WIDTH); tj++)
        {
            std::cout << "Processing Tile " << ti*(iw/OUT_BUF_WIDTH) + tj + 1;
            std::cout << "/" << (416/OUT_BUF_HEIGHT) * (416/OUT_BUF_WIDTH) << std::endl;    
            
            for(int b = 0; b < (16/OUT_BUF_DEPTH); b++)
            {
                for(int d = 0; d < (3/IN_BUF_DEPTH1); d++)
                {
                    load_input_tile_block_from_DRAM_id3(conv_in_buf, input_feature_map, ti, tj, d);
                    load_layer_params_from_DRAM_id3_0(conv_wt_buf, layer_conv_weights, b, d);
                    conv_3x3_id3(conv_out_buf, conv_in_buf, conv_wt_buf);
                    save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, d);
                }
                max_pool_2D(partial_out_fm_buf, maxpool_out_fm_buf);
                store_maxpool_output_tile_to_DRAM(output_feature_map, maxpool_out_fm_buf, ti, tj, b);
            }
        }
    }
}

void tiled_conv_maxpool_id16_0 (
    fm_t input_feature_map[1024][416][416],
    wt_t layer_conv_weights[1024][1024][3][3],
    fm_t output_feature_map[1024][416][416],
    fm_t conv_output_feature_map[1024][416][416],
    int id,
    int ih,
    int iw,
    int od
)
{    
    //--------------------------------------------------------------------------
    // On-chip buffers
    //--------------------------------------------------------------------------
    fm_t conv_in_buf_Ping[IN_BUF_DEPTH2][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    fm_t conv_in_buf_Pong[IN_BUF_DEPTH2][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    wt_t conv_wt_buf_Ping[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3];
    wt_t conv_wt_buf_Pong[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3];
    fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];
    
    // Partial output storage buffer   
    fm_t partial_out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];

    // Maxpool output storage buffer
    fm_t maxpool_out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT/2][OUT_BUF_WIDTH/2];

    // Process each tile iteratively
    for(int ti = 0; ti < (208/OUT_BUF_HEIGHT); ti++)
    {
        for(int tj = 0; tj < (208/OUT_BUF_WIDTH); tj++)
        {
            std::cout << "Processing Tile " << ti*(iw/OUT_BUF_WIDTH) + tj + 1;
            std::cout << "/" << (208/OUT_BUF_HEIGHT) * (208/OUT_BUF_WIDTH) << std::endl;    
            
            for(int b = 0; b < (32/OUT_BUF_DEPTH); b++)
            {
                load_input_tile_block_from_DRAM_id16(conv_in_buf_Ping, input_feature_map, ti, tj, 0);
                load_layer_params_from_DRAM_id16(conv_wt_buf_Ping, layer_conv_weights, b, 0);
                for(int d = 0; d < (16/IN_BUF_DEPTH2) - 1; d++)
                {
                //#pragma HLS DATAFLOW
                    if(d % 2 == 0)
                    {
                        conv_3x3_id16(conv_out_buf, conv_in_buf_Ping, conv_wt_buf_Ping);
                        save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, d);
                        load_input_tile_block_from_DRAM_id16(conv_in_buf_Pong, input_feature_map, ti, tj, d+1);
                        load_layer_params_from_DRAM_id16(conv_wt_buf_Pong, layer_conv_weights, b, d+1);
                    }
                    else if (d % 2 == 1)
                    {
                        conv_3x3_id16(conv_out_buf, conv_in_buf_Pong, conv_wt_buf_Pong);
                        save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, d);
                        load_input_tile_block_from_DRAM_id16(conv_in_buf_Ping, input_feature_map, ti, tj, d+1);
                        load_layer_params_from_DRAM_id16(conv_wt_buf_Ping, layer_conv_weights, b, d+1);
                    }
                }
                if (((16/IN_BUF_DEPTH2) - 1) % 2 == 0)
                {
                    conv_3x3_id16(conv_out_buf, conv_in_buf_Ping, conv_wt_buf_Ping);
                    save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, (16/IN_BUF_DEPTH2) - 1);
                }
                else if (((16/IN_BUF_DEPTH2) - 1) % 2 == 1)
                {
                    conv_3x3_id16(conv_out_buf, conv_in_buf_Pong, conv_wt_buf_Pong);
                    save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, (16/IN_BUF_DEPTH2) - 1);
                }
                //#pragma HLS DATAFLOW
                store_output_tile_to_DRAM_ver1(conv_output_feature_map, partial_out_fm_buf, ti, tj, b);
                max_pool_2D(partial_out_fm_buf, maxpool_out_fm_buf);
                store_maxpool_output_tile_to_DRAM(output_feature_map, maxpool_out_fm_buf, ti, tj, b);
            }
        }
    }
}


void tiled_conv_maxpool_id16_1 (
    fm_t input_feature_map[1024][416][416],
    wt_t layer_conv_weights[1024][1024][3][3],
    fm_t output_feature_map[1024][416][416],
    fm_t conv_output_feature_map[1024][416][416],
    int id,
    int ih,
    int iw,
    int od
)
{    
    //--------------------------------------------------------------------------
    // On-chip buffers
    //--------------------------------------------------------------------------
    fm_t conv_in_buf_Ping[IN_BUF_DEPTH2][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    fm_t conv_in_buf_Pong[IN_BUF_DEPTH2][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    wt_t conv_wt_buf_Ping[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3];
    wt_t conv_wt_buf_Pong[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3];
    fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];
    
    // Partial output storage buffer   
    fm_t partial_out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];

    // Maxpool output storage buffer
    fm_t maxpool_out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT/2][OUT_BUF_WIDTH/2];

    // Process each tile iteratively
    for(int ti = 0; ti < (104/OUT_BUF_HEIGHT); ti++)
    {
        for(int tj = 0; tj < (104/OUT_BUF_WIDTH); tj++)
        {
            std::cout << "Processing Tile " << ti*(104/OUT_BUF_WIDTH) + tj + 1;
            std::cout << "/" << (104/OUT_BUF_HEIGHT) * (104/OUT_BUF_WIDTH) << std::endl;    
            
            for(int b = 0; b < (64/OUT_BUF_DEPTH); b++)
            {
                load_input_tile_block_from_DRAM_id16(conv_in_buf_Ping, input_feature_map, ti, tj, 0);
                load_layer_params_from_DRAM_id16(conv_wt_buf_Ping, layer_conv_weights, b, 0);
                for(int d = 0; d < (32/IN_BUF_DEPTH2) - 1; d++)
                {
                //#pragma HLS DATAFLOW
                    if(d % 2 == 0)
                    {
                        conv_3x3_id16_0(conv_out_buf, conv_in_buf_Ping, conv_wt_buf_Ping);
                        save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, d);
                        load_input_tile_block_from_DRAM_id16(conv_in_buf_Pong, input_feature_map, ti, tj, d+1);
                        load_layer_params_from_DRAM_id16_0(conv_wt_buf_Pong, layer_conv_weights, b, d+1);
                    }
                    else if (d % 2 == 1)
                    {
                        conv_3x3_id16_1(conv_out_buf, conv_in_buf_Pong, conv_wt_buf_Pong);
                        save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, d);
                        load_input_tile_block_from_DRAM_id16(conv_in_buf_Ping, input_feature_map, ti, tj, d+1);
                        load_layer_params_from_DRAM_id16_1(conv_wt_buf_Ping, layer_conv_weights, b, d+1);
                    }
                }
                if (((32/IN_BUF_DEPTH2) - 1) % 2 == 0)
                {
                    conv_3x3_id16(conv_out_buf, conv_in_buf_Ping, conv_wt_buf_Ping);
                    save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, (32/IN_BUF_DEPTH2) - 1);
                }
                else if (((id/IN_BUF_DEPTH2) - 1) % 2 == 1)
                {
                    conv_3x3_id16(conv_out_buf, conv_in_buf_Pong, conv_wt_buf_Pong);
                    save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, (32/IN_BUF_DEPTH2) - 1);
                }
                //#pragma HLS DATAFLOW
                store_output_tile_to_DRAM_ver1(conv_output_feature_map, partial_out_fm_buf, ti, tj, b);
                max_pool_2D(partial_out_fm_buf, maxpool_out_fm_buf);
                store_maxpool_output_tile_to_DRAM(output_feature_map, maxpool_out_fm_buf, ti, tj, b);
            }
        }
    }
}

void tiled_conv_maxpool_id16_2 (
    fm_t input_feature_map[1024][416][416],
    wt_t layer_conv_weights[1024][1024][3][3],
    fm_t output_feature_map[1024][416][416],
    fm_t conv_output_feature_map[1024][416][416],
    int id,
    int ih,
    int iw,
    int od
)
{    
    //--------------------------------------------------------------------------
    // On-chip buffers
    //--------------------------------------------------------------------------
    fm_t conv_in_buf_Ping[IN_BUF_DEPTH2][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    fm_t conv_in_buf_Pong[IN_BUF_DEPTH2][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    wt_t conv_wt_buf_Ping[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3];
    wt_t conv_wt_buf_Pong[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3];
    fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];
    
    // Partial output storage buffer   
    fm_t partial_out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];

    // Maxpool output storage buffer
    fm_t maxpool_out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT/2][OUT_BUF_WIDTH/2];

    // Process each tile iteratively
    for(int ti = 0; ti < (52/OUT_BUF_HEIGHT); ti++)
    {
        for(int tj = 0; tj < (52/OUT_BUF_WIDTH); tj++)
        {
            std::cout << "Processing Tile " << ti*(52/OUT_BUF_WIDTH) + tj + 1;
            std::cout << "/" << (52/OUT_BUF_HEIGHT) * (52/OUT_BUF_WIDTH) << std::endl;    
            
            for(int b = 0; b < (128/OUT_BUF_DEPTH); b++)
            {
                load_input_tile_block_from_DRAM_id16(conv_in_buf_Ping, input_feature_map, ti, tj, 0);
                load_layer_params_from_DRAM_id16(conv_wt_buf_Ping, layer_conv_weights, b, 0);
                for(int d = 0; d < (id/IN_BUF_DEPTH2) - 1; d++)
                {
                //#pragma HLS DATAFLOW
                    if(d % 2 == 0)
                    {
                        conv_3x3_id16(conv_out_buf, conv_in_buf_Ping, conv_wt_buf_Ping);
                        save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, d);
                        load_input_tile_block_from_DRAM_id16(conv_in_buf_Pong, input_feature_map, ti, tj, d+1);
                        load_layer_params_from_DRAM_id16(conv_wt_buf_Pong, layer_conv_weights, b, d+1);
                    }
                    else if (d % 2 == 1)
                    {
                        conv_3x3_id16(conv_out_buf, conv_in_buf_Pong, conv_wt_buf_Pong);
                        save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, d);
                        load_input_tile_block_from_DRAM_id16(conv_in_buf_Ping, input_feature_map, ti, tj, d+1);
                        load_layer_params_from_DRAM_id16(conv_wt_buf_Ping, layer_conv_weights, b, d+1);
                    }
                }
                if (((64/IN_BUF_DEPTH2) - 1) % 2 == 0)
                {
                    conv_3x3_id16(conv_out_buf, conv_in_buf_Ping, conv_wt_buf_Ping);
                    save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, (64/IN_BUF_DEPTH2) - 1);
                }
                else if (((id/IN_BUF_DEPTH2) - 1) % 2 == 1)
                {
                    conv_3x3_id16(conv_out_buf, conv_in_buf_Pong, conv_wt_buf_Pong);
                    save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, (64/IN_BUF_DEPTH2) - 1);
                }
                //#pragma HLS DATAFLOW
                store_output_tile_to_DRAM_ver1(conv_output_feature_map, partial_out_fm_buf, ti, tj, b);
                max_pool_2D(partial_out_fm_buf, maxpool_out_fm_buf);
                store_maxpool_output_tile_to_DRAM(output_feature_map, maxpool_out_fm_buf, ti, tj, b);
            }
        }
    }
}

void tiled_conv_maxpool_id16_3 (
    fm_t input_feature_map[1024][416][416],
    wt_t layer_conv_weights[1024][1024][3][3],
    fm_t output_feature_map[1024][416][416],
    fm_t conv_output_feature_map[1024][416][416],
    int id,
    int ih,
    int iw,
    int od
)
{    
    //--------------------------------------------------------------------------
    // On-chip buffers
    //--------------------------------------------------------------------------
    fm_t conv_in_buf_Ping[IN_BUF_DEPTH2][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    fm_t conv_in_buf_Pong[IN_BUF_DEPTH2][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    wt_t conv_wt_buf_Ping[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3];
    wt_t conv_wt_buf_Pong[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3];
    fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];
    
    // Partial output storage buffer   
    fm_t partial_out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];

    // Maxpool output storage buffer
    fm_t maxpool_out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT/2][OUT_BUF_WIDTH/2];

    // Process each tile iteratively
    for(int ti = 0; ti < (ih/OUT_BUF_HEIGHT); ti++)
    {
        for(int tj = 0; tj < (iw/OUT_BUF_WIDTH); tj++)
        {
            std::cout << "Processing Tile " << ti*(iw/OUT_BUF_WIDTH) + tj + 1;
            std::cout << "/" << (ih/OUT_BUF_HEIGHT) * (iw/OUT_BUF_WIDTH) << std::endl;    
            
            for(int b = 0; b < (od/OUT_BUF_DEPTH); b++)
            {
                load_input_tile_block_from_DRAM_id16(conv_in_buf_Ping, input_feature_map, ti, tj, 0);
                load_layer_params_from_DRAM_id16(conv_wt_buf_Ping, layer_conv_weights, b, 0);
                for(int d = 0; d < (id/IN_BUF_DEPTH2) - 1; d++)
                {
                //#pragma HLS DATAFLOW
                    if(d % 2 == 0)
                    {
                        conv_3x3_id16(conv_out_buf, conv_in_buf_Ping, conv_wt_buf_Ping);
                        save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, d);
                        load_input_tile_block_from_DRAM_id16(conv_in_buf_Pong, input_feature_map, ti, tj, d+1);
                        load_layer_params_from_DRAM_id16(conv_wt_buf_Pong, layer_conv_weights, b, d+1);
                    }
                    else if (d % 2 == 1)
                    {
                        conv_3x3_id16(conv_out_buf, conv_in_buf_Pong, conv_wt_buf_Pong);
                        save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, d);
                        load_input_tile_block_from_DRAM_id16(conv_in_buf_Ping, input_feature_map, ti, tj, d+1);
                        load_layer_params_from_DRAM_id16(conv_wt_buf_Ping, layer_conv_weights, b, d+1);
                    }
                }
                if (((id/IN_BUF_DEPTH2) - 1) % 2 == 0)
                {
                    conv_3x3_id16(conv_out_buf, conv_in_buf_Ping, conv_wt_buf_Ping);
                    save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, (id/IN_BUF_DEPTH2) - 1);
                }
                else if (((id/IN_BUF_DEPTH2) - 1) % 2 == 1)
                {
                    conv_3x3_id16(conv_out_buf, conv_in_buf_Pong, conv_wt_buf_Pong);
                    save_partial_output_tile_block(partial_out_fm_buf, conv_out_buf, (id/IN_BUF_DEPTH2) - 1);
                }
                //#pragma HLS DATAFLOW
                store_output_tile_to_DRAM_ver1(conv_output_feature_map, partial_out_fm_buf, ti, tj, b);
                max_pool_2D(partial_out_fm_buf, maxpool_out_fm_buf);
                store_maxpool_output_tile_to_DRAM(output_feature_map, maxpool_out_fm_buf, ti, tj, b);
            }
        }
    }
}


void tiled_conv (
    fm_t input_feature_map[1024][416][416],
    wt_t layer_conv_weights[1024][1024][3][3],
    fm_t output_feature_map[1024][416][416],
    int id,
    int ih,
    int iw,
    int od
)
{    
    //--------------------------------------------------------------------------
    // On-chip buffers
    //--------------------------------------------------------------------------
    fm_t conv_in_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT2][IN_BUF_WIDTH2];
    wt_t conv_wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3];
    fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2];
    
    // Partial output storage buffer   
    fm_t partial_out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2];

    // Process each tile iteratively
    for(int ti = 0; ti < (ih/OUT_BUF_HEIGHT2); ti++)
    {
        for(int tj = 0; tj < (iw/OUT_BUF_WIDTH2); tj++)
        {
            std::cout << "Processing Tile " << ti*(iw/OUT_BUF_WIDTH2) + tj + 1;
            std::cout << "/" << (ih/OUT_BUF_HEIGHT2) * (iw/OUT_BUF_WIDTH2) << std::endl;    
            
            for(int b = 0; b < (od/OUT_BUF_DEPTH); b++)
            {
                for(int d = 0; d < (id/IN_BUF_DEPTH2); d++)
                {
                    load_input_tile_block_from_DRAM_conv(conv_in_buf, input_feature_map, ti, tj, d);
                    load_layer_params_from_DRAM_id16(conv_wt_buf, layer_conv_weights, b, d);
                    conv_ver3(conv_out_buf, conv_in_buf, conv_wt_buf);
                    save_partial_output_tile_block_conv(partial_out_fm_buf, conv_out_buf, d);
                }
                store_output_tile_to_DRAM(output_feature_map, partial_out_fm_buf, ti, tj, b);
            }
        }
    }
}

void tiled_conv_1x1 (
    fm_t input_feature_map[1024][416][416],
    wt_t layer_conv_weights[1024][1024][1][1],
    fm_t output_feature_map[1024][416][416],
    int id,
    int ih,
    int iw,
    int od
)
{    
    //--------------------------------------------------------------------------
    // On-chip buffers
    //--------------------------------------------------------------------------
    fm_t conv_in_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT2][IN_BUF_WIDTH2];
    wt_t conv_wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][1][1];
    fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2];
    
    // Partial output storage buffer   
    fm_t partial_out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2];

    // Process each tile iteratively
    for(int ti = 0; ti < (ih/OUT_BUF_HEIGHT2); ti++)
    {
        for(int tj = 0; tj < (iw/OUT_BUF_WIDTH2); tj++)
        {
            std::cout << "Processing Tile " << ti*(iw/OUT_BUF_WIDTH2) + tj + 1;
            std::cout << "/" << (ih/OUT_BUF_HEIGHT2) * (iw/OUT_BUF_WIDTH2) << std::endl;    
            
            for(int b = 0; b < (od/OUT_BUF_DEPTH); b++)
            {
                for(int d = 0; d < (id/IN_BUF_DEPTH2); d++)
                {
                    load_input_tile_block_from_DRAM_conv(conv_in_buf, input_feature_map, ti, tj, d);
                    load_layer_params_from_DRAM_conv1x1(conv_wt_buf, layer_conv_weights, b, d);
                    conv_1x1(conv_out_buf, conv_in_buf, conv_wt_buf);
                    save_partial_output_tile_block_conv(partial_out_fm_buf, conv_out_buf, d);
                }
                store_output_tile_to_DRAM(output_feature_map, partial_out_fm_buf, ti, tj, b);
            }
        }
    }
}

void tiled_conv_bias (
    fm_t input_feature_map[1024][416][416],
    wt_t layer_conv_weights[1024][1024][1][1],
    wt_t layer_bias[255],
    fm_t output_feature_map[1024][416][416],
    int id,
    int ih,
    int iw,
    int od
)
{    
    //--------------------------------------------------------------------------
    // On-chip buffers
    //--------------------------------------------------------------------------
    fm_t conv_in_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT2][IN_BUF_WIDTH2];
    wt_t conv_wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][1][1];
    wt_t conv_bias_buf[OUT_BUF_DEPTH];
    fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2];
    
    // Partial output storage buffer   
    fm_t partial_out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2];

    // Process each tile iteratively
    for(int ti = 0; ti < (ih/OUT_BUF_HEIGHT2); ti++)
    {
        for(int tj = 0; tj < (iw/OUT_BUF_WIDTH2); tj++)
        {
            std::cout << "Processing Tile " << ti*(iw/OUT_BUF_WIDTH2) + tj + 1;
            std::cout << "/" << (ih/OUT_BUF_HEIGHT2) * (iw/OUT_BUF_WIDTH2) << std::endl;    
            
            for(int b = 0; b < (od/OUT_BUF_DEPTH); b++)
            {
                for(int d = 0; d < (id/IN_BUF_DEPTH2); d++)
                {
                    load_input_tile_block_from_DRAM_conv(conv_in_buf, input_feature_map, ti, tj, d);
                    load_layer_params_from_DRAM_bias(conv_wt_buf, conv_bias_buf, layer_conv_weights, layer_bias, b, d);
                    conv_1x1(conv_out_buf, conv_in_buf, conv_wt_buf);
                    save_partial_output_tile_block_bias(partial_out_fm_buf, conv_out_buf, conv_bias_buf, d);
                }
                store_output_tile_to_DRAM(output_feature_map, partial_out_fm_buf, ti, tj, b);
            }
        }
    }
}

void tiled_maxpool_stride1(
    fm_t input_mp_feature_map[1024][416][416],
    fm_t output_mp_feature_map[1024][416][416]
)
{
    //--------------------------------------------------------------------------
    // On-chip buffers
    //--------------------------------------------------------------------------
    fm_t maxpool_in_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2+1][OUT_BUF_WIDTH2+1];
    fm_t maxpool_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2];

    for(int d = 0; d < (512/OUT_BUF_DEPTH); d++)
    {
        load_maxpool_input_tile_block_from_DRAM(maxpool_in_buf, input_mp_feature_map, d);
        max_pool_2D_stride1(maxpool_in_buf, maxpool_out_buf);
        store_maxpool_output_tile_to_DRAM_stride1(output_mp_feature_map, maxpool_out_buf, d);
    }
}

void tiled_upsample(
    fm_t input_us_feature_map[1024][416][416],
    fm_t output_us_feature_map[1024][416][416]
)
{
    //--------------------------------------------------------------------------
    // On-chip buffers
    //--------------------------------------------------------------------------
    fm_t upsample_in_buf[US_BUF_DEPTH][US_BUF_HEIGHT][US_BUF_WIDTH];
    fm_t upsample_out_buf[US_BUF_DEPTH][US_BUF_HEIGHT*2][US_BUF_WIDTH*2];

    for(int d = 0; d < (128/US_BUF_DEPTH); d++)
    {
        load_upsample_input_tile_block_from_DRAM(upsample_in_buf, input_us_feature_map, d);
        upsample_2D(upsample_in_buf, upsample_out_buf);
        store_upsample_output_tile_to_DRAM(output_us_feature_map, upsample_out_buf, d);
    }
}

void copy_output_to_input(
    fm_t input_feature_map[1024][416][416],
    fm_t output_feature_map[1024][416][416],
    int d,
    int h,
    int w
)
{
    for (int id = 0; id < d; id++)
        for (int ih = 0; ih < h; ih++)
            for (int iw = 0; iw < w; iw++)
            {
                input_feature_map[id][ih][iw] = output_feature_map[id][ih][iw];
            }
}

void yolov3_tiny (
    fm_t input_image[3][416][416],
    wt_t conv_layer_1_weights[1024][1024][3][3],
    wt_t conv_layer_2_weights[1024][1024][3][3],
    wt_t conv_layer_3_weights[1024][1024][3][3],
    wt_t conv_layer_4_weights[1024][1024][3][3],
    wt_t conv_layer_5_weights[1024][1024][3][3],
    wt_t conv_layer_6_weights[1024][1024][3][3],
    wt_t conv_layer_7_weights[1024][1024][3][3],
    wt_t conv_layer_8_weights[1024][1024][1][1],
    wt_t conv_layer_9_weights[1024][1024][3][3],
    wt_t conv_layer_10_weights[1024][1024][1][1],
    wt_t conv_layer_11_weights[1024][1024][1][1],
    wt_t conv_layer_12_weights[1024][1024][3][3],
    wt_t conv_layer_13_weights[1024][1024][1][1],
    wt_t bias_layer_10[255],
    wt_t bias_layer_13[255],
    fm_t input_feature_map[1024][416][416],
    fm_t output_feature_map[1024][416][416],
    fm_t output13_feature_map[1024][416][416],
    fm_t output8_feature_map[1024][416][416]
)
{
    //--------------------------------------------------------------------------
    // Defines interface IO ports for HLS. 
    //--------------------------------------------------------------------------
    #pragma HLS INTERFACE m_axi depth=3*416*416      port=input_image          bundle=fm
    #pragma HLS INTERFACE m_axi depth=16*3*3*3       port=conv_layer_1_weights bundle=wt
    #pragma HLS INTERFACE m_axi depth=32*16*3*3      port=conv_layer_2_weights bundle=wt
    #pragma HLS INTERFACE m_axi depth=64*32*3*3      port=conv_layer_3_weights bundle=wt
    #pragma HLS INTERFACE m_axi depth=128*64*3*3     port=conv_layer_4_weights bundle=wt
    #pragma HLS INTERFACE m_axi depth=256*128*3*3    port=conv_layer_5_weights bundle=wt
    #pragma HLS INTERFACE m_axi depth=512*256*3*3    port=conv_layer_6_weights bundle=wt
    #pragma HLS INTERFACE m_axi depth=1024*512*3*3   port=conv_layer_7_weights bundle=wt
    #pragma HLS INTERFACE m_axi depth=256*1024*1*1   port=conv_layer_8_weights bundle=wt
    #pragma HLS INTERFACE m_axi depth=512*256*3*3    port=conv_layer_9_weights bundle=wt
    #pragma HLS INTERFACE m_axi depth=255*512*1*1    port=conv_layer_10_weights bundle=wt
    #pragma HLS INTERFACE m_axi depth=128*256*1*1    port=conv_layer_11_weights bundle=wt
    #pragma HLS INTERFACE m_axi depth=256*384*3*3    port=conv_layer_12_weights bundle=wt
    #pragma HLS INTERFACE m_axi depth=255*256*1*1    port=conv_layer_13_weights bundle=wt
    #pragma HLS INTERFACE m_axi depth=255            port=bias_layer_10         bundle=wt
    #pragma HLS INTERFACE m_axi depth=255            port=bias_layer_13         bundle=wt
    #pragma HLS INTERFACE m_axi depth=1024*416*416   port=input_feature_map    bundle=fm
    #pragma HLS INTERFACE m_axi depth=1024*416*416   port=output_feature_map   bundle=fm
    #pragma HLS INTERFACE m_axi depth=1024*416*416   port=output13_feature_map   bundle=fm
    #pragma HLS INTERFACE m_axi depth=1024*416*416   port=output8_feature_map   bundle=fm

    #pragma HLS INTERFACE s_axilite register	port=return

    for (int id = 0; id < 3; id++)
        for (int ih = 0; ih < 416; ih++)
            for (int iw = 0; iw < 416; iw++)
            {
                input_feature_map[id][ih][iw] = input_image[id][ih][iw];
            }

    //Layers 0,1 - Convolution layer 1/Maxpool
    tiled_conv_maxpool_id3_0 (input_feature_map, 
                            conv_layer_1_weights,
                            output_feature_map,
                            3,
                            416,
                            416,
                            16
                            );
    copy_output_to_input(input_feature_map, output_feature_map, 16, 208, 208);
    
    //#pragma HLS dependence variable=input_feature_map intra true

    //Layerss 2,3 - Convolution layer 2/Maxpool
    tiled_conv_maxpool_id16_0 (input_feature_map, 
                            conv_layer_2_weights,
                            output_feature_map,
                            output8_feature_map,
                            16,
                            208,
                            208,
                            32
                            );
    copy_output_to_input(input_feature_map, output_feature_map, 32, 104, 104);

    // //Layers 4,5 - Convolution layer 3/Maxpool
    // tiled_conv_maxpool_id16_1 (input_feature_map, 
    //                         conv_layer_3_weights,
    //                         output_feature_map,
    //                         output8_feature_map,
    //                         32,
    //                         104,
    //                         104,
    //                         64
    //                         );
    // copy_output_to_input(input_feature_map, output_feature_map, 64, 52, 52);

    // //Layers 6,7 - Convolution layer 4/Maxpool
    // tiled_conv_maxpool_id16_2 (input_feature_map, 
    //                         conv_layer_4_weights,
    //                         output_feature_map,
    //                         output8_feature_map,
    //                         64,
    //                         52,
    //                         52,
    //                         128
    //                         );
    // copy_output_to_input(input_feature_map, output_feature_map, 128, 26, 26);

    // //Layers 8,9 - Convolution layer 5/Maxpool
    // tiled_conv_maxpool_id16_3 (input_feature_map, 
    //                         conv_layer_5_weights,
    //                         output_feature_map,
    //                         output8_feature_map,
    //                         128,
    //                         26,
    //                         26,
    //                         256
    //                         );
    // copy_output_to_input(input_feature_map, output_feature_map, 256, 13, 13);

    // //Layer 10 - Convolution layer 6
    // tiled_conv (input_feature_map, 
    //             conv_layer_6_weights,
    //             output_feature_map,
    //             256,
    //             13,
    //             13,
    //             512
    //             );
    // copy_output_to_input(input_feature_map, output_feature_map, 512, 13, 13);
    
    // //Layer 11 - Maxpool
    // tiled_maxpool_stride1 (input_feature_map,
    //                output_feature_map
    //               );
    // copy_output_to_input(input_feature_map, output_feature_map, 512, 13, 13);

    // //Layer 12 - Convolution layer 7
    // tiled_conv (input_feature_map, 
    //             conv_layer_7_weights,
    //             output_feature_map,
    //             512,
    //             13,
    //             13,
    //             1024
    //             );
    // copy_output_to_input(input_feature_map, output_feature_map, 1024, 13, 13);

    // //Layer 13 - Convolution layer 8
    // tiled_conv_1x1 (input_feature_map, 
    //                 conv_layer_8_weights,
    //                 output_feature_map,
    //                 1024,
    //                 13,
    //                 13,
    //                 256
    //                 );
    // copy_output_to_input(input_feature_map, output_feature_map, 256, 13, 13);

    // //Layer 17 - Route 13
    // copy_output_to_input(output13_feature_map, output_feature_map, 256, 13, 13);

    // //Layer 14 - Convolution layer 9
    // tiled_conv (input_feature_map, 
    //             conv_layer_9_weights,
    //             output_feature_map,
    //             256,
    //             13,
    //             13,
    //             512
    //             );
    // copy_output_to_input(input_feature_map, output_feature_map, 512, 13, 13);

    // //Layer 15 - Convolution layer 10
    // tiled_conv_bias(input_feature_map, 
    //                 conv_layer_10_weights,
    //                 bias_layer_10,
    //                 output_feature_map,
    //                 512,
    //                 13,
    //                 13,
    //                 255
    //                 );
    // copy_output_to_input(input_feature_map, output_feature_map, 255, 13, 13);

    // //TODO: Layer 16 - Yolo

    // //Layer 18 - Convolution layer 11
    // tiled_conv_1x1 (output13_feature_map,
    //                 conv_layer_11_weights,
    //                 output_feature_map,
    //                 256,
    //                 13,
    //                 13,
    //                 128
    //                 );
    // copy_output_to_input(input_feature_map, output_feature_map, 128, 13, 13);

    // //Layer 19 - Upsampling
    // tiled_upsample(input_feature_map,
    //                output_feature_map
    //               );

    // //Layer 20 - Route 19,8
    // copy_output_to_input(input_feature_map, output_feature_map, 128, 26, 26);
    // for (int id = 128; id < 384; id++)
    //     for (int ih = 0; ih < 26; ih++)
    //         for (int iw = 0; iw < 26; iw++)
    //         {
    //             input_feature_map[id][ih][iw] = output8_feature_map[id][ih][iw];
    //         }
    
    // //Layer 21 - Convolution layer 12
    // tiled_conv (input_feature_map, 
    //             conv_layer_12_weights,
    //             output_feature_map,
    //             384,
    //             26,
    //             26,
    //             256
    //             );
    // copy_output_to_input(input_feature_map, output_feature_map, 256, 26, 26);

    // //Layer 22 - Convolution layer 13
    // tiled_conv_bias(input_feature_map, 
    //                 conv_layer_13_weights,
    //                 bias_layer_13,
    //                 output_feature_map,
    //                 256,
    //                 26,
    //                 26,
    //                 255
    //                 );
    // copy_output_to_input(input_feature_map, output_feature_map, 255, 26, 26);

    //TODO: Layer 23 - Yolo
}