#include "ap_fixed.h"
#include "dcl.h"
#include "hls_math.h"

// Apply the hls::exp on the elements of a vector
tile_fixed_t exp_vector(tile_fixed_t& vec)
{
    tile_fixed_t result = {};
    EXP_VECTOR:
    for (int i = 0; i < NUM_PACKED_TILE; ++i) {
#pragma HLS unroll        
        result[i] = hls::exp(vec[i]); 
    }
    return result;
}

// Compute Q * K^T 
void q_mult_k_transposed(packed_32_fixed_t Q[B][N][NUM_VECS], packed_32_fixed_t K[B][N][NUM_VECS], tile_fixed_t out[B][N][NUM_TILES])
{
    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);

    Q_MULT_K_TRANSPOSED:
    for (int b = 0; b < B; ++b) {
        packed_32_fixed_t k_buf[N][NUM_VECS];
        packed_32_fixed_t q_buf[N][NUM_VECS];
#pragma HLS array_partition variable=q_buf dim=2 factor=2 cyclic        
#pragma HLS array_partition variable=k_buf dim=2 factor=2 cyclic        

        LOAD_Q_and_K:
        for (int i = 0; i < N; ++i) {
            LOAD_Q_and_K_VECS:
            for (int j = 0; j < NUM_VECS; ++j) {
#pragma HLS pipeline II=1
                // Load the Q and K vectors into buffers
                k_buf[i][j] = K[b][i][j];
                q_buf[i][j] = Q[b][i][j];
            }
        }

        Q_MULT_K_ROWS:
        for (int i = 0; i < N; ++i) {
            Q_MULT_K_VECS:
            for (int j = 0; j < NUM_TILES; j++) {
                tile_fixed_t tile_buf;
                
                Q_MULT_K_VEC:
                for (int k = 0; k < NUM_PACKED_TILE; ++k) {
#pragma HLS pipeline II=1
                    ap_fixed<32, 8> sum = 0;
                    ACCUMULATE:
                    for (int l = 0; l < NUM_VECS; ++l) {
                        // Multiply and accumulate
                        sum += (q_buf[i][l] * k_buf[j * NUM_PACKED_TILE + k][l]).reduce_add();
                    }
                    // Scale the result and store it in the tile buffer
                    tile_buf[k] = sum * scale;
                }
                // Output the tile for the next stage
                out[b][i][j] = tile_buf;
            }
        }
    }
}

// Find the maximum value in each row of the input tensor
void find_max(tile_fixed_t in[B][N][NUM_TILES], tile_fixed_t out[B][N][NUM_TILES]) {
    FIND_MAX:
    for (int b = 0; b < B; b++) {
        FIND_MAX_ROWS:
        for (int i = 0; i < N; i++) {
            tile_fixed_t row_buf[NUM_TILES];
            fixed_t max_buf[NUM_TILES];

#pragma HLS array_partition variable=row_buf dim=1 factor=2 cyclic
#pragma HLS array_partition variable=max_buf dim=1 factor=2 cyclic
             
            FIND_MAX_TILES:
            for (int j = 0; j < NUM_TILES; j++) {
#pragma HLS pipeline II=1
                // Read the tile into a buffer
                tile_fixed_t tile_buf = in[b][i][j];

                // Find the maximum value in the tile
                fixed_t tile_max = tile_buf[0];
                FIND_MAX_TILE:
                for (int k = 1; k < NUM_PACKED_TILE; ++k) {
                    if (tile_buf[k] > tile_max)
                        tile_max = tile_buf[k];
                }

                // Store the maximum value and the tile buffer
                max_buf[j] = tile_max;
                row_buf[j] = tile_buf;
            }

            fixed_t max_val = max_buf[0];
            FIND_MAX_VAL:
            for (int j = 1; j < NUM_TILES; j++) {
#pragma HLS unroll
                // Find the maximum value in the row
                if (max_buf[j] > max_val)
                    max_val = max_buf[j];
            }

            SUBSTRACT:
            for (int j = 0; j < NUM_TILES; j++) {
#pragma HLS pipeline II=1
                // Subtract the maximum value from each vector
                out[b][i][j] = row_buf[j] - max_val;
            }
        }
    }
}

// Apply softmax to the input tensor
void softmax(tile_fixed_t in[B][N][NUM_TILES], tile_fixed_t out[B][N][NUM_TILES])
{
    SOFTMAX:
    for (int b = 0; b < B; ++b) {
        SOFTMAX_ROWS:
        for (int i = 0; i < N; ++i) {
            ap_fixed<32, 8> sum = 0;
            tile_fixed_t row_buf[NUM_TILES];
#pragma HLS array_partition variable=row_buf dim=1 factor=2 cyclic

            EXP_TILES:
            for (int j = 0; j < NUM_TILES; j++) {
#pragma HLS pipeline II=1
                // Exp
                tile_fixed_t tile_buf = in[b][i][j];
                row_buf[j] = exp_vector(tile_buf);
                
                // Accumulate
                sum += row_buf[j].reduce_add();
            }

            DIV_TILES:
            for (int j = 0; j < NUM_TILES; j++) {
#pragma HLS pipeline II=1
                DIV_TILE:
                for (int k = 0; k < NUM_PACKED_TILE; ++k) {
                    row_buf[j][k] = ap_fixed<32, 8>(row_buf[j][k]) / sum;
                }
                // Output
                out[b][i][j] = row_buf[j];
            }
        }
    }
}

// Compute the attention * V
void attention_mult_v(tile_fixed_t in[B][N][NUM_TILES], tile_fixed_t V[B][dv][NUM_TILES], fixed_t Output[B][N][dv])
{
    ATTENTION_MULT_V:
    for (int b = 0; b < B; ++b) {
        tile_fixed_t v_buf[dv][NUM_TILES];
#pragma HLS array_partition variable=v_buf dim=2 factor=2 cyclic

        LOAD_BUFFERS:
        for (int i = 0; i < dv; ++i) {
            LOAD_V:
            for (int j = 0; j < NUM_TILES; ++j) {
#pragma HLS pipeline II=1
                // Load the V tile into a buffer
                v_buf[i][j] = V[b][i][j];
            }
        }

        ATTENTION_MULT_V_ROWS:
        for (int i = 0; i < N; ++i) {
            tile_fixed_t data_buf[NUM_TILES];
#pragma HLS array_partition variable=data_buf dim=1 factor=2 cyclic

            LOAD_DATA:
            for (int j = 0; j < NUM_TILES; ++j) {
#pragma HLS pipeline II=1
                // Load the data into a row buffer
                data_buf[j] = in[b][i][j];
            } 

            ATTENTION_MULT_V_TILES:
            for (int j = 0; j < dv; ++j) {
                ap_fixed<32,8> sum = 0;
#pragma HLS pipeline II=1
                ATTENTION_MULT_V_TILE:
                for(int k = 0; k < NUM_TILES; ++k) {
                    // Multiply and accumulate
                    sum += (data_buf[k] * v_buf[j][k]).reduce_add();
                }
                // Output the result of the output tensor
                Output[b][i][j] = sum;
            }
        }
    }
}

void compute_attention_HLS(packed_32_fixed_t Q[B][N][NUM_VECS], packed_32_fixed_t K[B][N][NUM_VECS], tile_fixed_t V[B][dv][NUM_TILES], fixed_t Output[B][N][dv])
{
#pragma HLS interface mode=axis port=Q
#pragma HLS interface mode=axis port=K
#pragma HLS interface mode=axis port=V
#pragma HLS interface mode=axis port=Output

#pragma HLS interface s_axilite port=return

#pragma HLS dataflow 

    // Array streams
    tile_fixed_t mult_result[B][N][NUM_TILES];
    tile_fixed_t max_result[B][N][NUM_TILES];
    tile_fixed_t softmax_result[B][N][NUM_TILES];

#pragma HLS stream variable=mult_result depth=4
#pragma HLS stream variable=max_result depth=4
#pragma HLS stream variable=softmax_result depth=256

    // Compute Q * K^T
    q_mult_k_transposed(Q, K, mult_result);

    // Apply softmax
    find_max(mult_result, max_result);
    softmax(max_result, softmax_result);

    // Compute Attention * V
    attention_mult_v(softmax_result, V, Output);
}