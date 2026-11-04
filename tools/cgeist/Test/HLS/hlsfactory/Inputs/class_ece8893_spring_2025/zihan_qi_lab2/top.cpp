#include "dcl.h"

void softmax_HLS(fixed_t matrix[N]) {
            // Find the largest number in each row
            //ap_fixed<32, 8> max_val = matrix[0];
            // for (int j = 0; j < N; ++j) {
            //     #pragma HLS unroll factor=4
            //     //#pragma HLS unroll
            //     if (matrix[j] > max_val) {
            //         max_val = matrix[j];
            //     }
            // }

            ap_fixed<32, 8> sum = 0;
            for (int j = 0; j < N; ++j) {
                #pragma HLS unroll factor=16
                //#pragma HLS unroll
                // Change: simplify or approximate softmax (But don't find improvement in LightningSim)
                //matrix[j] = hls::exp(matrix[j] - max_val);
                matrix[j] = hls::exp(matrix[j]);
                //ap_fixed<32, 8> x = matrix[j] - max_val;
                // Change: Fourth order
                //matrix[j] = 1 + x + ap_fixed<32, 8>(0.5) * x * x + ap_fixed<32, 8>(1.0 / 6.0) * x * x * x + ap_fixed<32, 8>(1.0 / 24.0) * x * x * x * x;
                // Change: Third order
                //matrix[j] = 1 + x + ap_fixed<32, 8>(0.5) * x * x + ap_fixed<32, 8>(1.0 / 6.0) * x * x * x;
                sum += matrix[j];
            }

            for (int j = 0; j < N; ++j) {
                #pragma HLS unroll factor=16
                matrix[j] /= sum;
            }
    
}


void compute_attention_HLS(vec32_t Q[B][N][dk/32], vec32_t K[B][N][dk/32], vec32_t V[B][N][dv/32], vec32_t Output[B][N][dv/32]) {

// Change: the bundle there to use different AXI port for data transferring 
#pragma HLS interface m_axi port=Q offset=slave max_widen_bitwidth=512 bundle=Q
#pragma HLS interface m_axi port=K offset=slave max_widen_bitwidth=512 bundle=K
#pragma HLS interface m_axi port=V offset=slave max_widen_bitwidth=512 bundle=V
#pragma HLS interface m_axi port=Output offset=slave max_widen_bitwidth=512 bundle=Output
#pragma HLS interface s_axilite port=return

	// Define local buffers (in BRAM)
    vec32_t Q_local[dk / 32];
    vec32_t K_local[N][dk / 32];
    fixed_t attention[N];
    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);
    vec32_t V_local[N][dv / 32];

    #pragma HLS array_partition variable=Q_local complete
    //#pragma HLS array_partition variable=K_local cyclic factor=2 dim=1
    #pragma HLS array_partition variable=K_local complete dim=2

    #pragma HLS array_partition variable=V_local complete dim=2
    //#pragma HLS array_partition variable=V_local cyclic factor=2 dim=1
    #pragma HLS array_partition variable=attention complete

    for (int b = 0; b < B; b++){
        // Every batch
        //#pragma HLS dataflow
        // transfer block K

        for (int j = 0; j < N;j++){
            for (int k = 0; k < dk / 32;k++){
                #pragma HLS pipeline II=1
                K_local[j][k] = K[b][j][k];
            }
            for (int v = 0; v < dv / 32;v++){
                #pragma HLS pipeline II=1
                V_local[j][v] = V[b][j][v];
            }
        }
        

        
        for (int i = 0; i < N; i++){
            vec32_t Output_local[dv / 32] = {};
            static ap_fixed<32, 8> Output_temp[dv / 32][32] = {0};
            
//#pragma HLS bind_storage variable = Output_temp type = RAM_2P impl = BRAM
#pragma HLS array_partition variable = Output_temp complete

#pragma HLS array_partition variable = Output_local complete
            // each row i in Q, return row i for Output

            // transfer block Q

            for (int k = 0; k < dk / 32; k++){
                #pragma HLS pipeline II=1
                Q_local[k] = Q[b][i][k];
            }
            
            
            // Compute Q_block * K^T_block
            for (int j = 0; j < N;j++){
                //#pragma HLS unroll factor=2
                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < dk / 32; k++){
                    #pragma HLS unroll skip_exit_check
                    for (int t = 0; t < 32;t++){
                        #pragma HLS unroll skip_exit_check
                        sum+=Q_local[k][t] * K_local[j][k][t];
                    }
                }
                attention[j] = sum * scale;
            }
            // // print attention
            // //fixed_t attention[4][1][N];
            // for (int bi = 0; bi < 4;bi++){
            //     for (int j = 0; j < N; j++)
            //     {
            //         printf("TEST: attention[%d][%d][%d]: %.8f; \n", 4 * b + bi, i, j, attention[bi][0][j].to_float());
            //     }
            // }


            
            // attention block softmax
            softmax_HLS(attention);
            

            // compute attention * V
            for (int j = 0; j < N; j++){
                //#pragma HLS unroll factor=2
                for (int k = 0; k < dk / 32;k++){
                    #pragma HLS unroll skip_exit_check
                    for (int t = 0; t < 32;t++){
                        #pragma HLS unroll skip_exit_check
                        Output_temp[k][t] += attention[j] * V_local[j][k][t];
                    }
                }
            }

            for (int k = 0; k < dk / 32;k++){
                #pragma HLS unroll skip_exit_check
                for (int t = 0; t < 32;t++){
                    #pragma HLS unroll skip_exit_check
                    Output_local[k][t] = Output_temp[k][t];
                    Output_temp[k][t] = 0;
                }
                
            }

            // transfer back to dram
            for (int k = 0; k < dk / 32; k++)
            {
                #pragma HLS pipeline II=1
                Output[b][i][k] = Output_local[k];
            }
            

        }
    }
}