#include "dcl.h"
// #include <hls_vector.h>
// #include <hls_stream.h>

// typedef hls::vector<fixed_t, 32> fixed_vector_t;
// typedef hls::stream<fixed_t> fixed_stream_t;
// Declaring the stream


// Pipeline the softmax function 
// void softmax_HLS(fixed_t matrix[N]) {
void softmax_HLS(hls::stream<fixed_t> &attention_Stream, hls::stream<fixed_t> &softmax_stream) {
    // for (int b = 0; b < B; ++b) {
        // for (int i = 0; i < N; ++i) {
            // ap_fixed<32, 8> max_val = matrix[0];
            fixed_t matrix[N];
            #pragma HLS array_partition variable=matrix type=cyclic factor=TF 
            
            ap_fixed<32, 8> max_val;
            attention_Stream >> matrix[0];
            max_val = matrix[0];

            // Implement tiling
            attention_read:for (int j = 1; j < N; ++j) {
                attention_Stream >> matrix[j];
                if (matrix[j] > max_val) {
                    max_val = matrix[j];
                }
            }

            // Partition array
            ap_fixed<32, 8> sum = 0;

            // Tile + Unroll
            softmax_sum_exp:for (int j = 0; j < N; j += TF) {
                for (int jj=0; jj < TF; ++jj) {
                // #pragma HLS unroll
                    if(j + jj < N) {
                        matrix[j + jj] = hls::exp(matrix[j + jj] - max_val);
                        sum += matrix[j + jj];
                    }
                }
            }

            // Tile + Unroll
            softmax_norm:for (int j = 0; j < N; j += TF) {
                for (int jj=0; jj < TF; ++jj) {
                #pragma HLS unroll
                    if(j + jj < N) {
                        matrix[j + jj] /= sum;
                    }
                }
            }

            softmax_write:for (int i = 0; i < N; ++i)
                softmax_stream << matrix[i];

        }
//     }
// }

// void kernel_compute_acc(fixed_t Q_local[dk], fixed_t K_local[N][dk], fixed_t attention[N]){
void kernel_compute_acc(fixed_t Q_local[dk], fixed_t K_local[N][dk], hls::stream<fixed_t> &attention_stream){
    // #pragma HLS pipeline
    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);
    
    // for (int b = 0; b < B; ++b) {
    //     for (int i = 0; i < N; ++i) {
            kernel_main:for (int j = 0; j < N; ++j) {
                // Partition
                ap_fixed<32, 8> sum = 0;

                // Tile + Unroll
                kernel_acc:for (int k = 0; k < dk; k += TF) {
                    for (int kk = 0; kk < TF; ++kk) {
                        #pragma HLS unroll
                        // sum += Q[b][i][kk] * K[b][j][kk];
                        sum += Q_local[k + kk] * K_local[k + kk];
                    }
                }
                // attention[j] = sum * scale;
                attention_stream << sum * scale;
            }
        }
//     }
// }

// void kernel_attention_V(fixed_t V_local[N][dv], fixed_t attention[N], fixed_t Output_local[dv]){
void kernel_attention_V(fixed_t V_local[N][dv], hls::stream<fixed_t> &softmax_stream, fixed_t Output_local[dv]){
    // for (int b = 0; b < B; ++b) {
    //     for (int i = 0; i < N; ++i) {
            fixed_t attention[N];
            #pragma HLS array_partition variable=attention type=cyclic factor=TF 
            
            softmax_read:for (int i=0; i<N; ++i)
                softmax_stream >> attention[i];

            vector_main:for (int j = 0; j < dv; ++j) {

                // Partition
                ap_fixed<32, 8> sum = 0;

                // Tile + Unroll
                vector_mul_acc:for (int k = 0; k < N; k += TF) {
                    for (int kk = 0; kk < TF; ++kk) {
                        #pragma HLS unroll
                        // sum += Q[b][i][kk] * K[b][j][kk];
                        // sum += Q_local[b][i][kk] * K_local[b][j][kk];
                        if (k + kk < N)
                            sum += attention[k + kk] * V_local[k + kk][j];
                    }
                }
                
                Output_local[j] = sum;
            }
        }

void write_output(fixed_t Output_lcl[dv], fixed_t Output[dv]){
    output_write:for (int k = 0; k < dv; k += BurstSize){
        // Burst Read
        for (int k1 = 0; k1 < BurstSize; ++k1){
        //     Output[i][j][k2].range(0 + l*16, 31 + l*16) = Output_local[i][j][32*k2 + l];
            Output[k + k1] = Output_lcl[k + k1];
        }
    }

}

void dflow_fn(fixed_t Q_lcl[dk], fixed_t K_lcl[N][dk], fixed_t V_lcl[N][dv], fixed_t Output[dv]){
    #pragma HLS dataflow
    hls::stream<fixed_t, N+1> attention_stream("attention_stream");
    hls::stream<fixed_t, N+1> softmax_stream("softmax_stream");
    
    fixed_t Output_lcl[dv];
    #pragma HLS array_partition variable=Output_lcl type=cyclic factor=TF dim=1 

            // Compute Q * K^T
            // // kernel_compute_acc(Q_local[b][i], K_local[b], attention[b][i]);
            kernel_compute_acc(Q_lcl, K_lcl, attention_stream);

            // // Apply softmax
            // // softmax_HLS(attention[b][i]);
            softmax_HLS(attention_stream, softmax_stream);

            // // Compute Attention * V
            // // kernel_attention_V(V_local[b], attention[b][i], Output_local[b][i]);
            kernel_attention_V(V_lcl, softmax_stream, Output_lcl);
            
            // Copying Back Output
            write_output(Output_lcl, Output);

}
//     }

// }

// void load_via_vector_HLS(fixed_stream_t &Q_stream, fixed_stream_t &K_stream, fixed_stream_t &V_stream) {
//     // Burst Read using the HLS Vector interface
// }
void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
#pragma HLS interface m_axi port=Q offset=slave bundle=mem2 max_read_burst_length=BurstSize max_write_burst_length=BurstSize
#pragma HLS interface m_axi port=K offset=slave bundle=mem1 max_read_burst_length=BurstSize max_write_burst_length=BurstSize
#pragma HLS interface m_axi port=V offset=slave bundle=mem1 max_read_burst_length=BurstSize max_write_burst_length=BurstSize
#pragma HLS interface m_axi port=Output offset=slave bundle=mem2 max_read_burst_length=BurstSize max_write_burst_length=BurstSize

#pragma HLS interface s_axilite port=return

// #pragma HLS dataflow

// Local Variables
// fixed_t Q_local[B][N][dk], K_local[B][N][dk], V_local[B][N][dv], Output_local[B][N][dv];
fixed_t Q_lcl[dk], K_lcl[N][dk], V_lcl[N][dv];
// fixed_vector_t Q_lcl[dk], K_lcl[N][dk], V_lcl[N][dv], Output_lcl[dv];

// fixed_t attention[B][N][N];

// Partition V in the second dimension
#pragma HLS array_partition variable=V_lcl type=cyclic factor=TF dim=1 
// Partition K, Q, Output in the first dimension
#pragma HLS array_partition variable=Q_lcl type=cyclic factor=TF dim=1 
#pragma HLS array_partition variable=K_lcl type=cyclic factor=TF dim=2 
// #pragma HLS array_partition variable=V_local type=cyclic factor=4 dim=3 
// #pragma HLS array_partition variable=attention type=cyclic factor=TF dim=3 


    // #pragma HLS dataflow

    // Read and write into local Q, K, V and Output buffers
    // for(int i=0; i<B; ++i) {
    //     for (int j=0; j<N; ++j){
    //         for (int k = 0; k < dk; k += BurstSize){
    //             // Burst Read
    //             for (int k1 = 0; k1 < BurstSize; ++k1){
    //             #pragma HLS pipeline
    //             //     Q_local[i][j][32*k1 + l] = Q[i][j][k1].range(0 + l*16, 31 + l*16);
    //             //     K_local[i][j][32*k1 + l] = K[i][j][k1].range(0 + l*16, 31 + l*16);
    //                 Q_local[i][j][k + k1] = Q[i][j][k + k1];
    //             }

                
    //         }
            
    //     }
    // }

    // for(int i=0; i<B; ++i) {
    //     for (int j=0; j<N; ++j){
    //         for (int k = 0; k < dk; k += BurstSize){
    //             // Burst Read
    //             for (int k1 = 0; k1 < BurstSize; ++k1) {
    //             #pragma HLS pipeline
    //                 K_local[i][j][k + k1] = K[i][j][k + k1];

    //             }
    //         }
    //     }
    // }
    
    // for(int i=0; i<B; ++i) {
    //     for (int j=0; j<N; ++j){
    //         for (int k = 0; k < dv; k += BurstSize){
    //             // Burst Read
    //             for (int k1 = 0; k1 < BurstSize; ++k1){
    //             #pragma HLS pipeline
    //                 V_local[i][j][k1 + k] = V[i][j][k1 + k];
    //             }
    //             // V_local[i][j][k2] = V[i][j][k2];
    //         }
    //     }
    // }



    main:for (int b = 0; b < B; ++b) {

        // Read K from memory 
        read_k:for (int j=0; j<N; ++j){
            for (int k = 0; k < dk; k += BurstSize){
                // Burst Read
                for (int k1 = 0; k1 < BurstSize; ++k1) {
                #pragma HLS pipeline
                    K_lcl[j][k + k1] = K[b][j][k + k1];

                }
            }
        }

        // Read V from memory
        read_v:for (int j=0; j<N; ++j){
            for (int k = 0; k < dv; k += BurstSize){
                // Burst Read
                for (int k1 = 0; k1 < BurstSize; ++k1){
                #pragma HLS pipeline
                    V_lcl[j][k1 + k] = V[b][j][k1 + k];
                }
                // V_local[i][j][k2] = V[i][j][k2];
            }
        }

        dflow_main:for (int i = 0; i < N; ++i) {
            #pragma HLS dataflow

            // Read Q from memory
            q_read:for (int k = 0; k < dk; k += BurstSize){
                // Burst Read
                for (int k1 = 0; k1 < BurstSize; ++k1){
                #pragma HLS pipeline
                //     Q_local[i][j][32*k1 + l] = Q[i][j][k1].range(0 + l*16, 31 + l*16);
                //     K_local[i][j][32*k1 + l] = K[i][j][k1].range(0 + l*16, 31 + l*16);
                    Q_lcl[k + k1] = Q[b][i][k + k1];
                }

                
            }

            dflow_fn(Q_lcl, K_lcl, V_lcl, Output[b][i]);
            // Compute Q * K^T
            // // kernel_compute_acc(Q_local[b][i], K_local[b], attention[b][i]);
            // kernel_compute_acc(Q_lcl, K_lcl, attention_stream);

            // // // Apply softmax
            // // // softmax_HLS(attention[b][i]);
            // softmax_HLS(attention_stream, softmax_stream);

            // // // Compute Attention * V
            // // // kernel_attention_V(V_local[b], attention[b][i], Output_local[b][i]);
            // kernel_attention_V(V_lcl, softmax_stream, Output_lcl);
            
            // // Copying Back Output
            // output_write:for (int k = 0; k < dv; k += BurstSize){
            //     // Burst Read
            //     for (int k1 = 0; k1 < BurstSize; ++k1){
            //     //     Output[i][j][k2].range(0 + l*16, 31 + l*16) = Output_local[i][j][32*k2 + l];
            //         Output[b][i][k + k1] = Output_lcl[k + k1];
            //     }
            // }
        }
    }


    // Copying back to output
    // for(int i=0; i<B; ++i) {
    //     for (int j=0; j<N; ++j){
            
    //         for (int k = 0; k < dv; k += BurstSize){
    //             // Burst Read
    //             for (int k1 = 0; k1 < BurstSize; ++k1){
    //             //     Output[i][j][k2].range(0 + l*16, 31 + l*16) = Output_local[i][j][32*k2 + l];
    //                 Output[i][j][k + k1] = Output_local[i][j][k + k1];
    //             }
    //         }
    //     }
    // }


}