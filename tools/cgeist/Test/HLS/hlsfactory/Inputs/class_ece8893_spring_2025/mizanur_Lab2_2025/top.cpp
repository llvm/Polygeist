#include "dcl.h"

void softmax_HLS(fixed_t matrix[B][N][N], fixed_t matrix_out[B][N][N]) {
    //#pragma HLS INLINE 
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            #pragma HLS pipeline
            ap_fixed<32, 8> max_val = matrix[b][i][0];

            // Find max value
            for (int j = 1; j < N; ++j) {
                //#pragma HLS unroll
                if (matrix[b][i][j] > max_val) {
                    max_val = matrix[b][i][j];
                }
            }

            // Compute approximate softmax
            ap_fixed<32, 8> sum = 0;
            for (int j = 0; j < N; ++j) {
                #pragma HLS unroll
                ap_fixed<32, 8> x = matrix[b][i][j] - max_val;
                matrix_out[b][i][j] = 1 + x + ap_fixed<32, 8>(0.5) * x * x  ;  // Approximate exp(x)
                sum += matrix_out[b][i][j];
            }

            // Normalize
            ap_fixed<32, 8> sum_inv =  ap_fixed<32, 8>(1.0) / sum;
            for (int j = 0; j < N; ++j) {
                #pragma HLS unroll
                matrix_out[b][i][j] *= sum_inv;
            }
        }
    }
}


void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {


    #pragma HLS interface m_axi port=Q offset=slave bundle=mem1
    #pragma HLS interface m_axi port=K offset=slave bundle=mem1
    #pragma HLS interface m_axi port=V offset=slave bundle=mem3
    #pragma HLS interface m_axi port=Output offset=slave bundle=mem2

    #pragma HLS interface s_axilite port=return

    #pragma HLS DATAFLOW

    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);



    fixed_t attention_scores[B][N][N];
    #pragma HLS ARRAY_PARTITION variable=attention_scores dim=3 factor=4 cyclic

    fixed_t attention_prob[B][N][N];
    #pragma HLS ARRAY_PARTITION variable=attention_prob dim=3 factor=4 cyclic

    // Step 1: Compute Attention Scores
    for (int b = 0; b < B; ++b) {

        fixed_t mat_Q[N][dk], mat_K[N][dk];
        for (int i = 0; i < N; ++i) {
            //#pragma HLS UNROLL 
            for (int k = 0; k < dk; ++k) {
            mat_K[i][k] = K[b][i][k];
            mat_Q[i][k] = Q[b][i][k];   
            }
        }



        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                #pragma HLS PIPELINE

                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < dk; ++k) {
                    #pragma HLS UNROLL factor=8
                    sum += mat_Q[i][k] * mat_K[j][k];
                }
                attention_scores[b][i][j] = sum * scale;
            }
        }
    }

    // Step 2: Apply Softmax
    softmax_HLS(attention_scores, attention_prob);


    // Step 3: Compute Weighted Sum with V
    for (int b = 0; b < B; ++b) {


        fixed_t mat_V[N][dv];
        for (int i = 0; i < N; ++i) {
            //#pragma HLS UNROLL 
            for (int k = 0; k < dv; ++k) {
            mat_V[i][k] = V[b][i][k];
      
            }
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j) {
                #pragma HLS PIPELINE

                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < N; ++k) {
                    #pragma HLS UNROLL factor=8
                    sum += attention_prob[b][i][k] * mat_V[k][j];
                }
                Output[b][i][j] = sum;
            }
        }
    }
}

