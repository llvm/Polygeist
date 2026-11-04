#include "dcl.h"

// LUT-based exponentiation approximation
fixed_t approx_exp(q_fixed_t x) {
    // Adjusted clamping for ap_fixed<8,3> range
    if (x < -4) return 0;
    if (x > 4) return 55; // exp(4) is ~54.59, upper bound

    // Improved polynomial approximation: exp(x) ≈ 1 + x + x^2/2
    return 1 + x + (x * x >> 1); 
}


// Reciprocal approximation using Newton-Raphson method
q_fixed_t approx_recip(q_fixed_t x) {
    return q_fixed_t(1.25) - (x >> 2); // Approximate 1/x using linear fit
}


void softmax_HLS(fixed_t matrix_in[B][N][N], fixed_t matrix_out[B][N][N]) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW // Enables pipeline execution of independent loops

    q_fixed_t max_vals[B][N];  
    q_fixed_t exp_sums[B][N];

    #pragma HLS ARRAY_PARTITION variable=max_vals complete dim=2
    #pragma HLS ARRAY_PARTITION variable=exp_sums complete dim=2

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            #pragma HLS PIPELINE II=1
            q_fixed_t max_val = matrix_in[b][i][0];

            for (int j = 1; j < N; ++j) {
                #pragma HLS UNROLL
                if (matrix_in[b][i][j] > max_val) {
                    max_val = matrix_in[b][i][j];
                }
            }

            max_vals[b][i] = max_val; // Store max values
        }
    }

    q_fixed_t exp_buf[B][N][N];  // Buffer for exponentials
    #pragma HLS ARRAY_PARTITION variable=exp_buf complete dim=3

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            #pragma HLS PIPELINE II=1
            q_fixed_t sum = 0;

            for (int j = 0; j < N; ++j) {
                #pragma HLS UNROLL
                q_fixed_t exp_val = approx_exp(matrix_in[b][i][j] - max_vals[b][i]);
                exp_buf[b][i][j] = exp_val;
                sum += exp_val;
            }

            exp_sums[b][i] = sum; // Store sum of exponentials
        }
    }

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            #pragma HLS PIPELINE II=1
            q_fixed_t sum_inv = approx_recip(exp_sums[b][i]);

            for (int j = 0; j < N; ++j) {
                #pragma HLS UNROLL
                matrix_out[b][i][j] = exp_buf[b][i][j] * sum_inv;
            }
        }
    }
}

void compute_attention_HLS(wide_t Q[B][N][dk / 32], wide_t K[B][N][dk / 32], wide_t V[B][N][dv / 32], wide_t Output[B][N][dv / 32]) 
{
    #pragma HLS INTERFACE m_axi port=Q offset=slave bundle=mem1 depth=512
    #pragma HLS INTERFACE m_axi port=K offset=slave bundle=mem1 depth=512
    #pragma HLS INTERFACE m_axi port=V offset=slave bundle=mem1 depth=512
    #pragma HLS INTERFACE m_axi port=Output offset=slave bundle=mem2 depth=512
    #pragma HLS INTERFACE s_axilite port=return

    fixed_t Q_buf[B][N][dk];
    fixed_t K_buf[B][N][dk];
    fixed_t V_buf[B][N][dv];

    #pragma HLS ARRAY_PARTITION variable=Q_buf cyclic factor=8 dim=3
    #pragma HLS ARRAY_PARTITION variable=K_buf cyclic factor=8 dim=3
    #pragma HLS ARRAY_PARTITION variable=V_buf cyclic factor=8 dim=3

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            #pragma HLS PIPELINE
            for (int k = 0; k < dk; k += 32) { 
                wide_t Q_wide = Q[b][i][k / 32];
                wide_t K_wide = K[b][i][k / 32];

                for (int j = 0; j < 32; ++j) {
                    Q_buf[b][i][k + j] = *((fixed_t*)&Q_wide + j);
                    K_buf[b][i][k + j] = *((fixed_t*)&K_wide + j);
                }
            }

            for (int d = 0; d < dv; d += 32) {
                wide_t V_wide = V[b][i][d / 32];

                for (int j = 0; j < 32; ++j) {
                    V_buf[b][i][d + j] = *((fixed_t*)&V_wide + j);
                }
            }
        }
    }

    fixed_t attention_buf[B][N][N];  
    #pragma HLS ARRAY_PARTITION variable=attention_buf cyclic factor=8 dim=2

    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                #pragma HLS PIPELINE
                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < dk; ++k) {
                    #pragma HLS UNROLL factor=8
                    sum += Q_buf[b][i][k] * K_buf[b][j][k];
                }
                attention_buf[b][i][j] = sum * scale;
            }
        }
    }

    // Apply softmax
    softmax_HLS(attention_buf);
    fixed_t Output_buf[B][N][dv];
    #pragma HLS ARRAY_PARTITION variable=Output_buf cyclic factor=8 dim=3

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j) {
                #pragma HLS PIPELINE
                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < N; ++k) {
                    #pragma HLS UNROLL factor=8
                    sum += attention_buf[b][i][k] * V_buf[b][k][j];
                }
                Output_buf[b][i][j] = sum;
            }
        }
    }

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            #pragma HLS PIPELINE
            for (int d = 0; d < dv; d += 32) { 
                wide_t out_wide;
                for (int j = 0; j < 32; ++j) {
                    *((fixed_t*)&out_wide + j) = Output_buf[b][i][d + j];
                }
                Output[b][i][d / 32] = out_wide;
            }
        }
    }
}