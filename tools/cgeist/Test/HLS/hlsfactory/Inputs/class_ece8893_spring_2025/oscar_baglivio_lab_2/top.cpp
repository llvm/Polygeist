#include "dcl.h"

// divisor of dk (128)
// 1, 2, 4, 8, 16, 32, 64, and 128
#define QK_FACT 128

// divisors of N (100)
// 1, 2, 4, 5, 10, 20, 25, 50, and 100
#define VAt_FACT 100

void softmax_HLS(fixed_t matrix[N]) {
  ap_fixed<32, 8> max_val = matrix[0];
SM_Max:
  for (int n = 1; n < N; ++n) {
    if (matrix[n] > max_val) {
      max_val = matrix[n];
    }
  }

SM_Exp:
  for (int n = 0; n < N; ++n) {
    //#pragma HLS unroll factor = 4 skip_exit_check
    matrix[n] = hls::exp(matrix[n] - max_val);
  }

  ap_fixed<32, 8> sum = 0;
SM_Sum:
  for (int n = 0; n < N; ++n) {
#pragma HLS unroll factor = VAt_FACT skip_exit_check
    sum += matrix[n];
  }

SM_Div:
  for (int n = 0; n < N; ++n) {
#pragma HLS unroll factor = VAt_FACT skip_exit_check
    matrix[n] /= sum;
  }
}

void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk],
                           fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
#pragma HLS interface m_axi port = Q offset = slave bundle = mem1
#pragma HLS interface m_axi port = K offset = slave bundle = mem2
#pragma HLS interface m_axi port = V offset = slave bundle = mem3
#pragma HLS interface m_axi port = Output offset = slave bundle = mem4

#pragma HLS interface s_axilite port = return
Batch:
  for (int b = 0; b < B; ++b) {
    fixed_t V_loc[N][dv];
    fixed_t K_loc[N][dk];
//     fixed_t Q_loc[N][dk];
// #pragma HLS ARRAY_PARTITION variable = Q_loc type = cyclic factor = \
//     QK_FACT dim = 2
#pragma HLS ARRAY_PARTITION variable = K_loc type = cyclic factor =            \
    QK_FACT dim = 2
#pragma HLS ARRAY_PARTITION variable = V_loc type = cyclic factor =            \
    VAt_FACT dim = 1
    //#pragma HLS ARRAY_PARTITION variable = K_loc complete dim = 2
    // V is split on the second dimension where innermost loops iterate over
    //#pragma HLS ARRAY_PARTITION variable = V_loc complete dim = 1

    memcpy(K_loc, K[b], N * dk * sizeof(fixed_t));

    memcpy(V_loc, V[b], N * dv * sizeof(fixed_t));

    // memcpy(Q_loc, Q[b], N * dk * sizeof(fixed_t));

  Query:
    for (int m = 0; m < N; ++m) {
      ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);

      fixed_t Q_loc[dk];

      fixed_t attention[N];

#pragma HLS ARRAY_PARTITION variable = Q_loc type = cyclic factor =            \
    QK_FACT dim = 1

#pragma HLS ARRAY_PARTITION variable = attention type = cyclic factor =        \
    VAt_FACT dim = 1
      //#pragma HLS ARRAY_PARTITION variable = Q_loc complete dim = 1
      //#pragma HLS ARRAY_PARTITION variable = attention complete dim = 1

    mc_Q:
      memcpy(Q_loc, Q[b][m], dk * sizeof(fixed_t));

      fixed_t Output_local[dv];

      // Compute Q * K^T

      // go through all n keys in K
    QK_Dot_Key:
      for (int n = 0; n < N; ++n) {
        // a query-key combo has been selected
        // perform a dot product on the pair
        // track a running sum
        ap_fixed<32, 8> sum = 0;
      QK_Dot_Sum:
        for (int k = 0; k < dk; ++k) {
#pragma HLS unroll factor = QK_FACT skip_exit_check
          sum += Q_loc[k] * K_loc[n][k];
        }
        attention[n] = sum * scale;
      }

      // Apply softmax
      softmax_HLS(attention);

      // Compute Attention * V
    SV_Dot_Value:
      for (int v = 0; v < dv; ++v) {
        ap_fixed<32, 8> sum = 0;
      SV_Dot_Sum:
        for (int n = 0; n < N; ++n) {
#pragma HLS unroll factor = VAt_FACT skip_exit_check
          sum += attention[n] * V_loc[n][v];
        }
        Output_local[v] = sum;
      }

    mc_Output:
      memcpy(Output[b][m], Output_local, dv * sizeof(fixed_t));
    }
  }
}