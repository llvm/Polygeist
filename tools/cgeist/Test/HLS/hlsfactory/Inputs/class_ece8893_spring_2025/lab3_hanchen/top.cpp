#include "dcl.h"
#include <hls_stream.h>

#define MAX_NNZ_PER_COL 64

struct NonZeroA {
    int row;
    int col;
    data_t val;
    bool last;
};

struct NonZeroC {
    int row;
    int col;
    data_t val;
    bool last;
};

// Unified Function to Perform Entire Process
void sparse_matrix_multiply_HLS(
    data_t A_values[N*M],
    int A_col_indices[N*M],
    int A_row_ptr[N+1],
    data_t B_values[M*K],
    int B_row_indices[M*K],
    int B_col_ptr[M+1],
    data_t C[N][K]
) {
    #pragma HLS INTERFACE m_axi port=A_values offset=slave bundle=mem1
    #pragma HLS INTERFACE m_axi port=A_col_indices offset=slave bundle=mem1
    #pragma HLS INTERFACE m_axi port=A_row_ptr offset=slave bundle=mem1

    #pragma HLS INTERFACE m_axi port=B_values offset=slave bundle=mem2
    #pragma HLS INTERFACE m_axi port=B_row_indices offset=slave bundle=mem2
    #pragma HLS INTERFACE m_axi port=B_col_ptr offset=slave bundle=mem2

    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=mem3
    #pragma HLS INTERFACE s_axilite port=return

    static data_t localB[M][MAX_NNZ_PER_COL];
    static int localRowIdx[M][MAX_NNZ_PER_COL];
    static int localLen[M];

    #pragma HLS bind_storage variable=localB type=ram_1p impl=bram
    #pragma HLS bind_storage variable=localRowIdx type=ram_1p impl=bram

    int colPointers[M+1];
    for (int i = 0; i <= M; i++) {
        #pragma HLS PIPELINE II=1
        colPointers[i] = B_col_ptr[i];
    }

    for (int i = 0; i < M; i++) {
        #pragma HLS PIPELINE II=1
        localLen[i] = colPointers[i+1] - colPointers[i];
        for (int j = 0; j < localLen[i]; j++) {
            localB[i][j] = B_values[colPointers[i] + j];
            localRowIdx[i][j] = B_row_indices[colPointers[i] + j];
        }
    }

    int rowPointers[N+1];
    for (int i = 0; i <= N; i++) {
        #pragma HLS PIPELINE II=1
        rowPointers[i] = A_row_ptr[i];
    }

    static data_t accum[K];
    #pragma HLS bind_storage variable=accum type=ram_2p impl=bram

    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < K; j++) {
            accum[j] = 0;
        }

        for (int idx = rowPointers[i]; idx < rowPointers[i+1]; idx++) {
            int col = A_col_indices[idx];
            data_t aval = A_values[idx];
            for (int k = 0; k < localLen[col]; k++) {
                int rowB = localRowIdx[col][k];
                data_t bval = localB[col][k];
                accum[rowB] += aval * bval;
            }
        }

        for (int j = 0; j < K; j++) {
            #pragma HLS PIPELINE II=1
            C[i][j] = accum[j];
        }
    }
}
