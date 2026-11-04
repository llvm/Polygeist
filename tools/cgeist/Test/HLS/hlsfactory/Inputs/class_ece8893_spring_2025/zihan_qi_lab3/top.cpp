#include "dcl.h"


// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(
    data_t values_A[non_zero], int column_indices_A[non_zero], vec16_int_t row_ptr_A[N / 16], 
    data_t values_B[non_zero], int row_indices_B[non_zero], vec16_int_t col_ptr_B[M / 16], vec32_t C[N][K / 32])
{
#pragma HLS interface m_axi port=values_A offset=slave bundle=mem1
#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem2
#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem3

#pragma HLS interface m_axi port=values_B offset=slave bundle=mem4
#pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem4
#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem6

#pragma HLS interface m_axi port=C offset=slave bundle=mem7

#pragma HLS interface s_axilite port=return
    data_t local_value_A[non_zero];
    int local_column_indices_A[non_zero];
    data_t local_value_B[non_zero];
    int local_row_indices_B[non_zero];

    vec32_t C_buf_wide[N][K / 32];

    int local_row_ptr_A[N + 1];
    local_row_ptr_A[0] = 0;
    int local_col_ptr_B[M + 1];
    local_col_ptr_B[0] = 0;
#pragma HLS array_partition variable=local_row_ptr_A complete
#pragma HLS array_partition variable=local_col_ptr_B complete
#pragma HLS array_partition variable=local_value_B dim=1 factor=f cyclic
#pragma HLS array_partition variable=local_row_indices_B dim=1 factor=f cyclic
#pragma HLS array_partition variable=local_value_A dim=1 factor=f cyclic
#pragma HLS array_partition variable=local_column_indices_A dim=1 factor=f cyclic
#pragma HLS array_partition variable=C_buf_wide complete dim=2

//#pragma HLS dataflow

    for (int i = 0; i < N / 16; i++) {
        #pragma HLS pipeline II=1
        for (int j = 0; j < 16; j++) {
            #pragma HLS unroll skip_exit_check
            local_row_ptr_A[i * 16 + j + 1] = row_ptr_A[i][j];
        }
    }
    for (int i = 0; i < M / 16; i++) {
        #pragma HLS pipeline II=1
        for (int j = 0; j < 16; j++) {
            #pragma HLS unroll skip_exit_check
            local_col_ptr_B[i * 16 + j + 1] = col_ptr_B[i][j];
        }
    }

    for (int i = 0; i < non_zero; i++) {
        #pragma HLS pipeline II=1
        local_value_A[i] = values_A[i];
    }
    for (int i = 0; i < non_zero; i++) {
        #pragma HLS pipeline II=1
        local_column_indices_A[i] = column_indices_A[i];
    }

    for (int i = 0; i < non_zero; i++) {
        #pragma HLS pipeline II=1
        local_value_B[i] = values_B[i];
    }
    for (int i = 0; i < non_zero; i++) {
        #pragma HLS pipeline II=1
        local_row_indices_B[i] = row_indices_B[i];
    }

    //hls::stream<int, FIFO_DEPTH> myFIFO;
    hls::stream<vec32_t, 8> C_stream[N];
    // //Perform Sparse x Sparse Multiplication
    for (int i = 0; i < N; i++) {
        data_t acc[K] = {0};
        #pragma HLS array_partition variable=acc complete
        for (int idx_A = local_row_ptr_A[i]; idx_A < local_row_ptr_A[i + 1]; idx_A++) {
            data_t acc_temp[K] = {0};
            #pragma HLS array_partition variable=acc_temp complete
            //#pragma HLS unroll skip_exit_check
            //#pragma HLS pipeline II=1
            int k = local_column_indices_A[idx_A]; // Column index of A
            data_t value_A = local_value_A[idx_A];

            // Iterate over columns of B corresponding to row k
            for (int idx_B = local_col_ptr_B[k]; idx_B < local_col_ptr_B[k + 1]; idx_B++) {
                #pragma HLS pipeline II=1
                #pragma HLS unroll factor=2
                int j = local_row_indices_B[idx_B]; // Row index of B
                data_t value_B = local_value_B[idx_B];

                // Accumulate the product into C[i][j]
                acc_temp[j] = value_A * value_B;
            }

            for (int j = 0; j < K;j++){
                #pragma HLS unroll skip_exit_check
                acc[j] += acc_temp[j];
            }
        }
        for (int j = 0; j < K / 32; j++) {
            #pragma HLS pipeline II=1
            vec32_t out;
            for (int k = 0; k < 32; k++) {
                #pragma HLS unroll skip_exit_check
                out[k] = acc[j * 32 + k];
            }
            C_stream[i].write(out);
        }

    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K / 32; j++) {
            vec32_t val = C_stream[i].read();
            C[i][j] = val;
        }
    }

}
