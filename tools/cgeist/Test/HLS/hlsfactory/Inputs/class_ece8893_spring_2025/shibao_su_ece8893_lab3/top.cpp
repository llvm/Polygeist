#include "dcl.h"

// Define FIFO type for local_C elements
typedef hls::stream<data_t> fifo_t;

// Task 1: Load data from DRAM to BRAM
void load_data(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1],
               data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1],
               data_t local_values_A[N * M], int local_column_indices_A[N * M], int local_row_ptr_A[N + 1],
               data_t local_values_B[M * K], int local_row_indices_B[M * K], int local_col_ptr_B[M + 1]) {
#pragma HLS INLINE off
//brust read 
    for (int i = 0; i < N * M; i++) {
#pragma HLS PIPELINE II =1
        local_values_A[i] = values_A[i];
        local_column_indices_A[i] = column_indices_A[i];
    }
    for (int i = 0; i < N + 1; i++) {
#pragma HLS PIPELINE II =1
        local_row_ptr_A[i] = row_ptr_A[i];
    }
    for (int i = 0; i < M * K; i++) {
#pragma HLS PIPELINE II =1
        local_values_B[i] = values_B[i];
        local_row_indices_B[i] = row_indices_B[i];
    }
    for (int i = 0; i < M + 1; i++) {
#pragma HLS PIPELINE II =1
        local_col_ptr_B[i] = col_ptr_B[i];
    }
}

// Task 2: Perform sparse matrix multiplication
void compute(data_t local_values_A[N * M], int local_column_indices_A[N * M], int local_row_ptr_A[N + 1],
             data_t local_values_B[M * K], int local_row_indices_B[M * K], int local_col_ptr_B[M + 1],
             fifo_t &fifo_C) {
#pragma HLS INLINE off
    data_t local_C[N][K];

   
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
#pragma HLS PIPELINE II = 1
            local_C[i][j] = 0;
        }
    }

    // Perform computation
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II =1
        for (int idx_A = local_row_ptr_A[i]; idx_A < local_row_ptr_A[i + 1]; idx_A++) {
#pragma HLS UNROLL factor = 16
            int k = local_column_indices_A[idx_A];
            data_t value_A = local_values_A[idx_A];

            for (int idx_B = local_col_ptr_B[k]; idx_B < local_col_ptr_B[k + 1]; idx_B++) {
#pragma HLS UNROLL factor = 16
                int j = local_row_indices_B[idx_B];
                data_t value_B = local_values_B[idx_B];

                // Accumulate the product into local_C[i][j]
                local_C[i][j] += value_A * value_B;
            }
        }
    }

    // Write local_C to FIFO
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
#pragma HLS PIPELINE II = 1
            fifo_C.write(local_C[i][j]);
        }
    }
}

// Task 3: Write results back to DRAM
void store_result(fifo_t &fifo_C, data_t C[N][K]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
#pragma HLS PIPELINE II = 1
            C[i][j] = fifo_C.read();
        }
    }
}

// Top function 
void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1],
                                data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1],
                                data_t C[N][K]) {
#pragma HLS INTERFACE m_axi port=values_A offset=slave bundle=mem1 max_read_burst_length=16 num_read_outstanding=16 channel=0
#pragma HLS INTERFACE m_axi port=column_indices_A offset=slave max_read_burst_length=16 num_read_outstanding=16 bundle=mem1 channel=0
#pragma HLS INTERFACE m_axi port=row_ptr_A offset=slave bundle=mem1 max_read_burst_length=16 num_read_outstanding=16 channel=0
#pragma HLS INTERFACE m_axi port=values_B offset=slave bundle=mem2 max_read_burst_length=16 num_read_outstanding=16 channel=1 
#pragma HLS INTERFACE m_axi port=row_indices_B offset=slave max_read_burst_length=16 num_read_outstanding=16 bundle=mem2 channel=1
#pragma HLS INTERFACE m_axi port=col_ptr_B offset=slave max_read_burst_length=16 num_read_outstanding=16 bundle=mem2 channel=1
#pragma HLS INTERFACE m_axi port=C offset=slave max_read_burst_length=16 num_read_outstanding=16 bundle=mem3 channel=2
#pragma HLS INTERFACE s_axilite port=return

    // Local BRAM arrays
    data_t local_values_A[N * M];
    int local_column_indices_A[N * M];
    int local_row_ptr_A[N + 1];
    data_t local_values_B[M * K];
    int local_row_indices_B[M * K];
    int local_col_ptr_B[M + 1];

    // FIFO for passing local_C between tasks
    fifo_t fifo_C;
    #pragma HLS STREAM variable=fifo_C depth = 256 // 1024 is too much, 256 produce the best result

    // Enable TLP
#pragma HLS DATAFLOW
    load_data(values_A, column_indices_A, row_ptr_A, values_B, row_indices_B, col_ptr_B,
              local_values_A, local_column_indices_A, local_row_ptr_A, local_values_B, local_row_indices_B, local_col_ptr_B);
    compute(local_values_A, local_column_indices_A, local_row_ptr_A, local_values_B, local_row_indices_B, local_col_ptr_B, fifo_C);
    store_result(fifo_C, C);
}
