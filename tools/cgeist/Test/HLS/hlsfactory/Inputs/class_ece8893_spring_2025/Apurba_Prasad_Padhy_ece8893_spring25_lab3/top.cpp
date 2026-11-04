#include "dcl.h"
#include <ap_int.h>
#include <hls_stream.h>

#define PAR_FACTOR 24

typedef ap_fixed<16, 5> data_tx;

//--------------------------------------------------------------------------
// Load data from external memory into local buffers with aggressive partitioning.
//--------------------------------------------------------------------------
void load_data(
    data_t values_A[N * M],
    int column_indices_A[N * M],
    int row_ptr_A[N + 1],
    data_t values_B[M * K],
    int row_indices_B[M * K],
    int col_ptr_B[M + 1],
    data_tx local_values_A[N * M],
    int local_column_indices_A[N * M],
    int local_row_ptr_A[N + 1],
    data_tx local_values_B[M * K],
    int local_row_indices_B[M * K],
    int local_col_ptr_B[M + 1]
) {
    // Load matrix A nonzero values and column indices.
    LOAD_A_VALUES:
    for (int i = 0; i < N * M; i++) {
        #pragma HLS PIPELINE II=1
        local_values_A[i] = values_A[i];
        local_column_indices_A[i] = column_indices_A[i];
        local_values_B[i] = values_B[i];
        local_row_indices_B[i] = row_indices_B[i];
    }
    // Load matrix A row pointer.
    LOAD_A_ROW_PTR:
    for (int i = 0; i < N + 1; i++) {
        #pragma HLS PIPELINE II=1
        local_row_ptr_A[i] = row_ptr_A[i];
        local_col_ptr_B[i] = col_ptr_B[i];
    }
    // Load matrix B nonzero values and row indices.
    // LOAD_B_VALUES:
    // for (int i = 0; i < M * K; i++) {
    //     #pragma HLS PIPELINE II=1
        
    // }
    // // Load matrix B column pointer.
    // LOAD_B_COL_PTR:
    // for (int i = 0; i < M + 1; i++) {
    //     #pragma HLS PIPELINE II=1
        
    // }
}

//--------------------------------------------------------------------------
// Process a single row 'i' of matrix A.
//   - Initialize a fully partitioned local register array (row[]) to zero,
//   - For each nonzero in row i, accumulate products from B,
//   - Write the entire row to an output stream.
//--------------------------------------------------------------------------
void process_row(
    int i,
    data_tx local_values_A[N * M],
    int local_column_indices_A[N * M],
    int local_row_ptr_A[N + 1],
    data_tx local_values_B[M * K],
    int local_row_indices_B[M * K],
    int local_col_ptr_B[M + 1],
    hls::stream<data_tx> &row_stream
) {
    #pragma HLS INLINE   // Inline to help fuse loops into the outer parallel group

    // Local register array for current row; completely partitioned.
    data_tx row[K];
    #pragma HLS ARRAY_PARTITION variable=row complete

    // Initialize row to zero. Unroll fully.
    INIT_ROW:
    for (int j = 0; j < K; j++) {
        #pragma HLS UNROLL
        row[j] = 0;
    }

    // Accumulate nonzeros for this row.
    ROW_LOOP:
    for (int idx_A = local_row_ptr_A[i]; idx_A < local_row_ptr_A[i + 1]; idx_A++) {
        #pragma HLS PIPELINE II=1
        int k = local_column_indices_A[idx_A];
        data_tx value_A = local_values_A[idx_A];

        // Multiply with every nonzero in column k of B.
        COL_LOOP:
        for (int idx_B = local_col_ptr_B[k]; idx_B < local_col_ptr_B[k + 1]; idx_B++) {
            #pragma HLS PIPELINE II=1
            int j = local_row_indices_B[idx_B];
            row[j] += value_A * local_values_B[idx_B];
        }
    }

    // Write the computed row to the output stream.
    WRITE_ROW:
    for (int j = 0; j < K; j++) {
        #pragma HLS PIPELINE II=1
        row_stream.write(row[j]);
    }
}

//--------------------------------------------------------------------------
// Compute stage: Process rows in groups of PAR_FACTOR concurrently.
// An array of streams is used to output each row from parallel PEs.
//--------------------------------------------------------------------------
void compute(
    data_tx local_values_A[N * M],
    int local_column_indices_A[N * M],
    int local_row_ptr_A[N + 1],
    data_tx local_values_B[M * K],
    int local_row_indices_B[M * K],
    int local_col_ptr_B[M + 1],
    hls::stream<data_tx> row_stream[PAR_FACTOR]
) {
    // Process rows in groups.
    GROUP_LOOP:
    for (int i = 0; i < N; i += PAR_FACTOR) {
        // Launch PAR_FACTOR parallel row computations.
        PAR_LOOP:
        for (int p = 0; p < PAR_FACTOR; p++) {
            #pragma HLS UNROLL
            if(i + p < N)
                process_row(i + p,
                            local_values_A, local_column_indices_A, local_row_ptr_A,
                            local_values_B, local_row_indices_B, local_col_ptr_B,
                            row_stream[p]);
        }
    }
}

//--------------------------------------------------------------------------
// Store stage: Read the streamed rows (from an array of streams)
// and write them into matrix C in the correct order.
//--------------------------------------------------------------------------
void store_data(
    hls::stream<data_tx> row_stream[PAR_FACTOR],
    data_t C[N][K]
) {
    GROUP_OUT:
    for (int i = 0; i < N; i += PAR_FACTOR) {
        PAR_OUT:
        for (int p = 0; p < PAR_FACTOR; p++) {
            #pragma HLS UNROLL
            if(i + p < N) {
                for (int j = 0; j < K; j++) {
                    #pragma HLS PIPELINE II=1
                    C[i + p][j] = row_stream[p].read();
                }
            }
        }
    }
}

//--------------------------------------------------------------------------
// Top-level kernel: Sparse Matrix Multiply (A in CSR, B in CSC)
// with aggressive memory partitioning, parallelized compute, and streaming.
//--------------------------------------------------------------------------
void sparse_matrix_multiply_HLS(
    data_t values_A[N * M],
    int column_indices_A[N * M],
    int row_ptr_A[N + 1],
    data_t values_B[M * K],
    int row_indices_B[M * K],
    int col_ptr_B[M + 1],
    data_t C[N][K]
)
{
    // AXI interface directives for off-chip memory.
    #pragma HLS INTERFACE m_axi port=values_A         offset=slave bundle=mem1
    #pragma HLS INTERFACE m_axi port=column_indices_A   offset=slave bundle=mem2
    #pragma HLS INTERFACE m_axi port=row_ptr_A          offset=slave bundle=mem3
    #pragma HLS INTERFACE m_axi port=values_B           offset=slave bundle=mem4
    #pragma HLS INTERFACE m_axi port=row_indices_B      offset=slave bundle=mem5
    #pragma HLS INTERFACE m_axi port=col_ptr_B          offset=slave bundle=mem6
    #pragma HLS INTERFACE m_axi port=C                  offset=slave bundle=mem7
    #pragma HLS INTERFACE s_axilite port=return

    // Local BRAM buffers for matrix A.
    data_tx local_values_A[N * M];
    int local_column_indices_A[N * M];
    int local_row_ptr_A[N + 1];
    #pragma HLS ARRAY_PARTITION variable=local_values_A cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_column_indices_A cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_row_ptr_A cyclic factor=4 dim=1

    // Local BRAM buffers for matrix B.
    data_tx local_values_B[M * K];
    int local_row_indices_B[M * K];
    int local_col_ptr_B[M + 1];
    #pragma HLS ARRAY_PARTITION variable=local_values_B cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_row_indices_B cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_col_ptr_B cyclic factor=4 dim=1

    // Use DATAFLOW to overlap load, compute, and store stages.
    #pragma HLS DATAFLOW

    load_data(values_A, column_indices_A, row_ptr_A,
              values_B, row_indices_B, col_ptr_B,
              local_values_A, local_column_indices_A, local_row_ptr_A,
              local_values_B, local_row_indices_B, local_col_ptr_B);

    // Create an array of HLS streams to pass computed rows.
    hls::stream<data_tx> row_stream[PAR_FACTOR];
    #pragma HLS STREAM variable=row_stream depth=64

    compute(local_values_A, local_column_indices_A, local_row_ptr_A,
            local_values_B, local_row_indices_B, local_col_ptr_B,
            row_stream);

    store_data(row_stream, C);
}
