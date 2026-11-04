#include "dcl.h"
#include <hls_stream.h>

//-----------------------------------------------------------------
// Define a structure to represent a sparse row from matrix A
//-----------------------------------------------------------------
typedef struct {
    int nnz;            // number of nonzero elements in the row
    data_t val[M];      // nonzero values (maximum M per row)
    int col_idx[M];     // corresponding column indices
} A_row_t;

//-----------------------------------------------------------------
// Define a structure for a complete output row
//-----------------------------------------------------------------
typedef struct {
    data_t data[K];
} row_t;

//-----------------------------------------------------------------
// Load matrix A rows from DRAM and stream them as A_row_t
//-----------------------------------------------------------------
void load_A_rows(
    data_t values_A[N * M],
    int column_indices_A[N * M],
    int row_ptr_A[N + 1],
    hls::stream<A_row_t> &a_row_stream
) {
    for (int i = 0; i < N; i++) {
        A_row_t a_row;
        int start = row_ptr_A[i];
        int end = row_ptr_A[i + 1];
        a_row.nnz = end - start;
        row_load: for (int j = 0; j < a_row.nnz; j++) {
            #pragma HLS pipeline II=1
            a_row.val[j]     = values_A[start + j];
            a_row.col_idx[j] = column_indices_A[start + j];
        }
        a_row_stream.write(a_row);
    }
}

//-----------------------------------------------------------------
// Load matrix B from DRAM into local BRAM buffers
//-----------------------------------------------------------------
void load_B_data(
    data_t values_B[M * K],
    int row_indices_B[M * K],
    int col_ptr_B[M + 1],
    data_t local_values_B[M * K],
    int local_row_indices_B[M * K],
    int local_col_ptr_B[M + 1]
) {
    B_values: for (int i = 0; i < M * K; i++) {
        #pragma HLS pipeline II=1
        local_values_B[i] = values_B[i];
    }
    B_indices: for (int i = 0; i < M * K; i++) {
        #pragma HLS pipeline II=1
        local_row_indices_B[i] = row_indices_B[i];
    }
    B_colptr: for (int i = 0; i < M + 1; i++) {
        #pragma HLS pipeline II=1
        local_col_ptr_B[i] = col_ptr_B[i];
    }
}

//-----------------------------------------------------------------
// Process each row: read an A row from stream, perform sparse multiplication
// using local matrix B, and write the computed output row into a stream
//-----------------------------------------------------------------
void process_rows(
    hls::stream<A_row_t> &a_row_stream,
    data_t local_values_B[M * K],
    int local_row_indices_B[M * K],
    int local_col_ptr_B[M + 1],
    hls::stream<row_t> &row_stream
) {
    for (int i = 0; i < N; i++) {
        A_row_t a_row = a_row_stream.read();
        row_t r;
        #pragma HLS array_partition variable=r.data complete dim=1

        // Initialize output row to zero (fully unrolled)
        init_C: for (int j = 0; j < K; j++) {
            #pragma HLS unroll
            r.data[j] = 0;
        }
        // Iterate over nonzero elements in this A row
        process_nonzeros: for (int idx = 0; idx < a_row.nnz; idx++) {
            #pragma HLS pipeline II=1
            int k = a_row.col_idx[idx];
            data_t A_val = a_row.val[idx];
            // For each nonzero element in column k of matrix B (CSC format)
            B_loop: for (int idx_B = local_col_ptr_B[k]; idx_B < local_col_ptr_B[k + 1]; idx_B++) {
                #pragma HLS pipeline II=1
                #pragma HLS unroll factor=16
                int j = local_row_indices_B[idx_B];
                data_t B_val = local_values_B[idx_B];
                r.data[j] += A_val * B_val;  // 直接使用乘法替代 dsp_mult
            }
        }
        row_stream.write(r);
    }
}

//-----------------------------------------------------------------
// Store computed rows: read from row stream and write to output matrix C
//-----------------------------------------------------------------
void store_rows(
    hls::stream<row_t> &row_stream,
    data_t C[N][K]
) {
    for (int i = 0; i < N; i++) {
        row_t r = row_stream.read();
        store_row: for (int j = 0; j < K; j++) {
            #pragma HLS pipeline II=1
            C[i][j] = r.data[j];
        }
    }
}

//-----------------------------------------------------------------
// Top-level function
//-----------------------------------------------------------------
void sparse_matrix_multiply_HLS(
    data_t values_A[N * M],
    int column_indices_A[N * M],
    int row_ptr_A[N + 1],
    data_t values_B[M * K],
    int row_indices_B[M * K],
    int col_ptr_B[M + 1],
    data_t C[N][K]
) {
    #pragma HLS interface m_axi port=values_A       offset=slave bundle=mem1
    #pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=row_ptr_A        offset=slave bundle=mem1
    #pragma HLS interface m_axi port=values_B         offset=slave bundle=mem2
    #pragma HLS interface m_axi port=row_indices_B      offset=slave bundle=mem2
    #pragma HLS interface m_axi port=col_ptr_B          offset=slave bundle=mem2
    #pragma HLS interface m_axi port=C                offset=slave bundle=mem3
    #pragma HLS interface s_axilite port=return

    #pragma HLS DATAFLOW

    // Local buffers for matrix B data (CSC format)
    data_t local_values_B[M * K];
    int local_row_indices_B[M * K];
    int local_col_ptr_B[M + 1];

    #pragma HLS array_partition variable=local_values_B cyclic factor=8
    #pragma HLS array_partition variable=local_row_indices_B cyclic factor=8
    #pragma HLS array_partition variable=local_col_ptr_B complete

    // Streams for matrix A rows and computed output rows
    hls::stream<A_row_t> a_row_stream;
    #pragma HLS stream variable=a_row_stream depth=8
    hls::stream<row_t> row_stream;
    #pragma HLS stream variable=row_stream depth=4

    // Load matrix B from DRAM into local BRAM
    load_B_data(values_B, row_indices_B, col_ptr_B,
                local_values_B, local_row_indices_B, local_col_ptr_B);

    // Load matrix A rows from DRAM and stream them
    load_A_rows(values_A, column_indices_A, row_ptr_A, a_row_stream);

    // Process each row: perform multiplication using streamed A rows and local B arrays
    process_rows(a_row_stream, local_values_B, local_row_indices_B, local_col_ptr_B, row_stream);

    // Store computed output rows from stream into output matrix C
    store_rows(row_stream, C);
}