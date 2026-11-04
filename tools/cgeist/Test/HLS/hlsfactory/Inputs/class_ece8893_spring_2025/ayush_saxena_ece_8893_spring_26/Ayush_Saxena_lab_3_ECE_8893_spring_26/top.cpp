#include "dcl.h"
#include <hls_stream.h>

// Note: The following definitions (data_t, N, M, K, and SPARSITY) are provided via dcl.h.
// We define the maximum number of nonzeros per row based on the sparsity ratio.
// For example, if SPARSITY represents the percentage of non-zero elements, then:
#define MAX_NNZ_PER_ROW 64

// Structure to wrap a row of the sparse matrix A (in CSR format)
struct RowA {
    int nnz;                         // Number of nonzero elements in the row
    int col_indices[MAX_NNZ_PER_ROW]; // Column indices for nonzeros
    data_t values[MAX_NNZ_PER_ROW];   // Nonzero values
};

// Structure to wrap one dense output row of C
struct RowC {
    data_t data[K];                  // Dense row with K columns
};

//
// Load Task:
// Reads one row at a time from A�s CSR arrays (values_A, column_indices_A, and row_ptr_A)
// and packs the nonzero entries into a RowA struct, which is then streamed via a FIFO.
//
void load_task(
    data_t values_A[],         // CSR: total nonzeros in A
    int column_indices_A[],    // CSR: corresponding column indices for A
    int row_ptr_A[N+1],        // Row pointers for A (nnz in row i = row_ptr_A[i+1] - row_ptr_A[i])
    hls::stream<RowA> &a_fifo
) {
    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        RowA row_a;
        int start = row_ptr_A[i];
        int end   = row_ptr_A[i+1];
        row_a.nnz = end - start;
        for (int idx = start; idx < end; idx++) {
        #pragma HLS PIPELINE II=1
            int local_idx = idx - start;
            row_a.col_indices[local_idx] = column_indices_A[idx];
            row_a.values[local_idx] = values_A[idx];
        }
        a_fifo.write(row_a);
    }
}

//
// Compute Task:
// Copies the entire CSC representation of B into local buffers.
// Then for each row of A streamed from the load task, it computes the corresponding dense row of C.
// For every nonzero A(i,k), the function iterates over the CSC entries for B (using B�s col_ptr array)
// to accumulate contributions into the output dense row. The computed row is streamed as a RowC.
//
void compute_task(
    hls::stream<RowA> &a_fifo,
    hls::stream<RowC> &c_fifo,
    data_t values_B[],        // CSC values for B (length = nnz_B)
    int row_indices_B[],      // CSC row indices for B (length = nnz_B)
    int col_ptr_B[M+1]        // CSC column pointer for B (nnz in �row� k = col_ptr_B[k+1]-col_ptr_B[k])
) {
    // Local copies for B�s CSC representation.
    // nnz_B is given by col_ptr_B[M]. We assume B fits in local memory.
    int nnz_B = col_ptr_B[M];
    // For simplicity, we allocate buffers assuming worst-case size.
    data_t B_values_local[M * K];
    int    B_row_indices_local[M * K];
    int    B_col_ptr_local[M+1];

    // Copy the col pointer array for B.
    for (int i = 0; i < M+1; i++) {
    #pragma HLS PIPELINE II=1
        B_col_ptr_local[i] = col_ptr_B[i];
    }
    // Copy the nonzero values and row indices for B.
    for (int i = 0; i < nnz_B; i++) {
    #pragma HLS PIPELINE II=1
        B_values_local[i] = values_B[i];
        B_row_indices_local[i] = row_indices_B[i];
    }

    // Process each row of A.
    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        RowA row_a = a_fifo.read();
        // Create a temporary dense row for C and initialize it to zero.
        data_t rowC[K];
        for (int j = 0; j < K; j++) {
        #pragma HLS PIPELINE II=1
            rowC[j] = 0;
        }
        // For each nonzero element in the current row of A.
        for (int idx = 0; idx < row_a.nnz; idx++) {
        #pragma HLS PIPELINE II=1
            int k = row_a.col_indices[idx];  // Column index from A (i.e., A(i,k))
            data_t a_val = row_a.values[idx];
            // For the given k, use B�s CSC representation to update C(i,*).
            int b_start = B_col_ptr_local[k];
            int b_end   = B_col_ptr_local[k+1];
            for (int b_idx = b_start; b_idx < b_end; b_idx++) {
            #pragma HLS PIPELINE II=1
                int j = B_row_indices_local[b_idx]; // Column index for C
                rowC[j] += a_val * B_values_local[b_idx];
            }
        }
        // Pack the computed dense row into a RowC struct and stream it out.
        RowC row_c;
        for (int j = 0; j < K; j++) {
        #pragma HLS PIPELINE II=1
            row_c.data[j] = rowC[j];
        }
        c_fifo.write(row_c);
    }
}

//
// Store Task:
// Reads each computed RowC from the FIFO and writes it to the global dense matrix C.
//
void store_task(
    hls::stream<RowC> &c_fifo,
    data_t C[N][K]
) {
    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        RowC row_c = c_fifo.read();
        for (int j = 0; j < K; j++) {
        #pragma HLS PIPELINE II=1
            C[i][j] = row_c.data[j];
        }
    }
}

//
// Top-Level Function:
// Connects the load, compute, and store tasks using HLS DATAFLOW and FIFO streams.
// Memory interfaces are assigned via m_axi ports as specified in dcl.h.
//
void sparse_matrix_multiply_HLS(
    data_t values_A[N * M],  // CSR values for A (actual length = total nnz_A, = N*M)
    int column_indices_A[N * M], // CSR column indices for A
    int row_ptr_A[N+1],      // CSR row pointer array for A
    data_t values_B[M * K],  // CSC values for B (actual length = total nnz_B, = M*K)
    int row_indices_B[M * K],// CSC row indices for B
    int col_ptr_B[M+1],      // CSC col pointer array for B
    data_t C[N][K]           // Output dense matrix C
)
{
#pragma HLS INTERFACE m_axi port=values_A offset=slave bundle=mem1
#pragma HLS INTERFACE m_axi port=column_indices_A offset=slave bundle=mem1
#pragma HLS INTERFACE m_axi port=row_ptr_A offset=slave bundle=mem1

#pragma HLS INTERFACE m_axi port=values_B offset=slave bundle=mem2
#pragma HLS INTERFACE m_axi port=row_indices_B offset=slave bundle=mem2
#pragma HLS INTERFACE m_axi port=col_ptr_B offset=slave bundle=mem2

#pragma HLS INTERFACE m_axi port=C offset=slave bundle=mem3
#pragma HLS INTERFACE s_axilite port=return

    hls::stream<RowA> a_fifo;
    hls::stream<RowC> c_fifo;
#pragma HLS STREAM variable=a_fifo depth=128
#pragma HLS STREAM variable=c_fifo depth=128

#pragma HLS DATAFLOW
    load_task(values_A, column_indices_A, row_ptr_A, a_fifo);
    compute_task(a_fifo, c_fifo, values_B, row_indices_B, col_ptr_B);
    store_task(c_fifo, C);
}
