#include "dcl.h"
#include "hls_stream.h" //should include in the header file

//Loader function for CSR matrix A
void load_csr_matrix(
    data_t csr_values[N*M], int col_indices[N*M], int row_ptrs[N+1],
    hls::stream<data_t>& A_values, hls::stream<int>& A_cols, hls::stream<int>& A_rowptrs) 
{
    // Burst-load row pointers first
	#pragma HLS pipeline II=1
    for (int r = 0; r <= N; r++) {
        A_rowptrs.write(row_ptrs[r]); 
    }

    // Stream non-zero values and columns row-by-row
	#pragma HLS pipeline II=1
    for (int r = 0; r < N; r++) {
        int row_begin = row_ptrs[r];
        int row_end = row_ptrs[r+1];
        
		#pragma HLS pipeline II=1 
        for (int e = row_begin; e < row_end; e++) { //Load each element
            A_values.write(csr_values[e]);
            A_cols.write(col_indices[e]);
        }
    }
}

// CSC Loader - on-chip memory for random access
void load_csc_matrix(
    data_t csc_values[M*K], int row_indices[M*K], int col_ptrs[M+1],
    data_t local_B[M][K], int local_rows[M][K], int local_col_ptrs[M+1]) 
{
    #pragma HLS array_partition variable=local_col_ptrs cyclic factor=4
    #pragma HLS array_partition variable=local_B cyclic factor=8 dim=1
    #pragma HLS array_partition variable=local_rows cyclic factor=8 dim=1
	
//Load column pointers first
	#pragma HLS pipeline II=1
    for (int c = 0; c <= M; c++) {
        local_col_ptrs[c] = col_ptrs[c];
    }

	#pragma HLS pipeline II=1
    for (int c = 0; c < M; c++) { //Load columns
        int col_begin = local_col_ptrs[c];
        int col_end = local_col_ptrs[c+1];
        
		#pragma HLS pipeline II=1
        for (int e = 0; e < col_end - col_begin; e++) { //Load elements
            local_B[c][e] = csc_values[col_begin + e];
            local_rows[c][e] = row_indices[col_begin + e];
        }
    }
}

// Matrix Multiplication
void matrix_multiplication(
    hls::stream<data_t>& A_values, hls::stream<int>& A_cols, hls::stream<int>& A_rowptrs,
    data_t local_B[M][K], int local_rows[M][K], int local_col_ptrs[M+1],
    hls::stream<data_t>& C) 
{
    #pragma HLS array_partition variable=local_B cyclic factor=8 dim=1
    #pragma HLS array_partition variable=local_rows cyclic factor=8 dim=1

    // Load row pointers
    int local_A_rowptrs[N+1];
	#pragma HLS pipeline II=1
    for (int r = 0; r <= N; r++) {
        local_A_rowptrs[r] = A_rowptrs.read();
    }

    data_t result[N][K] = {0}; 
    #pragma HLS array_partition variable=result cyclic factor=16 dim=2

	#pragma HLS pipeline II=1
    for (int i = 0; i < N; i++) {
        int row_start = local_A_rowptrs[i];
        int row_end = local_A_rowptrs[i+1];
        
		//Look for non zero elements in A
		#pragma HLS pipeline II=1
        for (int p = row_start; p < row_end; p++) {
            data_t val_A = A_values.read();
            int k = A_cols.read();
            
            int col_start = local_col_ptrs[k];
            int col_end = local_col_ptrs[k+1];
            
			//non-zero B
			#pragma HLS pipeline II=1
            for (int q = col_start; q < col_end; q++) {
                int j = local_rows[k][q - col_start];
                data_t val_B = local_B[k][q - col_start];
                result[i][j] += val_A * val_B;
            }
        }
    }

    // Stream out results
	#pragma HLS pipeline II=1
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            #pragma HLS unroll factor=16
            C.write(result[i][j]);
        }
    }
}

// Stream results to memory
void store_result(hls::stream<data_t>& C_stream, data_t C[N][K]) {
    #pragma HLS inline off
    
	#pragma HLS pipeline II=1
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            #pragma HLS unroll factor=16
            C[i][j] = C_stream.read();
        }
    }
}

// Top function
void sparse_matrix_multiply_HLS(
    data_t values_A[N*M], int column_indices_A[N*M], int row_ptr_A[N+1],
    data_t values_B[M*K], int row_indices_B[M*K], int col_ptr_B[M+1],
    data_t C[N][K]) 
{
    #pragma HLS interface m_axi port=values_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1

    #pragma HLS interface m_axi port=values_B offset=slave bundle=mem2
    #pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2
    #pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2

    #pragma HLS interface m_axi port=C offset=slave bundle=mem3
    #pragma HLS interface s_axilite port=return
    
    #pragma HLS dataflow

    // Streams for Matrix A (CSR) and result C
	hls::stream<data_t> A_values, C_stream;
	hls::stream<int> A_cols, A_rowptrs;

	#pragma HLS STREAM variable=A_values depth=128    // CSR non-zero values
	#pragma HLS STREAM variable=A_cols depth=128      // CSR column indices
	#pragma HLS STREAM variable=A_rowptrs depth=128   // CSR row pointers
	#pragma HLS STREAM variable=C_stream depth=64     // Results (row-wise)

    // On-chip storage for Matrix B (CSC)
    data_t local_B[M][K];
    int local_B_rows[M][K];
    int local_B_colptrs[M+1];
    #pragma HLS array_partition variable=local_B cyclic factor=8 dim=1
    #pragma HLS array_partition variable=local_B_rows cyclic factor=8 dim=1

    // Execute in parallel
    load_csr_matrix(values_A, column_indices_A, row_ptr_A, A_values, A_cols, A_rowptrs);
    load_csc_matrix(values_B, row_indices_B, col_ptr_B, local_B, local_B_rows, local_B_colptrs);
    matrix_multiplication(A_values, A_cols, A_rowptrs, local_B, local_B_rows, local_B_colptrs, C_stream);
	store_result(C_stream, C); 
}