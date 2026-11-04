
#include "dcl.h"
#include <hls_stream.h>

// load data from matrix A
static void loadA(data_t *values_A, int *column_indices_A, int *row_ptr_A, hls::stream<data_t> &valA_stream, hls::stream<int> &colIdxA_stream, hls::stream<int> &rowPtrA_stream, hls::stream<int> &totalA_stream, int Nmax) {
    int total_A_elems = row_ptr_A[Nmax];
    totalA_stream.write(total_A_elems);
    for (int i = 0; i < Nmax + 1; i++) {
#pragma HLS PIPELINE II=1
        rowPtrA_stream.write(row_ptr_A[i]);
    }
    for (int i = 0; i < total_A_elems; i++) {
#pragma HLS PIPELINE II=1
        valA_stream.write(values_A[i]);
        colIdxA_stream.write(column_indices_A[i]);
    }
}

// load data from matrix B
static void loadB(data_t *values_B, int *row_indices_B, int *col_ptr_B, hls::stream<data_t> &valB_stream, hls::stream<int> &rowIdxB_stream, hls::stream<int> &colPtrB_stream, hls::stream<int> &totalB_stream, int Mmax) {
    int total_B_elems = col_ptr_B[Mmax];
    totalB_stream.write(total_B_elems);
    for (int i = 0; i < Mmax + 1; i++) {
#pragma HLS PIPELINE II=1
        colPtrB_stream.write(col_ptr_B[i]);
    }
    for (int i = 0; i < total_B_elems; i++) {
#pragma HLS PIPELINE II=1
        valB_stream.write(values_B[i]);
        rowIdxB_stream.write(row_indices_B[i]);
    }
}

// load data to local computation memory for compute_row function
static void prepareBData(hls::stream<int> &colPtrB_stream, hls::stream<data_t> &valB_stream, hls::stream<int> &rowIdxB_stream, int total_B_elems, int col_ptr_B_local[M+1], data_t valB_local[M*K], int rowIdxB_local[M*K]) {
    for (int i = 0; i < M+1; i++) {
#pragma HLS PIPELINE II=1
        col_ptr_B_local[i] = colPtrB_stream.read();
    }
    for (int i = 0; i < total_B_elems; i++) {
#pragma HLS PIPELINE II=1
        valB_local[i] = valB_stream.read();
        rowIdxB_local[i] = rowIdxB_stream.read();
    }
}

// store each row ptr locally
static void readRowPtrA(hls::stream<int> &rowPtrA_stream_s, int row_ptr_A_local[N+1]) {
    for (int i = 0; i < N+1; i++) {
#pragma HLS PIPELINE II=1
        row_ptr_A_local[i] = rowPtrA_stream_s.read();
    }
}

// inner two loops of computation. Note, we are only doing the math for the non-zero entry space
static void compute_row(int row_start, int row_end, hls::stream<data_t> &valA_stream, hls::stream<int> &colIdxA_stream, const int col_ptr_B_local[M+1], const data_t valB_local[M*K], const int rowIdxB_local[M*K], hls::stream<data_t> &cRowStream, int K_dim) {
    data_t c_row[K];
#pragma HLS ARRAY_PARTITION variable=c_row complete dim=1
	// unroll and clear c_row
    for (int j = 0; j < K_dim; j++) {
#pragma HLS UNROLL II=1
        c_row[j] = 0;
    }
	
	// for every element in the row start to row end area
    for (int idx = row_start; idx < row_end; idx++) {
#pragma HLS PIPELINE II=1
		// read the data and corresponding column space
        data_t a_val = valA_stream.read();
        int a_col = colIdxA_stream.read();
        int col_start = col_ptr_B_local[a_col];
        int col_end = col_ptr_B_local[a_col + 1];
		// for every corresponding non-zero column area
        for (int idxB = col_start; idxB < col_end; idxB++) {
#pragma HLS PIPELINE II=1
            data_t b_val = valB_local[idxB];
            int b_row = rowIdxB_local[idxB];
			// multiply the two elements and add it
            c_row[b_row] += a_val * b_val;
        }
    }
    for (int j = 0; j < K_dim; j++) {
#pragma HLS PIPELINE II=1
        cRowStream.write(c_row[j]);
    }
}

// compute rows main loop
static void computeRows(const int row_ptr_A_local[N+1], hls::stream<data_t> &valA_stream, hls::stream<int> &colIdxA_stream, const int col_ptr_B_local[M+1], const data_t valB_local[M*K], const int rowIdxB_local[M*K], hls::stream<data_t> &cRowStream, int K_dim) {
    // for every row, do the math
	for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        int row_start = row_ptr_A_local[i];
        int row_end = row_ptr_A_local[i+1];
        compute_row(row_start, row_end, valA_stream, colIdxA_stream, col_ptr_B_local, valB_local, rowIdxB_local, cRowStream, K_dim);
    }
}

// final store back to DRAM
static void storeC(hls::stream<data_t> &cRowStream, data_t C_out[N][K]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
#pragma HLS PIPELINE II=1
            C_out[i][j] = cRowStream.read();
        }
    }
}

void sparse_matrix_multiply_HLS(data_t values_A[N*M], int column_indices_A[N*M], int row_ptr_A[N+1], data_t values_B[M*K], int row_indices_B[M*K], int col_ptr_B[M+1], data_t C[N][K]) {
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE m_axi port=values_A offset=slave bundle=mem1 depth=4096 max_read_burst_length=512 num_read_outstanding=16 bus_bitwidth=512
#pragma HLS INTERFACE m_axi port=column_indices_A offset=slave bundle=mem1 depth=4096 max_read_burst_length=512 num_read_outstanding=16 bus_bitwidth=512
#pragma HLS INTERFACE m_axi port=row_ptr_A offset=slave bundle=mem1 depth=65 max_read_burst_length=512 num_read_outstanding=16 bus_bitwidth=512
#pragma HLS INTERFACE m_axi port=values_B offset=slave bundle=mem2 depth=4096 max_read_burst_length=512 num_read_outstanding=16 bus_bitwidth=512
#pragma HLS INTERFACE m_axi port=row_indices_B offset=slave bundle=mem2 depth=4096 max_read_burst_length=512 num_read_outstanding=16 bus_bitwidth=512
#pragma HLS INTERFACE m_axi port=col_ptr_B offset=slave bundle=mem2 depth=65 max_read_burst_length=512 num_read_outstanding=16 bus_bitwidth=512
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=mem3 depth=4096 max_write_burst_length=512 num_write_outstanding=16 bus_bitwidth=512
    hls::stream<data_t> valA_stream("valA_stream");
#pragma HLS STREAM variable=valA_stream depth=128
    hls::stream<int> colIdxA_stream("colIdxA_stream");
#pragma HLS STREAM variable=colIdxA_stream depth=128
    hls::stream<int> rowPtrA_stream_s("rowPtrA_stream");
#pragma HLS STREAM variable=rowPtrA_stream_s depth=128
    hls::stream<int> totalA_stream("totalA_stream");
#pragma HLS STREAM variable=totalA_stream depth=4
    hls::stream<data_t> valB_stream("valB_stream");
#pragma HLS STREAM variable=valB_stream depth=128
    hls::stream<int> rowIdxB_stream("rowIdxB_stream");
#pragma HLS STREAM variable=rowIdxB_stream depth=128
    hls::stream<int> colPtrB_stream("colPtrB_stream");
#pragma HLS STREAM variable=colPtrB_stream depth=128
    hls::stream<int> totalB_stream("totalB_stream");
#pragma HLS STREAM variable=totalB_stream depth=4
    hls::stream<data_t> cRowStream("cRowStream");
#pragma HLS STREAM variable=cRowStream depth=128
    static int row_ptr_A_local[N+1];
#pragma HLS ARRAY_PARTITION variable=row_ptr_A_local complete dim=1
    static int col_ptr_B_local[M+1];
#pragma HLS ARRAY_PARTITION variable=col_ptr_B_local complete dim=1
    static data_t valB_local[M*K];
    static int rowIdxB_local[M*K];
	
#pragma HLS DATAFLOW

	// dataflow bhavior
    loadA(values_A, column_indices_A, row_ptr_A, valA_stream, colIdxA_stream, rowPtrA_stream_s, totalA_stream, N);
    loadB(values_B, row_indices_B, col_ptr_B, valB_stream, rowIdxB_stream, colPtrB_stream, totalB_stream, M);
	
    int totalA = totalA_stream.read();
    int totalB = totalB_stream.read();
	
    readRowPtrA(rowPtrA_stream_s, row_ptr_A_local);
    prepareBData(colPtrB_stream, valB_stream, rowIdxB_stream, totalB, col_ptr_B_local, valB_local, rowIdxB_local);
    computeRows(row_ptr_A_local, valA_stream, colIdxA_stream, col_ptr_B_local, valB_local, rowIdxB_local, cRowStream, K);
    storeC(cRowStream, C);
}
