#include "dcl.h"
#include "hls_stream.h"


void copy_B(
    data_t B_values_src[M * K],
    int    B_row_indices_src[M * K],
    int    B_col_ptr_src[M + 1],
    data_t B_values_dst[M * K],
    int    B_row_indices_dst[M * K],
    int    B_col_ptr_dst[M + 1]
) {
    
    copy_B_ptrs: for (int i = 0; i < M + 1; i++) {
    #pragma HLS PIPELINE II=1
        B_col_ptr_dst[i] = B_col_ptr_src[i];
    }

    int B_nnz = B_col_ptr_src[M];

    copy_B_vals: for (int i = 0; i < B_nnz; i++) {
    #pragma HLS PIPELINE II=1
        B_values_dst[i]      = B_values_src[i];
        B_row_indices_dst[i] = B_row_indices_src[i];
    }
}

void copy_A(
       data_t A_values_src[N * M],
       int    A_col_indices_src[N * M],
       int    A_row_ptr_src[N + 1],
       data_t A_values_dst[N * M],
       int    A_col_indices_dst[N * M],
      int    A_row_ptr_dst[N + 1]
  ) {
 
      copy_A_ptrs: for (int i = 0; i < N + 1; i++) {
      #pragma HLS PIPELINE II=1
          A_row_ptr_dst[i] = A_row_ptr_src[i];
      }
 
      int A_nnz = A_row_ptr_src[N];
 
      copy_A_vals: for (int i = 0; i < A_nnz; i++) {
      #pragma HLS PIPELINE II=1
          A_values_dst[i]         = A_values_src[i];
          A_col_indices_dst[i] = A_col_indices_src[i];
      }
  }


void get_A_elements(
    int row_idx,
    int A_row_ptr_dst[N + 1],
    int A_col_indices_dst[N * M],
    data_t A_values_dst[N * M],
    hls::stream<int> &stream_k,
    hls::stream<data_t> &stream_A_val
) {
    get_A_loop: for (int idx = A_row_ptr_dst[row_idx]; idx < A_row_ptr_dst[row_idx + 1]; idx++) {
    #pragma HLS PIPELINE II=1
        int k = A_col_indices_dst[idx];
        data_t A_value = A_values_dst[idx];
        stream_k.write(k);
        stream_A_val.write(A_value);
    }
}

void get_B_elements(
    int row_idx,
    int A_row_ptr_dst[N + 1],
    int A_col_indices_dst[N * M],
    data_t A_values_dst[N * M],
    int B_col_ptr_dst[M + 1],
    int B_row_indices_dst[M * K],
    data_t B_values_dst[M * K],
    hls::stream<int> &stream_k,
    hls::stream<data_t> &stream_A_val,
    hls::stream<int> &stream_j,
    hls::stream<data_t> &stream_B_val,
    hls::stream<data_t> &stream_A_elem
) {
    int num_A = A_row_ptr_dst[row_idx + 1] - A_row_ptr_dst[row_idx];
    get_B_loop: for (int idx = 0; idx < num_A; idx++) {
    #pragma HLS PIPELINE II=1
        int k = stream_k.read();
        data_t A_value = stream_A_val.read();
        for (int idxB = B_col_ptr_dst[k]; idxB < B_col_ptr_dst[k + 1]; idxB++) {
        #pragma HLS PIPELINE II=1
            int j = B_row_indices_dst[idxB];
            data_t B_value = B_values_dst[idxB];
            stream_j.write(j);
            stream_B_val.write(B_value);
            stream_A_elem.write(A_value);
        }
    }
}

void compute_C(
    int row_idx,
    int A_row_ptr_dst[N + 1],
    int A_col_indices_dst[N * M],
    data_t A_values_dst[N * M],
    int B_col_ptr_dst[M + 1],
    int B_row_indices_dst[M * K],
    data_t B_values_dst[M * K],
    hls::stream<int> &stream_j,
    hls::stream<data_t> &stream_B_val,
    hls::stream<data_t> &stream_A_elem,
    data_t local_C_row[K]
) {
    int count = 0;
    count_loop: for (int idx = A_row_ptr_dst[row_idx]; idx < A_row_ptr_dst[row_idx + 1]; idx++) {
    #pragma HLS PIPELINE II=1
        int k = A_col_indices_dst[idx];
        count += (B_col_ptr_dst[k + 1] - B_col_ptr_dst[k]);
    }
    compute_loop: for (int idx = 0; idx < count; idx++) {
    #pragma HLS PIPELINE II=1
        int j = stream_j.read();
        data_t B_value = stream_B_val.read();
        data_t A_value = stream_A_elem.read();
        local_C_row[j] += A_value * B_value;
    }
}

void process_row(
    int row_idx,
    int A_row_ptr_dst[N + 1],
    int A_col_indices_dst[N * M],
    data_t A_values_dst[N * M],
    int B_col_ptr_dst[M + 1],
    int B_row_indices_dst[M * K],
    data_t B_values_dst[M * K],
    data_t local_C_row[K]
) {
    initialize_C: for (int j = 0; j < K; j++) {
    #pragma HLS PIPELINE II=1
        local_C_row[j] = 0;
    }
    {
        hls::stream<int> stream_k("stream_k");
        hls::stream<data_t> stream_A_val("stream_A_val");
        hls::stream<int> stream_j("stream_j");
        hls::stream<data_t> stream_B_val("stream_B_val");
        hls::stream<data_t> stream_A_elem("stream_A_elem");
    #pragma HLS STREAM variable=stream_k    depth=64
    #pragma HLS STREAM variable=stream_A_val depth=64
    #pragma HLS STREAM variable=stream_j    depth=4096
    #pragma HLS STREAM variable=stream_B_val depth=4096
    #pragma HLS STREAM variable=stream_A_elem depth=4096
	#pragma HLS DATAFLOW
        get_A_elements(row_idx, A_row_ptr_dst, A_col_indices_dst, A_values_dst, stream_k, stream_A_val);
        get_B_elements(row_idx, A_row_ptr_dst, A_col_indices_dst, A_values_dst, B_col_ptr_dst, B_row_indices_dst, B_values_dst, stream_k, stream_A_val, stream_j, stream_B_val, stream_A_elem);
        compute_C(row_idx, A_row_ptr_dst, A_col_indices_dst, A_values_dst, B_col_ptr_dst, B_row_indices_dst, B_values_dst, stream_j, stream_B_val, stream_A_elem, local_C_row);
    }
}
void sparse_matrix_multiply_HLS(
    data_t values_A[N * M],
    int    column_indices_A[N * M],
    int    row_ptr_A[N + 1],
    data_t values_B[M * K],
    int    row_indices_B[M * K],
    int    col_ptr_B[M + 1],
    data_t C[N][K]
) {
#pragma HLS interface m_axi port=values_A         offset=slave bundle=mem1
#pragma HLS interface m_axi port=column_indices_A   offset=slave bundle=mem1
#pragma HLS interface m_axi port=row_ptr_A          offset=slave bundle=mem1

#pragma HLS interface m_axi port=values_B         offset=slave bundle=mem2
#pragma HLS interface m_axi port=row_indices_B    offset=slave bundle=mem2
#pragma HLS interface m_axi port=col_ptr_B        offset=slave bundle=mem2

#pragma HLS interface m_axi port=C                  offset=slave bundle=mem3
#pragma HLS interface s_axilite port=return


    data_t values_A_local[N * M];
    int    column_indices_A_local[N * M];
    int    row_ptr_A_local[N + 1];

    data_t values_B_local[M * K];
    int    row_indices_B_local[M * K];
    int    col_ptr_B_local[M + 1];


    data_t local_C_copy[N][K];



    

    copy_A(
        values_A, column_indices_A, row_ptr_A,
        values_A_local, column_indices_A_local, row_ptr_A_local
    );
    copy_B(
        values_B, row_indices_B, col_ptr_B,
        values_B_local, row_indices_B_local, col_ptr_B_local
    );
    

    { 
#pragma HLS PIPELINE II=1
    process_rows: for (int i = 0; i < N; i++) {
#pragma HLS UNROLL factor=64  
        process_row(i, row_ptr_A_local, column_indices_A_local, values_A_local,
            col_ptr_B_local, row_indices_B_local, values_B_local, local_C_copy[i]);
    }
    }

    copy_C_output: for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
        #pragma HLS PIPELINE II=1
            C[i][j] = local_C_copy[i][j];
        }
    }
}

