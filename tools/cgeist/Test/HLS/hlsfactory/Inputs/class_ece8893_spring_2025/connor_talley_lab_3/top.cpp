#include "dcl.h"

//load A data
void load_A_HLS(data_t values[N * M], int column_indices[N * M], hls::stream<data_t> &A_values, hls::stream<coord_t> &column_indices_A, int end_index) {
    for (int i = 0; i < end_index; i++) {
        A_values.write(values[i]);
        column_indices_A.write(column_indices[i]);
    }
}

//load B data
void load_B_HLS(data_t values[M * K], int row_indices[M * K], hls::stream<data_t> &B_values, hls::stream<coord_t> &row_indices_B, int end_index) {
    for (int i = 0; i < end_index; i++) {
        B_values.write(values[i]);
        row_indices_B.write(row_indices[i]);
    }
}

//uncompress CSR A
void uncompress_A_HLS(hls::stream<data_t> &A_values, hls::stream<coord_t> &column_indices, const hls::vector<int, N+1> &row_ptr,
                        hls::stream<hls::vector<data_t, M>> &values_uncompressed) {
    // values_uncompressed = {};
    for (int i = 0; i < N; i++) {
        hls::vector<data_t, M> A_row = data_t(0);
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            int column_index = column_indices.read();
            A_row[column_index] = A_values.read();
        }
        values_uncompressed.write(A_row);
    }
}

//uncompress CSC B in transpose form (CSC is already column major)
void uncompress_B_HLS(hls::stream<data_t> &B_values, hls::stream<coord_t> &row_indices, const hls::vector<int, M+1> &col_ptr, hls::stream<hls::vector<data_t, M>> &values_uncompressed) {
    // values_uncompressed = {};
    hls::vector<data_t, M> B[K] = {hls::vector<data_t, M>(0)};
    for (int i = 0; i < K; i++) {
        hls::vector<data_t, M> B_col = data_t(0);
        for (int j = col_ptr[i]; j < col_ptr[i + 1]; j++) {
            coord_t row_index = row_indices.read();
            B_col[row_index] = B_values.read();
        }
        values_uncompressed.write(B_col);
    }
}

//matmul of A and B
void matmul_HLS(hls::stream<hls::vector<data_t, M>> &A, hls::stream<hls::vector<data_t, M>> &B, hls::stream<hls::vector<data_t, K>> &C) {
    //load B
    hls::vector<data_t, M> B_local[K];
    for (int i = 0; i < M; i++) {
        B_local[i] = B.read();
    }
    ROWS: for (int i = 0; i < N; i++) {
        // #pragma HLS unroll factor=2
        hls::vector<data_t, M> A_row = A.read();
        hls::vector<data_t, K> C_row = data_t(0);
        COLUMNS: for (int j = 0; j < K; j++) {
            #pragma HLS loop_flatten
            #pragma HLS unroll factor=5
            //get the mult result
            // hls::vector<data_t, M> mult = A_row * B_col;
            //sum the mult result
            data_t sum = 0;
            ACCUMULATE: for (int k = 0; k < M; k++) {
                // #pragma HLS pipeline
                sum += A_row[k] * B_local[k][j];
            }
            C_row[j] = sum;
            // C.write(sum);
        }
        C.write(C_row);
    }
}

void store_output_HLS(hls::stream<hls::vector<data_t, K>> &C_local, hls::vector<data_t, K> C[N]) {
    for (int i = 0; i < N; i++) {
        hls::vector<data_t, K> C_row = C_local.read();
        C[i] = C_row;
    }
}
    

// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], hls::vector<int, N+1> row_ptr_A, 
                             data_t values_B[M * K], int row_indices_B[M * K], hls::vector<int, M+1> col_ptr_B, hls::vector<data_t, K> C[N]) 
{
    #pragma HLS interface mode=ap_fifo port=values_A
    #pragma HLS interface mode=ap_fifo port=column_indices_A
    #pragma HLS interface s_axilite port=row_ptr_A

    #pragma HLS interface mode=ap_fifo port=values_B
    #pragma HLS interface mode=ap_fifo port=row_indices_B
    #pragma HLS interface s_axilite port=col_ptr_B

    #pragma HLS interface mode=ap_fifo port=C

    #pragma HLS interface s_axilite port=return

    #pragma HLS dataflow

    //ideas:
    //burst read rows and columns then use pointers to filter out what get stored in the dataflow
    //pair up the operands beforehand and use dataflow
    //load the entirety of the pointers to the chip in a burst
    //honestly first attempt is to just uncompress the matrices once they are on chip
    //index values never go higher than 64, so we can use 6 bit numbers for them instead of 32 bit


    #pragma HLS dataflow

    //create local copy of pointer vectors
    hls::vector<int, N + 1> row_ptr_A_local = row_ptr_A;
    hls::vector<int, M + 1> col_ptr_B_local = col_ptr_B;

    //create local arrays for compressed matrices
    hls::stream<coord_t, 2> column_indices_A_local;
    hls::stream<int, 2> row_indices_B_local;
    hls::stream<data_t, 2> values_A_local;
    hls::stream<data_t, 2> values_B_local;

    //create local arrays for result and uncompressed matrices
    // static hls::vector<data_t, M> values_A_uncompressed[N];
    hls::stream<hls::vector<data_t, M>, 2> values_A_uncompressed;
    // #pragma HLS array_partition variable=values_A_uncompressed type=cyclic factor=2

    // static hls::vector<data_t, M> values_B_uncompressed[K];
    hls::stream<hls::vector<data_t, M>, 2> values_B_uncompressed;
    // #pragma HLS array_partition variable=values_B_uncompressed type=cyclic factor=2

    //need to figure out why the stream is so much slower
    hls::stream<hls::vector<data_t, K>, 2> C_local;

    //load the matrices
    int end_index_A = row_ptr_A_local[N];
    int end_index_B = col_ptr_B_local[M];
    load_A_HLS(values_A, column_indices_A, values_A_local, column_indices_A_local, end_index_A);
    load_B_HLS(values_B, row_indices_B, values_B_local, row_indices_B_local, end_index_B);

    //uncompress the matrices
    // uncompress_A_HLS(values_A, column_indices_A, row_ptr_A_local, values_A_uncompressed);
    // uncompress_B_HLS(values_B, row_indices_B, col_ptr_B_local, values_B_uncompressed);
    uncompress_A_HLS(values_A_local, column_indices_A_local, row_ptr_A_local, values_A_uncompressed);
    uncompress_B_HLS(values_B_local, row_indices_B_local, col_ptr_B_local, values_B_uncompressed);

    //multiply the matrices
    matmul_HLS(values_A_uncompressed, values_B_uncompressed, C_local);

    //store the result in the output array
    store_output_HLS(C_local, C);

}
