#include "dcl.h"

// void load_values_B(hls::vector<data_t, M> values_B[K], hls::vector<data_t, M> values_B_local[K]){
void load_values_B(hls::vector<data_t, VECFACTOR> values_B[R], hls::vector<data_t, VECFACTOR> values_B_local[R]){
    load_values_B:for(int i=0; i < (R); ++i) {
        // load_values_B:for(int j=0; j < K; ++j) {
        // #pragma HLS pipeline
        // #pragma HLS unroll factor=UFACTOR
        #pragma HLS performance target_tl=5000
        values_B_local[i] = values_B[i];
    }

}

void load_row_indices_B(hls::vector<int, VECFACTOR> row_indices_B[R], hls::vector<int, VECFACTOR> row_indices_B_local[R]){
    
    load_row_b:for(int i=0; i < (R); ++i) {
        // #pragma HLS performance
        // #pragma HLS pipeline
        // #pragma HLS unroll factor=UFACTOR
        #pragma HLS performance target_tl=5000
        row_indices_B_local[i] = row_indices_B[i];
    }

}

// void load_col_ptr_B(hls::vector<int, M + 1> &col_ptr_B, hls::vector<int, M+1> &col_ptr_B_local){
void load_col_ptr_B(int col_ptr_B[M + 1], hls::vector<int, M+1> &col_ptr_B_local){
    
    load_col_ptr_B:for(int i=0; i < (M + 1); ++i) {
        // # pragma 
        // #pragma HLS pipeline
        // #pragma HLS unroll factor=UFACTOR
        col_ptr_B_local[i] = col_ptr_B[i];
    }

}

void load_values_A(hls::vector<data_t, VECFACTOR> values_B[R], hls::stream<data_t> &values_B_local){
    load_values_A:for(int i=0; i < (R); ++i) {
        // load_values_B:for(int j=0; j < K; ++j) {
        // #pragma HLS pipeline
        // #pragma HLS unroll factor=UFACTOR
        #pragma HLS performance target_tl=5000
        for(int j=0; j < VECFACTOR; ++j) {
            values_B_local << values_B[i][j];
        }
    }

}

void load_row_indices_A(hls::vector<int, VECFACTOR> row_indices_B[R], hls::stream<int> &row_indices_B_local){
    
    load_row_a:for(int i=0; i < (R); ++i) {
        // #pragma HLS performance
        // #pragma HLS pipeline
        // #pragma HLS unroll factor=UFACTOR
        #pragma HLS performance target_tl=5000
        for(int j=0; j < VECFACTOR; ++j) {
            row_indices_B_local << row_indices_B[i][j];
        }
        // row_indices_B_local << row_indices_B[i];
    }

}

void load_col_ptr_A(int col_ptr_B[M + 1], hls::stream<int> &col_ptr_B_local){
    
    load_col_ptr_B:for(int i=0; i < (M + 1); ++i) {
        // # pragma 
        // #pragma HLS pipeline
        // #pragma HLS unroll factor=UFACTOR
        col_ptr_B_local << col_ptr_B[i];
    }

}

// void sparse_mult(hls::vector<data_t, N> values_A[M], hls::vector<int, N> column_indices_A[M], hls::vector<int, N + 1> row_ptr_A, 
//                              hls::vector<data_t, M> values_B[K], hls::vector<int, M> row_indices_B[K], hls::vector<int, M + 1> col_ptr_B, hls::vector<data_t, K> C[N]) {
// void sparse_mult(hls::vector<data_t, N> values_A[M], hls::vector<int, N> column_indices_A[M], hls::vector<int, N + 1> row_ptr_A, 
//                              hls::vector<data_t, M> values_B[K], hls::vector<int, M> row_indices_B[K], hls::vector<int, M + 1> col_ptr_B, hls::stream<hls::vector<data_t, K>> &C) {
void sparse_mult(hls::stream<data_t> &values_A, hls::stream<int> &column_indices_A, hls::stream<int> &row_ptr_stream, 
                             hls::vector<data_t, VECFACTOR> values_B[R], hls::vector<int, VECFACTOR> row_indices_B[R], hls::vector<int, M + 1> col_ptr_B, hls::stream<hls::vector<data_t, K>> &C) {
    
    int row_ptr_A_1, row_ptr_A_2;
    row_ptr_stream >> row_ptr_A_2;
    // row_ptr_stream >> row_ptr_A_2;

    loop_through_rows_A:for (int i = 0; i < N; i++) {
        hls::vector<data_t, K> C_temp;
        row_ptr_A_1 = row_ptr_A_2;
        row_ptr_stream >> row_ptr_A_2;
        for (int j = 0; j < K; j++) {
            C_temp[j] = 0;
        
        }
        row_ptr_A_loop:for (int idx_A = row_ptr_A_1; idx_A < row_ptr_A_2; idx_A++) {
            int idx_A_i = idx_A >> LOGVECFACTOR;
            int idx_A_j = idx_A & ((1 << LOGVECFACTOR) - 1);
            // int k = column_indices_A[idx_A_i][idx_A_j]; // Column index of A
            // data_t value_A = values_A[idx_A_i][idx_A_j];
            int k;
            data_t value_A;
            column_indices_A >> k; // Column index of A
            values_A >> value_A;

        


            // Iterate over columns of B corresponding to row k
            col_ptr_B_loop:for (int idx_B = col_ptr_B[k]; idx_B < col_ptr_B[k + 1]; idx_B++) {
            #pragma HLS pipeline II=1 
            #pragma HLS unroll factor=UFACTOR
                int idx_B_i = idx_B >> LOGVECFACTOR;
                int idx_B_j = idx_B & ((1 << LOGVECFACTOR) - 1);
                int j = row_indices_B[idx_B_i][idx_B_j]; // Column index of B
                data_t value_B = values_B[idx_B_i][idx_B_j];

                // Accumulate the product into C[i][j]
                C_temp[j] += value_A * value_B;
            }
        }
        C << C_temp;
    }
    
}

// void store_output(hls::vector<data_t, K> C_local[N], hls::stream<hls::vector<data_t, K>> &C){
void store_output(hls::stream<hls::vector<data_t, K>> &C_local, hls::vector<data_t, K> C[N]){
    store_output_row:for (int i = 0; i < N; i++) {
            // #pragma HLS pipeline
            // #pragma HLS unroll factor=UFACTOR
            C_local >> C[i];
    }
}

// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
// void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], 
                            //  data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t C[N][K]) 
// void sparse_matrix_multiply_HLS(hls::vector<data_t, N> values_A[M], hls::vector<int, N> column_indices_A[M], hls::vector<int, N + 1> row_ptr_A, 
                            //  hls::vector<data_t, M> values_B[K], hls::vector<int, M> row_indices_B[K], hls::vector<int, M + 1> col_ptr_B, data_t C[N][K]) 
void sparse_matrix_multiply_HLS(hls::vector<data_t, VECFACTOR> values_A[R], hls::vector<int, VECFACTOR> column_indices_A[R], int row_ptr_A[N + 1], 
                             hls::vector<data_t, VECFACTOR> values_B[R], hls::vector<int, VECFACTOR> row_indices_B[R], int col_ptr_B[M + 1], hls::vector<data_t, K> C[N]) 
{
#pragma HLS interface m_axi port=values_A offset=slave bundle=mem1 channel=1
#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1 channel=2
#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1 channel=3

#pragma HLS interface m_axi port=values_B offset=slave bundle=mem2 channel=1
#pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2 channel=2
#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2 channel=3

#pragma HLS interface m_axi port=C offset=slave bundle=mem3

#pragma HLS interface s_axilite port=return

#pragma HLS dataflow


    // Creating a local matrix to hold values
    // data_t values_B_local[M * K];
    // int row_indices_B_local[M * K], col_ptr_B_local[M + 1];
    hls::vector<data_t, VECFACTOR> values_B_local[R];
    hls::vector<int, VECFACTOR> row_indices_B_local[R]; 
    hls::vector<int, M + 1>col_ptr_B_local;
    
    // hls::vector<data_t, K> values_A_local[M];
    // hls::vector<int, K> column_indices_A_local[M]; 
    // hls::vector<int, M + 1>row_ptr_A_local;

    // data_t C_local[N][K];
    // hls::vector<data_t, K> C_local[N];
    hls::stream<hls::vector<data_t, K>, 4> C_local;
    hls::stream<data_t, 2> values_A_local;
    hls::stream<int, 2> column_indices_A_local;
    hls::stream<int, 2> row_ptr_A_local;
    
    
    
    
    // #pragma HLS array_partition variable=values_A_local type=cyclic factor=UFACTOR*2 
    // #pragma HLS array_partition variable=column_indices_A_local type=cyclic factor=UFACTOR*2 
    // #pragma HLS array_partition variable=row_ptr_A_local type=cyclic factor=UFACTOR*2 
    
    #pragma HLS array_partition variable=values_B_local type=cyclic factor=UFACTOR
    #pragma HLS array_partition variable=row_indices_B_local type=cyclic factor=UFACTOR 
    #pragma HLS array_partition variable=col_ptr_B_local type=cyclic factor=UFACTOR
    
    // #pragma HLS array_partition variable=C_local type=cyclic factor=UFACTOR*2

    // Partitioning local matrix

    // Copying to local matrices
    load_values_B(values_B, values_B_local);
    load_row_indices_B(row_indices_B, row_indices_B_local);
    load_col_ptr_B(col_ptr_B, col_ptr_B_local);
    
    load_values_A(values_A, values_A_local);
    load_row_indices_A(column_indices_A, column_indices_A_local);
    load_col_ptr_A(row_ptr_A, row_ptr_A_local);
    

    // Perform Sparse x Sparse Multiplication
    // sparse_mult(values_A_local, column_indices_A_local, row_ptr_A_local, values_B_local, row_indices_B_local, col_ptr_B_local, C);
    sparse_mult(values_A_local, column_indices_A_local, row_ptr_A_local, values_B_local, row_indices_B_local, col_ptr_B_local, C_local);

    store_output(C_local, C);
}
