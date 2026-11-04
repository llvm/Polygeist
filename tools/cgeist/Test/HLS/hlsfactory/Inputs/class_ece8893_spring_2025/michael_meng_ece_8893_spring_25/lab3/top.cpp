#include "dcl.h"
#include "string.h"


// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], 
                             data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], hls::vector<data_t, K> C[N]) 
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

hls::stream<hls::vector<data_t, K>, 2>  C_local;
//hls::stream<hls::vector<data_t, K>, 2>  values_A_local;
data_t  values_A_local[N * M];
data_t  values_B_local[M * K];
int column_indices_A_local[N * M];
int row_ptr_A_local[N + 1];
int row_indices_B_local[M * K];
int col_ptr_B_local[M + 1];


//#pragma HLS array_partition variable=C_local type=cyclic factor=16 dim=
#pragma HLS array_partition variable=values_A_local type=cyclic factor=8 dim=1
#pragma HLS array_partition variable=values_B_local type=cyclic factor=64 dim=1
#pragma HLS array_partition variable=column_indices_A_local type=cyclic factor=8 dim=1
#pragma HLS array_partition variable=row_ptr_A_local type=cyclic factor=8 dim=1
#pragma HLS array_partition variable=row_indices_B_local type=cyclic factor=8 dim=1
#pragma HLS array_partition variable=col_ptr_B_local type=cyclic factor=8 dim=1

memcpy(values_B_local, values_B, sizeof(data_t)*M*K);
memcpy(values_A_local, values_A, sizeof(data_t)*N*M);
memcpy(column_indices_A_local, column_indices_A, sizeof(int)*(N * M));
memcpy(row_ptr_A_local, row_ptr_A, sizeof(int)*(N + 1));
memcpy(row_indices_B_local, row_indices_B, sizeof(int)*(M*K));
memcpy(col_ptr_B_local, col_ptr_B, sizeof(int)*(M + 1));

    // Perform Sparse x Sparse Multiplication
    A_row_index: for (int i = 0; i < N; i++) {
        hls::vector<data_t, K> output = data_t{0}; 
        B_column_index: for (int idx_A = row_ptr_A_local[i]; idx_A < row_ptr_A_local[i + 1]; idx_A++) {
            #pragma HLS unroll factor=8
            int k = column_indices_A_local[idx_A]; // Column index of A
            data_t value_A = values_A_local[idx_A];
            // Iterate over columns of B corresponding to row k
            ACC: for (int idx_B = col_ptr_B_local[k]; idx_B < col_ptr_B_local[k + 1]; idx_B++) {
                int j = row_indices_B_local[idx_B]; // Row index of B
                data_t value_B = values_B_local[idx_B];
                // Accumulate the product into C[i][j]
                output[j] += value_A * value_B;
            }
        }
        C_local.write(output);

    }
    
    return_output: for (int n=0; n<N; n++){
        //#pragma HLS unroll factor=4
        hls::vector<data_t, K> out = C_local.read();
        C[n] = out;
    }

}

