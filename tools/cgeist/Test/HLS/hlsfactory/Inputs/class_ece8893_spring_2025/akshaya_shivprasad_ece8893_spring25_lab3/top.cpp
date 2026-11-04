//#include "dcl.h"
//
//
//// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
//void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], 
//                             data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t C[N][K]) 
//{
//#pragma HLS interface m_axi port=values_A offset=slave bundle=mem1
//#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1
//#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1
//
//#pragma HLS interface m_axi port=values_B offset=slave bundle=mem2
//#pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2
//#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2
//
//#pragma HLS interface m_axi port=C offset=slave bundle=mem3
//
//#pragma HLS interface s_axilite port=return
//
//    // Perform Sparse x Sparse Multiplication
//    for (int i = 0; i < N; i++) {
//        for (int idx_A = row_ptr_A[i]; idx_A < row_ptr_A[i + 1]; idx_A++) {
//            int k = column_indices_A[idx_A]; // Column index of A
//            data_t value_A = values_A[idx_A];
//
//            // Iterate over columns of B corresponding to row k
//            for (int idx_B = col_ptr_B[k]; idx_B < col_ptr_B[k + 1]; idx_B++) {
//                int j = row_indices_B[idx_B]; // Column index of B
//                data_t value_B = values_B[idx_B];
//
//                // Accumulate the product into C[i][j]
//                C[i][j] += value_A * value_B;
//            }
//        }
//    }
//}

// Optimized

#include "dcl.h"


void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], 
                    data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t C[N][K])
{

    #pragma HLS interface m_axi port=values_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1

    #pragma HLS interface m_axi port=values_B offset=slave bundle=mem2
    #pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2
    #pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2

    #pragma HLS interface m_axi port=C offset=slave bundle=mem3

    #pragma HLS interface s_axilite port=return

    // enable datflow 
    #pragma HLS dataflow

    row_loop: for (int i = 0; i < N; i++) {
        //pipeline the outer loop 
        #pragma HLS pipeline II=1

        //local accumulation buffer for row i
        data_t local_C[K];
        //partition 
        #pragma HLS array_partition variable=local_C complete

        //initialize to 0
        init_local: for (int j = 0; j < K; j++) {
            #pragma HLS unroll
            local_C[j] = 0;
        }

        //iterate over nonzero elements in row i of A
        process_A: for (int idx_A = row_ptr_A[i]; idx_A < row_ptr_A[i + 1]; idx_A++) {
            #pragma HLS pipeline II=1
            int k = column_indices_A[idx_A];
            data_t a_val = values_A[idx_A];

            //iterate over non-zero elements in column k of B
            process_B: for (int idx_B = col_ptr_B[k]; idx_B < col_ptr_B[k + 1]; idx_B++) {
                #pragma HLS pipeline II=1
                int j = row_indices_B[idx_B];
                data_t b_val = values_B[idx_B];

                //multiply and accumulate in the local buffer
                local_C[j] += a_val * b_val;
            }
        }

        //write computed row back to matrix C
        write_back: for (int j = 0; j < K; j++) {
            #pragma HLS unroll
            C[i][j] = local_C[j];
        }
    }
}


