#include "dcl.h"

#include <hls_math.h>
typedef ap_uint<512> wide_bus_t;
#define ELEMS_PER_BUS_FOR_DATA_T (512 / 8 / sizeof(data_t))
#define ELEMS_PER_BUS_FOR_INT (512 / 8 / sizeof(int))
#define number_of_parallel_blocks  16


void STORE_OUTPUT(
    data_t C_local[N][K],
    data_t C[N][K]
){
    const int TOTAL_ELEMENTS = N * K;
    STORE_OUTPUT:
    for (int i = 0; i < TOTAL_ELEMENTS; i += ELEMS_PER_BUS_FOR_DATA_T)
    {
    #pragma HLS PIPELINE II=1
        wide_bus_t temp_512 = 0;
        for (int ii = 0; ii < ELEMS_PER_BUS_FOR_DATA_T; ii++)
        {
    #pragma HLS UNROLL
            int index = i + ii;
            if (index < TOTAL_ELEMENTS) {
                int row = index / K;
                int col = index % K;
                ap_uint<16> bits_16 = C_local[row][col].range(15, 0);
                temp_512.range((ii + 1) * 16 - 1, ii * 16) = bits_16;
            }
        }
        *reinterpret_cast<wide_bus_t*>(&C[0][i]) = temp_512;
    }
}
//---------------------------
// Matrix A Loading Functions
//---------------------------
void LOAD_VALUES_A(
    data_t values_A[N * M],
    data_t values_A_local[N * M]
){
    const int TOTAL = N * M;
    LOAD_VALUES_A_LOOP:
    for(int i=0; i<TOTAL; i+=ELEMS_PER_BUS_FOR_DATA_T){
    #pragma HLS PIPELINE II=1
        wide_bus_t temp = *reinterpret_cast<wide_bus_t*>(&values_A[i]);
        LOAD_VALUES_A_LOOP_INNER:
        for(int ii=0; ii<ELEMS_PER_BUS_FOR_DATA_T; ii++){
        #pragma HLS UNROLL
            if((i+ii) < TOTAL) {
                ap_uint<16> bits = temp.range(16*(ii+1)-1, 16*ii);
                values_A_local[i+ii] = *reinterpret_cast<data_t*>(&bits);
            }
        }
    }
}

void LOAD_COLUMN_INDICES_A(
    int column_indices_A[N * M],
    int column_indices_A_local[N * M]
){
    const int TOTAL = N * M;
    LOAD_COLUMN_INDICES_A_LOOP:
    for(int i=0; i<TOTAL; i+=ELEMS_PER_BUS_FOR_INT){
    #pragma HLS PIPELINE II=1
        wide_bus_t temp = *reinterpret_cast<wide_bus_t*>(&column_indices_A[i]);
        LOAD_COLUMN_INDICES_A_LOOP_INNER:
        for(int ii=0; ii<ELEMS_PER_BUS_FOR_INT; ii++){
        #pragma HLS UNROLL
            if((i+ii) < TOTAL) {
                ap_uint<32> bits = temp.range(32*(ii+1)-1, 32*ii);
                column_indices_A_local[i+ii] = *reinterpret_cast<int*>(&bits);
            }
        }
    }
}

void LOAD_ROW_PTR_A(
    int row_ptr_A[N + 1],
    int row_ptr_A_local[N + 1]
){
    const int TOTAL = N + 1;
    LOAD_ROW_PTR_A_LOOP:
    for(int i=0; i<TOTAL; i+=ELEMS_PER_BUS_FOR_INT){
    #pragma HLS PIPELINE II=1
        wide_bus_t temp = *reinterpret_cast<wide_bus_t*>(&row_ptr_A[i]);
        LOAD_ROW_PTR_A_LOOP_INNER:
        for(int ii=0; ii<ELEMS_PER_BUS_FOR_INT; ii++){
        #pragma HLS UNROLL
            if((i+ii) < TOTAL) {
                ap_uint<32> bits = temp.range(32*(ii+1)-1, 32*ii);
                row_ptr_A_local[i+ii] = *reinterpret_cast<int*>(&bits);
            }
        }
    }
}

//---------------------------
// Matrix B Loading Functions
//---------------------------
void LOAD_VALUES_B(
    data_t values_B[M * K],  // Note different size
    data_t values_B_local[M * K]
){
    const int TOTAL = M * K;
    LOAD_VALUES_B_LOOP:
    for(int i=0; i<TOTAL; i+=ELEMS_PER_BUS_FOR_DATA_T){
    #pragma HLS PIPELINE II=1
        wide_bus_t temp = *reinterpret_cast<wide_bus_t*>(&values_B[i]);
        LOAD_VALUES_B_LOOP_INNER:
        for(int ii=0; ii<ELEMS_PER_BUS_FOR_DATA_T; ii++){
        #pragma HLS UNROLL
            if((i+ii) < TOTAL) {
                ap_uint<16> bits = temp.range(16*(ii+1)-1, 16*ii);
                values_B_local[i+ii] = *reinterpret_cast<data_t*>(&bits);
            }
        }
    }
}

void LOAD_ROW_INDICES_B(
    int row_indices_B[M * K],
    int row_indices_B_local[M * K]
){
    const int TOTAL = M * K;
    LOAD_ROW_INDICES_B_LOOP:
    for(int i=0; i<TOTAL; i+=ELEMS_PER_BUS_FOR_INT){
    #pragma HLS PIPELINE II=1
        wide_bus_t temp = *reinterpret_cast<wide_bus_t*>(&row_indices_B[i]);
        LOAD_ROW_INDICES_B_LOOP_INNER:
        for(int ii=0; ii<ELEMS_PER_BUS_FOR_INT; ii++){
        #pragma HLS UNROLL
            if((i+ii) < TOTAL) {
                ap_uint<32> bits = temp.range(32*(ii+1)-1, 32*ii);
                row_indices_B_local[i+ii] = *reinterpret_cast<int*>(&bits);
            }
        }
    }
}

void LOAD_COL_PTR_B(
    int col_ptr_B[K + 1],
    int col_ptr_B_local[K + 1]
){
    const int TOTAL = K + 1;
    LOAD_COL_PTR_B_LOOP:
    for(int i=0; i<TOTAL; i+=ELEMS_PER_BUS_FOR_INT){
    #pragma HLS PIPELINE II=1
        wide_bus_t temp = *reinterpret_cast<wide_bus_t*>(&col_ptr_B[i]);
        LOAD_COL_PTR_B_LOOP_INNER:
        for(int ii=0; ii<ELEMS_PER_BUS_FOR_INT; ii++){
        #pragma HLS UNROLL
            if((i+ii) < TOTAL) {
                ap_uint<32> bits = temp.range(32*(ii+1)-1, 32*ii);
                col_ptr_B_local[i+ii] = *reinterpret_cast<int*>(&bits);
            }
        }
    }
}








//---------------------------
// 解压矩阵A和B到密集矩阵
//---------------------------
void DECOMPRESS_MATRICES(
    data_t values_A_local[N * M], int column_indices_A_local[N * M], int row_ptr_A_local[N + 1],
    data_t values_B_local[M * K], int row_indices_B_local[M * K], int col_ptr_B_local[K + 1],
    data_t A_dense[N][M], data_t B_dense[M][K]
) {
    // 初始化A_dense为0
    INIT_A_DENSE:
    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
        for (int j = 0; j < M; j++) {
        #pragma HLS UNROLL factor=64
            A_dense[i][j] = 0;
        }
    }
    // 初始化B_dense为0
    INIT_B_DENSE:
    for (int i = 0; i < M; i++) {
    #pragma HLS PIPELINE II=1
        for (int j = 0; j < K; j++) {
        #pragma HLS UNROLL factor=64
            B_dense[j][i] = 0;
        }
    }
    FILL_DENSE:
    for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor = 64
        // #pragma DATAFLOW
        // #pragma HLS LOOP_FLATTEN off = true
        // #pragma HLS PIPELINE off = true
        int start_A = row_ptr_A_local[i];
        int end_A = row_ptr_A_local[i + 1];
        int start_B = col_ptr_B_local[i];
        int end_B = col_ptr_B_local[i + 1];
        for (int k = start_A; k < end_A; k++) {
            #pragma HLS PIPELINE
            A_dense[i][column_indices_A_local[k]] = values_A_local[k];
        }
        for (int k = start_B; k < end_B; k++) {
            #pragma HLS PIPELINE
            B_dense[row_indices_B_local[k]][i] = values_B_local[k];
        }
    }

    // 填充A_dense的非零元素
    // FILL_A_DENSE:
    // for (int i = 0; i < N; i++) {
    //     #pragma HLS UNROLL facotr = 64
    //     // #pragma HLS LOOP_FLATTEN off = true
    //     // #pragma HLS PIPELINE off = true
    //     int start = row_ptr_A_local[i];
    //     int end = row_ptr_A_local[i + 1];
    //     for (int k = start; k < end; k++) {
    //         // #pragma HLS UNROLL off = true
    //         // #pragma HLS LOOP_FLATTEN off = true
    //         // #pragma HLS PIPELINE
    //         // int col = column_indices_A_local[k];
    //         A_dense[i][column_indices_A_local[k]] = values_A_local[k];
            
    //     }
    // }
    // // 填充B_dense的非零元素
    // FILL_B_DENSE:
    // for (int j = 0; j < K; j++) {
    //     #pragma HLS UNROLL facotr = 64
    //     // #pragma HLS LOOP_FLATTEN off = true
    //     // #pragma HLS PIPELINE off = true
    //     int start = col_ptr_B_local[j];
    //     int end = col_ptr_B_local[j + 1];
    //     for (int k = start; k < end; k++) {
    //         // #pragma HLS UNROLL off = true
    //         // #pragma HLS LOOP_FLATTEN off = true
    //         // #pragma HLS PIPELINE
    //         // int row = row_indices_B_local[k];
    //         B_dense[row_indices_B_local[k]][j] = values_B_local[k];
    //     }
    // }
}

void DENSE_MATMUL(
    data_t A[N][M], data_t B[M][K], data_t C[N][K]
) {
    MATMUL_LOOP:
    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS UNROLL off = true
    #pragma HLS LOOP_FLATTEN off = true
    // #pragma HLS PIPELINE off = true
        for (int j = 0; j < K; j++) {
            // ap_fixed<32, 8> sum = 0;
            data_t sum = 0;
            for (int l = 0; l < M; l++) {
            // #pragma HLS UNROLL factor=64
                sum += A[i][l] * B[l][j];
            }
            C[i][j] = sum;
        }
    }
}











// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], 
                                data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t C[N][K])
{
#pragma HLS interface m_axi port=values_A offset=slave bundle=mem1 max_widen_bitwidth = 512 
#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1 max_widen_bitwidth = 512
#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1 max_widen_bitwidth = 512 //alignment_byte_size=512

#pragma HLS interface m_axi port=values_B offset=slave bundle=mem2 max_widen_bitwidth = 512
#pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2 max_widen_bitwidth = 512
#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2 max_widen_bitwidth = 512

#pragma HLS interface m_axi port=C offset=slave bundle=mem3

#pragma HLS interface s_axilite port=return


    // Copy the arrays into local memory
    data_t values_A_local[N * M];
    int column_indices_A_local[N * M];
    int row_ptr_A_local[N + 1];

    data_t values_B_local[M * K];
    int row_indices_B_local[M * K];
    int col_ptr_B_local[M + 1];

    // data_t values_B_local_t[M * K];
    // int row_indices_B_local_t[M * K];
    // int col_ptr_B_local_t[M + 1];

    data_t C_local[N][K] = {0};
    // ARRAY PARTITION
    #pragma HLS ARRAY_PARTITION variable=values_A_local cyclic factor=128
    #pragma HLS ARRAY_PARTITION variable=column_indices_A_local cyclic factor=128
    #pragma HLS ARRAY_PARTITION variable=row_ptr_A_local cyclic factor=32

    #pragma HLS ARRAY_PARTITION variable=values_B_local cyclic factor=128
    #pragma HLS ARRAY_PARTITION variable=row_indices_B_local cyclic factor=128
    #pragma HLS ARRAY_PARTITION variable=col_ptr_B_local cyclic factor=32
    
    // #pragma HLS ARRAY_PARTITION variable=values_B_local_t cyclic factor=128
    // #pragma HLS ARRAY_PARTITION variable=row_indices_B_local_t cyclic factor=32
    // #pragma HLS ARRAY_PARTITION variable=col_ptr_B_local_t cyclic factor=32

    #pragma HLS ARRAY_PARTITION variable=C_local cyclic factor=32 dim=2

    // Load the arrays into local memory

    LOAD_VALUES_A(values_A, values_A_local);
    LOAD_VALUES_B(values_B, values_B_local);
    LOAD_COLUMN_INDICES_A(column_indices_A, column_indices_A_local);
    LOAD_ROW_INDICES_B(row_indices_B, row_indices_B_local);
    LOAD_ROW_PTR_A(row_ptr_A, row_ptr_A_local);
    LOAD_COL_PTR_B(col_ptr_B, col_ptr_B_local);

    // // Step 1, Loop CSR matrix A along its Row
    // Step_1_For_SpM: for (unsigned int  A_row = 0; A_row < N; A_row++) {
    //     #pragma HLS UNROLL off = true
    //     #pragma HLS LOOP_FLATTEN off = true
    //     #pragma HLS PIPELINE off = true
    //     ap_uint<32> index_for_A = row_ptr_A_local[A_row];
    //     ap_uint<32> ending_index_for_A = row_ptr_A_local[A_row + 1];
    //     // Step 2, Loop CSR matrix B along its Column
    //     Step_2_For_SpM: for (unsigned int  gen_var_parallel_blocks = 0; gen_var_parallel_blocks < number_of_parallel_blocks; gen_var_parallel_blocks++){
    //         #pragma HLS UNROLL
    //         Partition_For_Step_2_For_SpM: for (unsigned int  B_column = gen_var_parallel_blocks*(M/number_of_parallel_blocks); B_column < (gen_var_parallel_blocks+1)*(M/number_of_parallel_blocks); B_column++) {
    //             #pragma HLS UNROLL off = true
    //             #pragma HLS LOOP_FLATTEN off = true
    //             #pragma HLS PIPELINE off = true
    //             ap_uint<32> index_for_B = col_ptr_B_local[B_column];
    //             ap_uint<32> ending_index_for_B = col_ptr_B_local[B_column + 1];
                
    //             // Step 3, Loop through the elements in the row of A and the column of B
    //             ap_uint<32> idx_A = index_for_A;
    //             ap_uint<32> idx_B = index_for_B;
    //             // clear terminal
    //             // printf("\033[2J\033[1;1H");
    //             // printf("idx_A %d, idx_B %d, ending_index_for_A %d, ending_index_for_B %d   \n", idx_A, idx_B, ending_index_for_A, ending_index_for_B);
    //             ap_fixed<32, 8> temp_sum = 0;
    //             // #pragma HLS DEPENDENCE variable=column_indices_A_local Inter false
    //             // #pragma HLS DEPENDENCE variable=row_indices_B_local Inter false
    //             // #pragma HLS DEPENDENCE variable=values_A_local Inter false
    //             // #pragma HLS DEPENDENCE variable=values_B_local Inter false
    //             AB_Iteration_For_SpM: while (true) {
    //                 #pragma HLS PIPELINE
    //                 if(idx_A == ending_index_for_A || idx_B == ending_index_for_B){
    //                     break;
    //                 }
    //                 // // printf("|idx_A %d, idx_B %d|columnA %d, rowB %d|", idx_A, idx_B, column_indices_A_local[idx_A], row_indices_B_local[idx_B]);
    //                 // // current idx a and idx b is
    //                 // if(column_indices_A_local[idx_A] == row_indices_B_local[idx_B]){
    //                 //     temp_sum += values_A_local[idx_A] * values_B_local[idx_B];
    //                 //     idx_A=idx_A+1;
    //                 //     idx_B=idx_B+1;
    //                 // }
    //                 // else if(column_indices_A_local[idx_A] < row_indices_B_local[idx_B]){
    //                 //     idx_A=idx_A+1;
    //                 // }
    //                 // else{
    //                 //     idx_B=idx_B+1;
    //                 // }
    //                 // printf("\n");
    //                 idx_A+=1;
    //                 idx_B+=1;
    //                 temp_sum += values_B_local[idx_B];
    //             }
    //             C_local[A_row][B_column] = temp_sum;
    //         }
    //     }
    // }
    // // // data_t C_local_ref[N][K] = {0};

    // // Perform Sparse x Sparse Multiplication
    // for (int i = 0; i < N; i++) {
    //     for (int idx_A = row_ptr_A_local[i]; idx_A < row_ptr_A_local[i + 1]; idx_A++) {
    //         int k = column_indices_A_local[idx_A]; // Column index of A
    //         data_t value_A = values_A_local[idx_A];

    //         // Iterate over columns of B corresponding to row k
    //         for (int idx_B = col_ptr_B_local_t[k]; idx_B < col_ptr_B_local_t[k + 1]; idx_B++) {
    //             int j = row_indices_B_local_t[idx_B]; // Row index of B
    //             data_t value_B = values_B_local_t[idx_B];

    //             // Accumulate the product into C[i][j]
    //             C_local[i][j] += value_A * value_B;
    //         }
    //     }
    // }

    // // Compare the results of the reference implementation with the optimized implementation
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < K; j++) {
    //         if (C_local[i][j] != C_local_ref[i][j]) {
    //             printf("Mismatch at (%d, %d): %.9f != %.9f\n", i, j, C_local[i][j]*1000, C_local_ref[i][j]*1000);
    //         }
    //     }
    // }

    data_t A_dense[N][M];// = {0};
    data_t B_dense[M][K];// = {0};
    #pragma HLS ARRAY_PARTITION variable=A_dense cyclic factor=64 dim=2
    #pragma HLS ARRAY_PARTITION variable=B_dense cyclic factor=64 dim=1

    DECOMPRESS_MATRICES(
        values_A_local, column_indices_A_local, row_ptr_A_local,
        values_B_local, row_indices_B_local, col_ptr_B_local,
        A_dense, B_dense
    );

    // // Perform Sparse x Sparse Multiplication
    // for (int i = 0; i < N; i++) {
    //     #pragma HLS UNROLL factor = 16
    //     for (int idx_A = row_ptr_A_local[i]; idx_A < row_ptr_A_local[i + 1]; idx_A++) {
    //         int k = column_indices_A_local[idx_A]; // Column index of A
    //         data_t value_A = values_A_local[idx_A];

    //         // Iterate over columns of B corresponding to row k
    //         for (int idx_B = col_ptr_B_local[k]; idx_B < col_ptr_B_local[k + 1]; idx_B++) {
    //             int j = row_indices_B_local[idx_B]; // Row index of B
    //             data_t value_B = values_B_local[idx_B];

    //             // Accumulate the product into C[i][j]
    //             C[i][j] += value_A * value_B;
    //         }
    //     }
    // }

    DENSE_MATMUL(A_dense, B_dense, C_local);

    STORE_OUTPUT(C_local, C);
}

