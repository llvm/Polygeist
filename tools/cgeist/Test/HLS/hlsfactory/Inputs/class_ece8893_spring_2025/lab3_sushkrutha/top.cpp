#include "dcl.h"

// Load CSR matrix A with controlled partitioning
void load_A(
    data_t values_A[N*M], int column_indices_A[N*M], int row_ptr_A[N+1],
    data_t local_values_A[N][M], int local_col_idx_A[N][M], int local_row_ptr_A[N+1])
{
  //  #pragma HLS performance target_tl= 5000
    
    // Partition row pointers
    #pragma HLS array_partition variable=local_row_ptr_A complete
    
    LOAD_A_ROW_PTR: for (int i = 0; i <= N; i++) {
        #pragma HLS pipeline II=1
        local_row_ptr_A[i] = row_ptr_A[i];
    }
    
    // Process rows with limited parallelism
    LOAD_A_ROWS: for (int i = 0; i < N; i++) {
        #pragma HLS pipeline II=1
        int row_start = local_row_ptr_A[i];
        int row_end = local_row_ptr_A[i+1];
        
        LOAD_A_ELEMENTS: for (int p = 0; p < row_end - row_start; p++) {
            #pragma HLS pipeline II=1
            local_values_A[i][p] = values_A[row_start + p];
            local_col_idx_A[i][p] = column_indices_A[row_start + p];
        }
    }
}

// Load CSC matrix B with controlled partitioning
void load_B(
    data_t values_B[M*K], int row_indices_B[M*K], int col_ptr_B[M+1],
    data_t local_values_B[M][K], int local_row_idx_B[M][K], int local_col_ptr_B[M+1])
{
    #pragma HLS performance target_tl=5000
    
    // Partition column pointers
    #pragma HLS array_partition variable=local_col_ptr_B cyclic factor=4
    
    LOAD_B_COL_PTR: for (int k = 0; k <= M; k++) {
        #pragma HLS pipeline II=1
        local_col_ptr_B[k] = col_ptr_B[k];
    }
    
    // Process columns with limited parallelism
    LOAD_B_COLS: for (int k = 0; k < M; k++) {
        #pragma HLS pipeline II=1
        int col_start = local_col_ptr_B[k];
        int col_end = local_col_ptr_B[k+1];
        
        LOAD_B_ELEMENTS: for (int p = 0; p < col_end - col_start; p++) {
            #pragma HLS pipeline II=1
            local_values_B[k][p] = values_B[col_start + p];
            local_row_idx_B[k][p] = row_indices_B[col_start + p];
        }
    }
}

// Core computation with controlled unrolling
void compute_C(
    data_t local_values_A[N][M], int local_col_idx_A[N][M], int local_row_ptr_A[N+1],
    data_t local_values_B[M][K], int local_row_idx_B[M][K], int local_col_ptr_B[M+1],
    data_t local_C[N][K])
{
    #pragma HLS performance target_tl=5000
    
    // Partition output matrix columns (limited to 4-way)
    #pragma HLS array_partition variable=local_C cyclic factor=4 dim=2
    
    INIT_C: for (int i = 0; i < N; i++) {
        #pragma HLS pipeline II=1
        for (int j = 0; j < K; j++) {
            
            local_C[i][j] = 0;
        }
    }

    ROW_LOOP: for (int i = 0; i < N; i++) {
        #pragma HLS pipeline II=1
        
        int row_start = local_row_ptr_A[i];
        int row_end = local_row_ptr_A[i+1];
        
        NNZ_A_LOOP: for (int p = row_start; p < row_end; p++) {
            #pragma HLS pipeline II=1
            int k = local_col_idx_A[i][p - row_start];
            data_t val_A = local_values_A[i][p - row_start];
            
            int col_start = local_col_ptr_B[k];
            int col_end = local_col_ptr_B[k+1];
            
            NNZ_B_LOOP: for (int q = col_start; q < col_end; q++) {
                #pragma HLS pipeline II=1
                int j = local_row_idx_B[k][q - col_start];
                data_t val_B = local_values_B[k][q - col_start];
                
                // Use manual accumulation to avoid dependency issues
                data_t product = val_A * val_B;
                local_C[i][j] += product;
            }
        }
    }
}

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
    
    // Local buffers with controlled partitioning
    data_t local_values_A[N][M];
    int local_col_idx_A[N][M];
    int local_row_ptr_A[N+1];
    
    data_t local_values_B[M][K];
    int local_row_idx_B[M][K];
    int local_col_ptr_B[M+1];
    
    data_t local_C[N][K];
    #pragma HLS array_partition variable=local_C cyclic factor=4 dim=2

    // Execute pipeline stages
    load_A(values_A, column_indices_A, row_ptr_A,
          local_values_A, local_col_idx_A, local_row_ptr_A);
    
    load_B(values_B, row_indices_B, col_ptr_B,
          local_values_B, local_row_idx_B, local_col_ptr_B);
    
    compute_C(local_values_A, local_col_idx_A, local_row_ptr_A,
             local_values_B, local_row_idx_B, local_col_ptr_B,
             local_C);
    
    // Write back results
    WRITE_BACK: for (int i = 0; i < N; i++) {
        #pragma HLS pipeline II=1
        for (int j = 0; j < K; j++) {
           
            C[i][j] = local_C[i][j];
        }
    }
}
