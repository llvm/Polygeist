#include "dcl.h"

constexpr int D = 8; 

void load_A(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], 
            data_t values_A_l[D][N * M], int column_indices_A_l[D][N * M], int row_ptr_A_l[D][N + 1]) {
	#pragma HLS array_partition variable=values_A_l complete dim=1
	#pragma HLS array_partition variable=column_indices_A_l complete dim=1
	#pragma HLS array_partition variable=row_ptr_A_l complete dim=1
	for(int i=0; i<(N+1); i++) {
		#pragma HLS pipeline
		int row_p_A = row_ptr_A[i];
		for (int d=0; d<D; d++) {
        	row_ptr_A_l[d][i] = row_p_A;
		}
	}
	for(int i=0; i<row_ptr_A_l[0][N]; i++) {
		#pragma HLS pipeline
		data_t val_A = values_A[i];
		for (int d=0; d<D; d++) {
        	values_A_l[d][i] = val_A;
		}
	}
	for(int i=0; i<row_ptr_A_l[0][N]; i++) {
		#pragma HLS pipeline
		int col_ind_A = column_indices_A[i];
		for (int d=0; d<D; d++) {
        	column_indices_A_l[d][i] = col_ind_A;
		}
	}
}	

void load_B(data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], 
            data_t values_B_l[D][M * K], int row_indices_B_l[D][M * K], int col_ptr_B_l[D][M + 1]) {
	#pragma HLS array_partition variable=values_B_l complete dim=1
	#pragma HLS array_partition variable=row_indices_B_l complete dim=1
	#pragma HLS array_partition variable=col_ptr_B_l complete dim=1
	for(int i=0; i<(M+1); i++) {
		#pragma HLS pipeline
        int col_p_B = col_ptr_B[i];
		for (int d=0; d<D; d++) {
        	col_ptr_B_l[d][i] = col_p_B;
		}
	}
	for(int i=0; i<col_ptr_B_l[0][M]; i++) {
		#pragma HLS pipeline
        data_t val_B = values_B[i];
		for (int d=0; d<D; d++) {
        	values_B_l[d][i] = val_B;
		}
	}
	for(int i=0; i<col_ptr_B_l[0][M]; i++) {
		#pragma HLS pipeline
        int row_ind_B = row_indices_B[i];
		for (int d=0; d<D; d++) {
        	row_indices_B_l[d][i] = row_ind_B;
		}
	}
}


void spmm(int d, int row_ptr_A_local[N + 1], int column_indices_A_local[N * M], data_t values_A_local[N * M],
									data_t values_B_local[M * K], int row_indices_B_local[M * K], int col_ptr_B_local[M + 1], data_t C_local[N/D][K]) {

	for (int i = 0; i < (N / D); i++) {
         for (int j = 0; j < K; j++) {
             C_local[i][j] = 0;
         }
     }
    for (int i = 0; i < (N / D); i++) {
		int start_A = row_ptr_A_local[i + d * (N / D)];
		int end_A = row_ptr_A_local[i + d * (N / D) + 1];
        for (int idx_A = start_A; idx_A < end_A; idx_A++) {
            int k = column_indices_A_local[idx_A];
            data_t value_A = values_A_local[idx_A];

            int start_B = col_ptr_B_local[k];
            int end_B = col_ptr_B_local[k + 1];
            for (int idx_B = start_B; idx_B < end_B; idx_B++) {
				#pragma HLS pipeline
                int j = row_indices_B_local[idx_B];
                data_t value_B = values_B_local[idx_B];
                C_local[i][j] += value_A * value_B;
            }
        }
    }
}

void write_C(data_t C[N][K], data_t C_l[D][N/D][K]) {
	for (int d=0; d<D; d++) {
    	for (int i = 0; i < (N/D); i++) {
    	    for (int j = 0; j < K; j++) {
    	        C[i+d*(N/D)][j] = C_l[d][i][j];
    	    }
    	}
	}
}

// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
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

#pragma HLS dataflow
	//Local Matrix A Data Structures
	data_t values_A_l[D][N*M];
	int column_indices_A_l[D][N*M];
	int row_ptr_A_l[D][N+1];

	//Local Matrix B Data Structures
	data_t values_B_l[D][M*K];
	int row_indices_B_l[D][M*K];
	int col_ptr_B_l[D][M+1];

	data_t C_l[D][N/D][K];


	
	#pragma HLS array_partition variable=C_l complete dim=1

    load_A(values_A, column_indices_A, row_ptr_A, values_A_l, column_indices_A_l, row_ptr_A_l);
    load_B(values_B, row_indices_B, col_ptr_B, values_B_l, row_indices_B_l, col_ptr_B_l);
    for (int d = 0; d < D; d++) {
		#pragma HLS unroll
    	spmm(d, row_ptr_A_l[d], column_indices_A_l[d], values_A_l[d], values_B_l[d], row_indices_B_l[d], col_ptr_B_l[d], C_l[d]);
	}
    write_C(C, C_l);
}
