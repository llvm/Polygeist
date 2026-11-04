#include "gemm.h"

void gemm(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N], DATA_TYPE D_out[N][N])
{
    int i, j, k;

    DATA_TYPE buff_A[N][N]; 
    DATA_TYPE buff_B[N][N]; 
    DATA_TYPE buff_C[N][N];
    DATA_TYPE tmp1[N][N];

    lprd_1: for (i = 0; i < N; i++){
        lprd_2: for (j = 0; j < N; j++){
            buff_A[i][j] = A[i][j];
            buff_B[i][j] = B[i][j];
            buff_C[i][j] = C[i][j];
            tmp1[i][j] = 0;
        }
    }

    lp1: for (i = 0; i < N; i++) {
        lp2: for (j = 0; j < N; j++) {
            lp3: for (k = 0; k < N; k++) {
                tmp1[i][j] = alpha * buff_A[i][k] * buff_B[k][j] + tmp1[i][j];
            }
        }
    }

    lp4: for (i = 0; i < N; i++) {
        lp5: for (j = 0; j < N; j++) {
	        buff_C[i][j] = beta * buff_C[i][j] + tmp1[i][j];
        }
    }

    lpwr_1: for (i = 0; i < N; i++){
        lpwr_2: for (j = 0; j < N; j++){
            D_out[i][j] = buff_C[i][j];
        }
    }
}