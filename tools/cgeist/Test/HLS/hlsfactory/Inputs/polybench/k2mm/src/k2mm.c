#include "k2mm.h"

/* D := alpha*A*B*C + beta*D */
void k2mm(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N], DATA_TYPE D[N][N], DATA_TYPE E_out[N][N])
{
    int i, j, k;
    DATA_TYPE buff_A[N][N]; 
    DATA_TYPE buff_B[N][N]; 
    DATA_TYPE buff_C[N][N];
    DATA_TYPE buff_D[N][N];
    DATA_TYPE buff_E_out[N][N];
    DATA_TYPE tmp1[N][N];
    DATA_TYPE tmp2[N][N];

    lprd_1: for (i = 0; i < N; i++){
        lprd_2: for (j = 0; j < N; j++){
            buff_A[i][j] = A[i][j];
            buff_B[i][j] = B[i][j];
            buff_C[i][j] = C[i][j];
            buff_D[i][j] = D[i][j];
            buff_E_out[i][j] = 0;
            tmp1[i][j] = 0;
            tmp2[i][j] = 0;
        }
    }

    lp1: for (i = 0; i < N; i++) {
        lp2: for (j = 0; j < N; j++) {
	        lp3: for (k = 0; k < N; ++k) {
                tmp1[i][j] += alpha * buff_A[i][k] * buff_B[k][j];
            }
        }
    }
    
    lp4: for (i = 0; i < N; i++) {
        lp5: for (j = 0; j < N; j++) {
	        lp6: for (k = 0; k < N; k++) {
                tmp2[i][j] += tmp1[i][k] * buff_C[k][j];
            }
        }
    }

    lp7: for (i = 0; i < N; i++) {
        lp8: for (j = 0; j < N; j++) {
            buff_E_out[i][j] = tmp2[i][j] + beta * buff_D[i][j];
        }
    }


    lpwr_1: for (i = 0; i < N; i++){
        lpwr_2: for (j = 0; j < N; j++){
            E_out[i][j] = buff_E_out[i][j];
        }
    }
}
