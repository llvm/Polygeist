#include "atax.h"

void atax(DATA_TYPE A[N][N], DATA_TYPE x[N], DATA_TYPE y_out[N])
{
    int i, j;
    DATA_TYPE buff_A[N][N];
    DATA_TYPE buff_x[N];
    DATA_TYPE buff_y_out[N];
    DATA_TYPE tmp1[N];

    lprd_1: for (i = 0; i < N; i++) {
        buff_x[i] = x[i];
    	buff_y_out[i] = 0;
        tmp1[i] = 0;
    	lprd_2: for (j = 0; j < N; j++) {
    		buff_A[i][j] = A[i][j];
    	}
    }

    lp1: for (i = 0; i < N; i++) {
        lp2: for (j = 0; j < N; j++) {
	        tmp1[i] = tmp1[i] + buff_A[i][j] * buff_x[j];
        }
    }

    lp3: for (i = 0; i < N; i++) {
        lp4: for (j = 0; j < N; j++) {
        	buff_y_out[j] = buff_y_out[j] + buff_A[i][j] * tmp1[i];
        }
    }

    lpwr_1: for (i = 0; i < N; i++) {
        y_out[i] = buff_y_out[i];
    }
}
