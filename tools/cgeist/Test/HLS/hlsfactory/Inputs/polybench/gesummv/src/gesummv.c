// #include "ap_cint.h"
#include "gesummv.h"

void gesummv(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE x[N], DATA_TYPE y_out[N])
{
    int i, j;
	DATA_TYPE buff_A[N][N];
	DATA_TYPE buff_B[N][N];
	DATA_TYPE buff_x[N];
	DATA_TYPE buff_y_out[N];
	DATA_TYPE tmp1[N];
	DATA_TYPE tmp2[N];

	lprd_1: for(i = 0; i < N; i++) {
		buff_x[i] = x[i];
		tmp1[i] = 0;
		tmp2[i] = 0;
		buff_y_out[i] = 0;
		lprd_2: for(j = 0; j < N; j++) {
			buff_A[i][j] = A[i][j];
			buff_B[i][j] = B[i][j];
		}
	}

    lp1: for(i = 0; i < N; i++) {
        lp2: for(j = 0; j < N; j++) {
	        tmp1[i] += alpha * buff_A[i][j] * buff_x[j];
        }
    }

	lp3: for(i = 0; i < N; i++) {
        lp4: for(j = 0; j < N; j++) {
	        tmp2[i] += beta * buff_B[i][j] * buff_x[j];
        }
    }

	lp5: for(i = 0; i < N; i++) {
		buff_y_out[i] = tmp1[i] + tmp2[i];
	}

	lpwr: for(i = 0; i < N; i++) {
		y_out[i] = buff_y_out[i];
	}
}
