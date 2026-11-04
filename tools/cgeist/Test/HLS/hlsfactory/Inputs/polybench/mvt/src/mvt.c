// #include "ap_cint.h"
#include "mvt.h"

void mvt(DATA_TYPE A[N][N], DATA_TYPE x1[N], DATA_TYPE x2[N], DATA_TYPE y1[N], DATA_TYPE y2[N], DATA_TYPE x1_out[N], DATA_TYPE x2_out[N])
{
	int i, j;
	DATA_TYPE buff_A[N][N];
	DATA_TYPE buff_x1[N];
	DATA_TYPE buff_x2[N];
	DATA_TYPE buff_y1[N];
	DATA_TYPE buff_y2[N];

	lprd_1: for (i = 0; i < N; i++) {
		buff_x1[i] = x1[i];
		buff_x2[i] = x2[i];
		buff_y1[i] = y1[i];
		buff_y2[i] = y2[i];
    	lprd_2: for (j = 0; j < N; j++) {
			buff_A[i][j] = A[i][j];
		}
	}

	lp1: for (i = 0; i < N; i++) {
    	lp2: for (j = 0; j < N; j++) {
      		buff_x1[i] = buff_x1[i] + buff_A[i][j] * buff_y1[j];
		}
	}

	lp3: for (i = 0; i < N; i++) {
		lp4: for (j = 0; j < N; j++) {
    		buff_x2[i] = buff_x2[i] + buff_A[j][i] * buff_y2[j];
		}
	}
	
	lpwr: for (i = 0; i < N; i++) {
		x1_out[i] = buff_x1[i];
		x2_out[i] = buff_x2[i];
	}
}
