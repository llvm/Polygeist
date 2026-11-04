// #include "ap_cint.h"
#include "syrk.h"

void syrk(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C_out[N][N]) {
	int i, j, k;

	DATA_TYPE buff_A0[N][N], buff_A1[N][N];
	DATA_TYPE buff_B[N][N];
	DATA_TYPE buff_C_out[N][N];

	lprd_1: for (i = 0; i < N; i++) {
		lprd_2: for (j = 0; j < N; j++) {
			buff_A0[i][j] = A[i][j];
			buff_A1[i][j] = A[i][j];
			buff_B[i][j] = B[i][j];
			buff_C_out[i][j] = 0;
		}
	}

	lp1: for (i = 0; i < N; i++) {
		lp2: for (j = 0; j < N; j++) {
			// if (j > i) continue;
			lp3: for (k = 0; k < N; k++) {
				buff_C_out[i][j] += alpha * buff_A0[i][k] * buff_A1[j][k];
			}
		}
	}

	lp4: for (i = 0; i < N; i++) {
		lp5: for (j = 0; j < N; j++) {
			// if (j > i) continue;
			buff_C_out[i][j] += beta * buff_B[i][j];
		}
	}

	

	lpwr_1: for(i = 0; i < N; i++) {
		lpwr_2: for(j = 0; j < N; j++) {
			if (j > i) C_out[i][j] = 0;
			else C_out[i][j] = buff_C_out[i][j];
		}
	}
}

