// #include "ap_cint.h"
#include "syr2k.h"

void syr2k(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N], DATA_TYPE D_out[N][N]){
	int i, j, k;
	DATA_TYPE buff_A0[N][N], buff_A1[N][N];
	DATA_TYPE buff_B0[N][N], buff_B1[N][N];
	DATA_TYPE buff_C[N][N];
	DATA_TYPE buff_D_out[N][N];
	DATA_TYPE tmp1[N][N];
	DATA_TYPE tmp2[N][N];


	lprd_1: for (i = 0; i < N; i++){
		lprd_2: for (j = 0; j < N; j++){
			buff_A0[i][j] = A[i][j];
			buff_A1[i][j] = A[i][j];
			buff_B0[i][j] = B[i][j];
			buff_B1[i][j] = B[i][j];
			buff_C[i][j] = C[i][j];
			buff_D_out[i][j] = 0;
			tmp1[i][j] = 0;
			tmp2[i][j] = 0;
		}
	}

	lp1: for (i = 0; i < N; i++){
		lp2: for (j = 0; j < N; j++){
			// if (j > i) continue;
			lp3: for (k = 0; k < N; k++){
				tmp1[i][j] += alpha * buff_A0[i][k] * buff_B0[j][k];
			}
		}
	}

	lp4: for (i = 0; i < N; i++){
		lp5: for (j = 0; j < N; j++){
			// if (j > i) continue;
			lp6: for (k = 0; k < N; k++){
				tmp2[i][j] += alpha * buff_B0[i][k] * buff_A1[j][k];
			}
		}
	}

	lp7: for (i = 0; i < N; i++){
		lp8: for (j = 0; j < N; j++){
			//if (j > i) continue;
			buff_D_out[i][j] = tmp1[i][j] + tmp2[i][j] + beta * buff_C[i][j];
		}
	}

	lpwr_1: for(i = 0; i < N; i++){
		lpwr_2: for(j = 0; j < N; j++){
			if (j > i) D_out[i][j] = 0;
			else D_out[i][j] = buff_D_out[i][j];
		}
	}
}

