#include "systolic.h"

void systolic_array(data_t query_matrix[ROW][COLUMN], data_t key_matrix[ROW][COLUMN],
                data_t value_matrix[ROW][COLUMN], data_t bias_matrix[ROW][COLUMN], data_t output[ROW][COLUMN])
{
    //#pragma HLS INTERFACE mode=axis port=query_matrix depth=50
    //#pragma HLS INTERFACE mode=axis port=key_matrix depth=50
    //#pragma HLS INTERFACE mode=axis port=value_matrix depth=50
    //#pragma HLS INTERFACE mode=axis port=output depth=50

    //#pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t logit[ROW][COLUMN];
    computeLogit(query_matrix, key_matrix, bias_matrix, logit);
    computeAttention(logit, value_matrix, output);
}

// Key weight stationary
void computeLogit(data_t query_matrix[ROW][COLUMN], data_t key_matrix[ROW][COLUMN],
					data_t bias_matrix[ROW][COLUMN], data_t logit[ROW][COLUMN])
{
#pragma HLS INLINE

#pragma HLS ARRAY_PARTITION variable=key_matrix dim=1 type=complete
#pragma HLS ARRAY_PARTITION variable=query_matrix dim=1 type=complete
#pragma HLS ARRAY_PARTITION variable=bias_matrix dim=1 type=complete
//#pragma HLS ARRAY_PARTITION variable=logit dim=1 type=complete

	   data_t local_out[ROW][COLUMN];
	   //data_t local_max[COLUMN];

//#pragma HLS ARRAY_PARTITION variable=local_out dim=2 type=complete
//#pragma HLS ARRAY_PARTITION variable=local_max dim=1 type=complete

    int maxf = ROW+COLUMN-2;
    // Systolic array is TxH, iterations are F
    systolic_f:
    for (int f = 0; f < ROW + maxf; ++f) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
#pragma HLS LOOP_FLATTEN OFF

    	systolic_t_outer:
    	for (int t0 = ROW/SYSTOLIC_DIM; t0 > 0; --t0) {
#pragma HLS UNROLL factor=16
    		systolic_h_outer:
    		for (int h0 = COLUMN/SYSTOLIC_DIM; h0 > 0; --h0) {
#pragma HLS UNROLL factor=16

    			systolic_t_inner:
    			for (int t = t0*SYSTOLIC_DIM-1; t > (t0-1)*SYSTOLIC_DIM-1; --t) {
#pragma HLS PIPELINE II=1
    				systolic_h_inner:
    				for (int h = h0*SYSTOLIC_DIM-1; h > (h0-1)*SYSTOLIC_DIM-1; --h) {
					bool active = (f-t-h >= 0 && f-t-h < ROW);
					data_t query = (active) ? query_matrix[f-t-h][h] : (data_t) 0;
					data_t prev_sum = (active) ? ((h == 0) ? bias_matrix[f-t][t] : local_out[t][h-1]) : (data_t) 0;

#pragma HLS BIND_OP variable=local_out op=mul impl=dsp latency=-1
					local_out[t][h] = query * key_matrix[t][h] + prev_sum;

					int trow = t+ROW-1;
					int findex = f-trow;
					if (h == COLUMN-1 && f >= trow && findex <= COLUMN-1) {
						// Write back
						logit[findex][t] = local_out[t][COLUMN-1];
						// Update max
					}
    				}
    			}
            }
        }
    }
}

// TODO: (IP) THIS STILL HAS HIGH FANOUT FOR INPUTS -> change to weight stationary
// TODO: Value weight stationary
void computeAttention(data_t logit[ROW][COLUMN], data_t value_matrix[ROW][COLUMN], data_t output[ROW][COLUMN])
{
#pragma HLS INLINE

#pragma HLS ARRAY_PARTITION variable=value_matrix dim=1 type=complete
#pragma HLS ARRAY_PARTITION variable=output dim=0 type=complete

data_t local_out[ROW][COLUMN];

//#pragma HLS ARRAY_PARTITION variable=local_out dim=0 type=complete

int maxk = ROW+COLUMN-2;
systolic1:
    for (int k = 0; k < COLUMN + maxk; ++k) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=1
#pragma HLS LOOP_FLATTEN OFF

	systolic2_outer:
		for (int i0 = ROW/SYSTOLIC_DIM; i0 > 0; --i0) {
#pragma HLS UNROLL factor=16
		systolic3_outer:
			for (int j0 = COLUMN/SYSTOLIC_DIM; j0 > 0; --j0) {
#pragma HLS UNROLL factor=16

				systolic2_inner:
				for (int i = i0*SYSTOLIC_DIM-1; i > (i0-1)*SYSTOLIC_DIM-1; --i) {
#pragma HLS PIPELINE II=1
					systolic3_inner:
					for (int j = j0*SYSTOLIC_DIM-1; j > (j0-1)*SYSTOLIC_DIM-1; --j) {


						bool active = (k-i-j >= 0 && k-i-j < ROW);
						data_t a_val = (active) ? logit[i][k-i-j] : (data_t) 0;
						data_t b_val = (active) ? value_matrix[k-i-j][j] : (data_t) 0;
						data_t last = (active) ? ((j == 0) ? (data_t) 0 : local_out[i][j-1]) : (data_t) 0;


#pragma HLS BIND_OP variable=local_out op=mul impl=dsp latency=-1
						local_out[i][j] = last + a_val * b_val;

						int irow = i+ROW-1;
						int kindex = k-irow;
						if (j == COLUMN-1 && k >= irow && kindex <= COLUMN-1) {
							// Write back
							output[kindex][i] = local_out[i][COLUMN-1];
						}
					}
				}
            }
        }
    }
}

// Output stationary
static void computeAttentionOSTATIONARY(data_t logit[ROW][COLUMN], data_t value_matrix[ROW][COLUMN], data_t output[ROW][COLUMN])
{

#pragma HLS ARRAY_PARTITION variable=value_matrix dim=1 type=complete
#pragma HLS ARRAY_PARTITION variable=output dim=0 type=complete

int maxk = ROW+COLUMN-2;
systolic1:
    for (int k = 0; k < COLUMN + maxk; ++k) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=1
#pragma HLS LOOP_FLATTEN OFF

	systolic2_outer:
		for (int i0 = ROW/SYSTOLIC_DIM; i0 > 0; --i0) {
#pragma HLS UNROLL factor=16
		systolic3_outer:
			for (int j0 = COLUMN/SYSTOLIC_DIM; j0 > 0; --j0) {
#pragma HLS UNROLL factor=16

				systolic2_inner:
				for (int i = i0*SYSTOLIC_DIM-1; i > (i0-1)*SYSTOLIC_DIM-1; --i) {
#pragma HLS PIPELINE II=1
					systolic3_inner:
					for (int j = j0*SYSTOLIC_DIM-1; j > (j0-1)*SYSTOLIC_DIM-1; --j) {

						bool active = (k-i-j >= 0 && k-i-j < ROW);
						data_t a_val = (active) ? logit[i][k-i-j] : (data_t) 0;
						data_t b_val = (active) ? value_matrix[k-i-j][j] : (data_t) 0;
						data_t last = (k == 0) ? (data_t) 0 : output[i][j];
#pragma HLS BIND_OP variable=output op=mul impl=dsp latency=-1
						output[i][j] = last + a_val * b_val;

					}
				}
            }
        }
    }
}

// Output stationary
static void computeAttentionORIGINAL(data_t logit[ROW][COLUMN], data_t value_matrix[ROW][COLUMN], data_t output[ROW][COLUMN])
{

#pragma HLS ARRAY_PARTITION variable=value_matrix dim=1 type=complete
#pragma HLS ARRAY_PARTITION variable=output dim=0 type=complete

systolic1:
    for (int k = 0; k < COLUMN; k++) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=1
#pragma HLS LOOP_FLATTEN OFF
    systolic2:
        for (int i = 0; i < ROW; i++) {
#pragma HLS UNROLL factor=16
        systolic3:
            for (int j = 0; j < ROW; j++) {
#pragma HLS UNROLL factor=16
                data_t last = (k == 0) ? 0 : output[i][j];
                data_t a_val = (i < ROW && k < COLUMN) ? logit[i][k] : 0;
                data_t b_val = (k < ROW && j < COLUMN) ? value_matrix[k][j] : 0;
#pragma HLS BIND_OP variable=output op=mul impl=dsp latency=-1
                output[i][j] = last + a_val * b_val;
            }
        }
    }
}
