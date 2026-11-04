#include "flat.h"
#include "hls_math.h"

#define SYSTOLIC_DIM 16

void Inter_Softmax(data_t logit[QUERY_LENGTH_F][KEY_LENGTH_T], data_t softmax[QUERY_LENGTH_F][KEY_LENGTH_T], data_t max_arr[QUERY_LENGTH_F])
{
	for (int f = 0; f < QUERY_LENGTH_F; ++f) {
		data_t buffer[KEY_LENGTH_T];
		data_t max = max_arr[f];
		data_t sum = 0;
		for (int t = 0; t < KEY_LENGTH_T; ++t) {
			buffer[t] = exp(logit[f][t] - max);
			sum += buffer[t];
		}
		if (sum == 0) sum = 1;
		for (int t = 0; t < KEY_LENGTH_T; ++t) {
			//std::cout << sum << std::endl;
			softmax[f][t] = buffer[t] / sum;
		}
	}
	//softmax = logit;
}

// Key weight stationary
void computeLogit(data_t query_matrix[QUERY_LENGTH_F][KEY_LENGTH_T], data_t key_matrix[QUERY_LENGTH_F][KEY_LENGTH_T],
					data_t bias_matrix[QUERY_LENGTH_F][KEY_LENGTH_T], data_t logit[QUERY_LENGTH_F][KEY_LENGTH_T], data_t max[QUERY_LENGTH_F])
{

// #pragma HLS ARRAY_PARTITION variable=key_matrix dim=0 type=complete
// #pragma HLS ARRAY_PARTITION variable=query_matrix dim=0 type=cyclic factor=4
// #pragma HLS ARRAY_PARTITION variable=bias_matrix dim=0 type=cyclic factor=4
// #pragma HLS ARRAY_PARTITION variable=logit dim=0 type=complete

	   data_t local_out[QUERY_LENGTH_F][KEY_LENGTH_T];

#pragma HLS ARRAY_PARTITION variable=local_out dim=0 type=cyclic factor=4
// #pragma HLS ARRAY_PARTITION variable=max dim=1 type=complete

    int maxf = QUERY_LENGTH_F+KEY_LENGTH_T-2;
    // Systolic array is TxH, iterations are F
    systolic_f:
    for (int f = 0; f < QUERY_LENGTH_F + maxf; ++f) {
//#pragma HLS PIPELINE OFF
//#pragma HLS UNROLL factor=1
//#pragma HLS LOOP_FLATTEN OFF

    	systolic_t_outer:
    	for (int t0 = QUERY_LENGTH_F/SYSTOLIC_DIM; t0 > 0; --t0) {
#pragma HLS UNROLL factor=16
    		systolic_h_outer:
    		for (int h0 = KEY_LENGTH_T/SYSTOLIC_DIM; h0 > 0; --h0) {
#pragma HLS UNROLL factor=16

    			systolic_t_inner:
    			for (int t = t0*SYSTOLIC_DIM-1; t > (t0-1)*SYSTOLIC_DIM-1; --t) {
    				systolic_h_inner:
    				for (int h = h0*SYSTOLIC_DIM-1; h > (h0-1)*SYSTOLIC_DIM-1; --h) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN

					data_t query = query_matrix[(f-t-h) % QUERY_LENGTH_F][h];
					data_t prev_sum = (h == 0) ? bias_matrix[(f-t) % QUERY_LENGTH_F][t] : local_out[t][(h-1) % KEY_LENGTH_T];

					data_t val;
#pragma HLS BIND_OP variable=val op=mul impl=dsp latency=-1
					val = query * key_matrix[t][h] + prev_sum;
					local_out[t][h] = val;

					unsigned findex = f-t-QUERY_LENGTH_F+1;
					if (h == KEY_LENGTH_T-1 && findex <= KEY_LENGTH_T-1) {
						// Write back
						logit[findex % QUERY_LENGTH_F][t] = val;
						// Update max
						//std::cout << max[findex] << " " << findex <<  std::endl;
						max[findex] = (t == 0) ? val : ((val > max[findex]) ? val : max[findex]);
					}
    				}
    			}
            }
        }
    }
}


// Value weight stationary
void computeAttention(data_t logit[QUERY_LENGTH_F][KEY_LENGTH_T], data_t value_matrix[QUERY_LENGTH_F][KEY_LENGTH_T], data_t output[QUERY_LENGTH_F][KEY_LENGTH_T])
{
// #pragma HLS ARRAY_PARTITION variable=logit dim=0 type=complete
// #pragma HLS ARRAY_PARTITION variable=value_matrix dim=0 type=complete
// #pragma HLS ARRAY_PARTITION variable=output dim=0 type=complete

data_t local_out[QUERY_LENGTH_F][KEY_LENGTH_T];

#pragma HLS ARRAY_PARTITION variable=local_out dim=0 type=cyclic factor=4

int maxk = QUERY_LENGTH_F+KEY_LENGTH_T-2;
systolic1:
    for (int k = 0; k < KEY_LENGTH_T + maxk; ++k) {
//#pragma HLS PIPELINE off
//#pragma HLS UNROLL factor=1
//#pragma HLS LOOP_FLATTEN OFF

	systolic2_outer:
		for (int i0 = QUERY_LENGTH_F/SYSTOLIC_DIM; i0 > 0; --i0) {
#pragma HLS UNROLL factor=16
		systolic3_outer:
			for (int j0 = KEY_LENGTH_T/SYSTOLIC_DIM; j0 > 0; --j0) {
#pragma HLS UNROLL factor=16

				systolic2_inner:
				for (int i = i0*SYSTOLIC_DIM-1; i > (i0-1)*SYSTOLIC_DIM-1; --i) {
					systolic3_inner:
					for (int j = j0*SYSTOLIC_DIM-1; j > (j0-1)*SYSTOLIC_DIM-1; --j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN

						data_t a_val = logit[i][(k-i-j) % KEY_LENGTH_T];
						data_t b_val = value_matrix[(k-i-j) % QUERY_LENGTH_F][j];
						data_t last = (j == 0) ? (data_t) 0 : local_out[i][(j-1) % KEY_LENGTH_T];

						data_t val;
#pragma HLS BIND_OP variable=val op=mul impl=dsp latency=-1
						val = last + a_val * b_val;
						local_out[i][j] = val;

						unsigned kindex = k-i-QUERY_LENGTH_F+1;
						if (j == KEY_LENGTH_T-1 && kindex <= KEY_LENGTH_T-1) {
							// Write back
							output[kindex % QUERY_LENGTH_F][i] = val;
						}
					}
				}
            }
        }
    }
}

// Output stationary
// static void computeAttentionOSTATIONARY(data_t logit[ROW][COLUMN], data_t value_matrix[ROW][COLUMN], data_t output[ROW][COLUMN])
// {

// #pragma HLS ARRAY_PARTITION variable=value_matrix dim=1 type=complete
// #pragma HLS ARRAY_PARTITION variable=output dim=0 type=complete

// int maxk = ROW+COLUMN-2;
// systolic1:
//     for (int k = 0; k < COLUMN + maxk; ++k) {
// #pragma HLS PIPELINE off
// #pragma HLS UNROLL factor=1
// #pragma HLS LOOP_FLATTEN OFF

// 	systolic2_outer:
// 		for (int i0 = ROW/SYSTOLIC_DIM; i0 > 0; --i0) {
// #pragma HLS UNROLL factor=16
// 		systolic3_outer:
// 			for (int j0 = COLUMN/SYSTOLIC_DIM; j0 > 0; --j0) {
// #pragma HLS UNROLL factor=16

// 				systolic2_inner:
// 				for (int i = i0*SYSTOLIC_DIM-1; i > (i0-1)*SYSTOLIC_DIM-1; --i) {
// #pragma HLS PIPELINE II=1
// 					systolic3_inner:
// 					for (int j = j0*SYSTOLIC_DIM-1; j > (j0-1)*SYSTOLIC_DIM-1; --j) {

// 						bool active = (k-i-j >= 0 && k-i-j < ROW);
// 						data_t a_val = (active) ? logit[i][k-i-j] : (data_t) 0;
// 						data_t b_val = (active) ? value_matrix[k-i-j][j] : (data_t) 0;
// 						data_t last = (k == 0) ? (data_t) 0 : output[i][j];
// #pragma HLS BIND_OP variable=output op=mul impl=dsp latency=-1
// 						output[i][j] = last + a_val * b_val;

// 					}
// 				}
//             }
//         }
//     }
// }

// // Output stationary
// static void computeAttentionORIGINAL(data_t logit[ROW][COLUMN], data_t value_matrix[ROW][COLUMN], data_t output[ROW][COLUMN])
// {

// #pragma HLS ARRAY_PARTITION variable=value_matrix dim=1 type=complete
// #pragma HLS ARRAY_PARTITION variable=output dim=0 type=complete

// systolic1:
//     for (int k = 0; k < COLUMN; k++) {
// #pragma HLS PIPELINE off
// #pragma HLS UNROLL factor=1
// #pragma HLS LOOP_FLATTEN OFF
//     systolic2:
//         for (int i = 0; i < ROW; i++) {
// #pragma HLS UNROLL factor=16
//         systolic3:
//             for (int j = 0; j < ROW; j++) {
// #pragma HLS UNROLL factor=16
//                 data_t last = (k == 0) ? 0 : output[i][j];
//                 data_t a_val = (i < ROW && k < COLUMN) ? logit[i][k] : 0;
//                 data_t b_val = (k < ROW && j < COLUMN) ? value_matrix[k][j] : 0;
// #pragma HLS BIND_OP variable=output op=mul impl=dsp latency=-1
//                 output[i][j] = last + a_val * b_val;
//             }
//         }
//     }
// }