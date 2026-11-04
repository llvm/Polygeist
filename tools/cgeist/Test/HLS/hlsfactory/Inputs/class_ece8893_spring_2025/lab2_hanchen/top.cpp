#include "dcl.h"


void softmax_HLS_inline(fixed_t matrix[N]) {
#pragma HLS inline

    fixed_t max_val = matrix[0];
    for (int j = 1; j < N; ++j) {
        if (matrix[j] > max_val) {
            max_val = matrix[j];
        }
    }

    fixed_t sum = 0;
    for (int j = 0; j < N; ++j) {
        matrix[j] = 1 + (matrix[j]-max_val) * \
                (1 + (matrix[j]-max_val)/2);  
        sum += matrix[j];
    }

    for (int j = 0; j < N; ++j) {
        if (sum == 0) {
            matrix[j] = 0;
        }
        else {
            matrix[j] /= sum;
        }
    }
}

void softmax_HLS(fixed_t matrix[B][N][N]) {
// #pragma HLS array_partition variable=matrix dim=1 complete

    for (int b = 0; b < B; ++b) {
    // #pragma HLS pipeline

        for (int i = 0; i < N; ++i) {
        // #pragma HLS unroll

            fixed_t max_val = matrix[b][i][0];
            for (int j = 1; j < N; ++j) {
                if (matrix[b][i][j] > max_val) {
                    max_val = matrix[b][i][j];
                }
            }

            fixed_t sum = 0;
            for (int j = 0; j < N; ++j) {
                matrix[b][i][j] = hls::exp(matrix[b][i][j] - max_val);

                // matrix[b][i][j] = 1 + (matrix[b][i][j]-max_val) * \
                //         (1 + (matrix[b][i][j]-max_val)/2);   //2hls::exp(matrix[b][i][j] - max_val);

                sum += matrix[b][i][j];
            }

            for (int j = 0; j < N; ++j) {
                matrix[b][i][j] /= sum;
            }
        }
    }

// #pragma HLS array_partition variable=matrix dim=1 complete

//     for (int i = 0; i < N; ++i) {
//     #pragma HLS pipeline

//         for (int b = 0; b < B; ++b) {
//         #pragma HLS unroll
        
//             ap_fixed<32, 8> max_val = matrix[b][i][0];


//             for (int j = 1; j < N; ++j) {
//                 if (matrix[b][i][j] > max_val) {
//                     max_val = matrix[b][i][j];
//                 }
//             }

//             ap_fixed<32, 8> sum = 0;
//             for (int j = 0; j < N; ++j) {
//                 // matrix[b][i][j] = hls::exp(matrix[b][i][j] - max_val);
//                 matrix[b][i][j] = 1 + (matrix[b][i][j]-max_val) * \
//                         (1 + (matrix[b][i][j]-max_val)/2);   //2hls::exp(matrix[b][i][j] - max_val);

//                 sum += matrix[b][i][j];
//             }

//             for (int j = 0; j < N; ++j) {
//                 matrix[b][i][j] /= sum;
//             }
//         }
//     }

}


void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
#pragma HLS interface m_axi port=Q offset=slave bundle=mem1
#pragma HLS interface m_axi port=K offset=slave bundle=mem1
#pragma HLS interface m_axi port=V offset=slave bundle=mem1
#pragma HLS interface m_axi port=Output offset=slave bundle=mem2

#pragma HLS interface s_axilite port=return

    // fixed_t scale = 1.0 / sqrt((float)dk);
    
    // fixed_t attention[B][N][N];

    // // // Init
    // // for(int j = 0; j < N; j++) {
    // //     for(int k = 0; k < N; k++) {
    // //     #pragma HLS pipeline
    // //         for(int i = 0; i < B; i++) {
    // //         #pragma HLS unroll
	// // 			attention[i][j][k] = 0;
	// // 		}
	// // 	}
	// // }


    // // #pragma HLS array_partition variable=Q dim=1 complete
    // // #pragma HLS array_partition variable=K dim=1 complete
    // // // #pragma HLS array_partition variable=V dim=1 complete
    // // #pragma HLS array_partition variable=attention dim=1 complete
    // // // #pragma HLS array_partition variable=Output dim=1 complete

    // // Compute Q * K^T
    // for (int b = 0; b < B; ++b) {
    //     for (int i = 0; i < N; ++i) {
    //         for (int j = 0; j < N; ++j) {
    //             ap_fixed<32, 8> sum = 0;
    //             for (int k = 0; k < dk; ++k) {
    //                 sum += Q[b][i][k] * K[b][j][k];
    //             }
    //             attention[b][i][j] = sum * scale;
    //         }
    //     }
    // }

    // // Apply softmax
    // softmax_HLS(attention);

    // // Compute Attention * V
    // for (int b = 0; b < B; ++b) {
    //     for (int i = 0; i < N; ++i) {
    //         for (int j = 0; j < dv; ++j) {
    //             ap_fixed<32, 8> sum = 0;
    //             for (int k = 0; k < N; ++k) {
    //                 sum += attention[b][i][k] * V[b][k][j];
    //             }
    //             Output[b][i][j] = sum;
    //         }
    //     }
    // }


    // // Compute Q * K^T
    // for (int b = 0; b < B; ++b) {
    //     for (int i = 0; i < N; ++i) {
    //         for (int k = 0; k < dk; ++k) {
    //         #pragma HLS pipeline
    //             for (int j = 0; j < N; ++j) {
    //             #pragma HLS unroll
    //                 attention[b][i][j] += Q[b][i][k] * K[b][j][k] * scale;
    //             }
    //         }
    //     }
    // }

    // // Compute Q * K^T
    // for (int j = 0; j < N; ++j) {
    //     for (int i = 0; i < N; ++i) {
    //         for (int k = 0; k < dk; ++k) {
    //         #pragma HLS pipeline
    //             for (int b = 0; b < B; ++b) {
    //             #pragma HLS unroll
    //                 attention[b][i][j] += Q[b][i][k] * K[b][j][k] * scale;
    //                 softmax_HLS_inline(attention[b][i]);
    //             }
    //         }
    //     }
    // }

/* Optimized */
// #pragma HLS array_partition variable=Q dim=1 complete
// #pragma HLS array_partition variable=K dim=1 complete
// #pragma HLS array_partition variable=attention dim=1 complete
// #pragma HLS array_partition variable=V dim=1 complete
// #pragma HLS array_partition variable=Output dim=1 complete

    fixed_t scale = 1.0 / sqrt((float)dk);
    
    fixed_t attention[B][N][N];

    // Compute Q * K^T
    for (int i = 0; i < N; ++i) {
    #pragma HLS pipeline II=4
        for (int b = 0; b < B; ++b) {
        #pragma HLS unroll
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < dk; ++k) {
                    attention[b][i][j] += Q[b][i][k] * K[b][j][k] * scale;
                }
            }
        
            softmax_HLS_inline(attention[b][i]);
            
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < dk; ++k) {
                    Output[b][i][j] += attention[b][i][k] * V[b][k][j] * scale;
                }
            }
        }
    }

// #pragma HLS array_partition variable=attention dim=1 complete
// #pragma HLS array_partition variable=V dim=1 complete
// #pragma HLS array_partition variable=Output dim=1 complete

    // // Compute Attention * V
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < dv; ++j) {
    //         for (int k = 0; k < N; ++k) {
    //         #pragma HLS pipeline
    //             for (int b = 0; b < B; ++b) {
    //             #pragma HLS unroll
    //                 Output[b][i][j] += attention[b][i][k] * V[b][k][j] * scale;
    //             }
    //         }
    //     }
    // }




}