#include "dcl.h"

// void load_qkv_HLS (fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], hls::stream<hls::vector<fixed_t, dk>>& Q_buffer, hls::stream<hls::vector<fixed_t, dk>>& K_buffer, hls::stream<hls::vector<fixed_t, N>>& VT_buffer) {
//     //load the data
//     LOAD: for (int b = 0; b < B; ++b) {
//         hls::vector<fixed_t, N> VT_mat[dv];
//         LOAD_ROW: for (int i = 0; i < N; ++i) {
//             hls::vector<fixed_t, dk> K_row;
//             hls::vector<fixed_t, dk> Q_row;
//             FETCH_Q: for (int j = 0; j < dk; ++j) {
//                 Q_row[j] = Q[b][i][j];
//             }
//             FETCH_K: for (int j = 0; j < dk; ++j) {
//                 K_row[j] = K[b][i][j];
//             }
//             //make sure to transpose VT
//             FETCH_VT: for (int j = 0; j < dv; ++j) {
//                 VT_mat[j][i] = V[b][i][j];
//             }
//             Q_buffer.write(Q_row);
//             K_buffer.write(K_row);
//         }
//         //write VT to buffer
//         WRITE_VT: for (int j = 0; j < dv; ++j) {
//             VT_buffer.write(VT_mat[j]);
//         }
//     }
// }
void load_q_HLS (hls::vector<fixed_t, dk> Q[B][N],  hls::stream<hls::vector<fixed_t, dk>>& Q_buffer) {
    //load the data
    LOAD: for (int b = 0; b < B; ++b) {
        LOAD_ROW: for (int i = 0; i < N; ++i) {
            // #pragma HLS pipeline II=2
            #pragma HLS performance target_tl=5000
            // hls::vector<fixed_t, dk> Q_row;
            // FETCH_Q: for (int j = 0; j < dk; ++j) {
            //     Q_row[j] = Q[b][i][j];
            // }
            // Q_buffer.write(Q_row);
            Q_buffer.write(Q[b][i]);
        }
    }
}

void load_k_HLS ( hls::vector<fixed_t, dk> K[B][N],  hls::stream<hls::vector<fixed_t, dk>>& K_buffer) {
    //load the data
    LOAD: for (int b = 0; b < B; ++b) {
        LOAD_ROW: for (int i = 0; i < N; ++i) {
            // #pragma HLS pipeline II=2
            #pragma HLS performance target_tl=5000
            // hls::vector<fixed_t, dk> K_row;
            // FETCH_K: for (int j = 0; j < dk; ++j) {
            //     K_row[j] = K[b][i][j];
            // }
            // K_buffer.write(K_row);
            K_buffer.write(K[b][i]);
        }
    }
}

void load_v_HLS (hls::vector<fixed_t, dv> V[B][N],  hls::stream<hls::vector<fixed_t, dv>>& V_buffer) {
    //load the data
    LOAD: for (int b = 0; b < B; ++b) {
        LOAD_ROW: for (int i = 0; i < N; ++i) {
            // #pragma HLS pipeline II=2
            #pragma HLS performance target_tl=5000
            // hls::vector<fixed_t, dk> K_row;
            // FETCH_K: for (int j = 0; j < dk; ++j) {
            //     K_row[j] = K[b][i][j];
            // }
            // K_buffer.write(K_row);
            V_buffer.write(V[b][i]);
        }
    }
}


void compute_qkt_HLS (hls::stream<hls::vector<fixed_t, dk>>& Q, hls::stream<hls::vector<fixed_t, dk>>& K, hls::stream<hls::vector<fixed_t, N>>& attention) {
    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);

    // Compute Q * K^T
    BATCH: for (int b = 0; b < B; ++b) {
        //get the K matrix from the buffer
        hls::vector<fixed_t, dk> K_mat[N];
        FETCH_K: for (int i = 0; i < N; ++i) {
            K_mat[i] = K.read();
        }
        COMPUTE: for (int i = 0; i < N; ++i) {
            hls::vector<fixed_t, N> attention_row;
            hls::vector<fixed_t, dk> Q_row = Q.read();
            ROW: for (int j = 0; j < N; ++j) {
                ap_fixed<32, 8> sum = 0;
                // hls::vector<fixed_t, dk> QK_mult = Q_row * K_mat[j];
                //can unroll the sum
                MAC: for (int k = 0; k < dk; ++k) {
                    #pragma HLS unroll
                    // sum += QK_mult[k];
                    sum += Q_row[k] * K_mat[j][k];
                }
                attention_row[j] = sum * scale;
            }
            attention.write(attention_row);
        }
    }
}

void softmax_HLS(hls::stream<hls::vector<fixed_t, N>>& attention, hls::stream<hls::vector<fixed_t, N>>& softmax_attention) {
    BATCH: for (int b = 0; b < B; ++b) {
        //want to try internal dataflow here
        COMPUTE: for (int i = 0; i < N; ++i) {
            hls::vector<fixed_t, N> attention_row = attention.read();
            fixed_t max_val = attention_row[0];
            MAX: for (int j = 1; j < N; ++j) {
                #pragma HLS unroll
                if (attention_row[j] > max_val) {
                    max_val = attention_row[j];
                }
            }
            
            hls::vector<fixed_t, N> attention_row_centered;
            hls::vector<fixed_t, N> attention_row_exp;
            ap_fixed<16, 8> sum = 0;
            EXP_ACC: for (int j = 0; j < N; ++j) {
                #pragma HLS unroll factor=4
                attention_row_centered[j] = attention_row[j] - max_val;
                attention_row_exp[j] = hls::exp(attention_row_centered[j]);
                sum += attention_row_exp[j];
            }

            // ap_fixed<16, 8> sum = 0;
            // SUM: for (int j = 0; j < N; ++j) {
            //     // #pragma HLS unroll
            //     sum += attention_row_exp[j];
            // }


            hls::vector<fixed_t, N> attention_row_norm;
            NORM: for (int j = 0; j < N; ++j) {
                #pragma HLS unroll factor=5
                attention_row_norm[j] = attention_row_exp[j] / sum;
            }
            softmax_attention.write(attention_row_norm);

            // for (int j = 0; j < N; ++j) {
            //     softmax_attention[b][i][j] /= sum;
            // }
        }
    }
}

void compute_attentionv_HLS(hls::stream<hls::vector<fixed_t, N>>& attention, hls::stream<hls::vector<fixed_t, dv>>& V, hls::stream<hls::vector<fixed_t, dv>>& Output) {
    // Compute Attention * VT
    BATCH: for (int b = 0; b < B; ++b) {
        //get the V matrix from the buffer
        hls::vector<fixed_t, dv> V_mat[N];
        FETCH_K: for (int i = 0; i < N; ++i) {
            V_mat[i] = V.read();
        }
        COMPUTE: for (int i = 0; i < N; ++i) {
            // #pragma HLS performance target_tl=5000
            hls::vector<fixed_t, N> attention_row = attention.read();
            hls::vector<fixed_t, dv> output_row = fixed_t(0);
            // ROW: for (int j = 0; j < dv; ++j) {
            //     #pragma HLS unroll
            //     fixed_t sum = 0;
            //     MAC: for (int k = 0; k < N; ++k) {
            //         // sum += attention_mult[k];
            //         sum += attention_row[k] * V_mat[k][j];
            //     }
            //     output_row[j] = sum;
            // }
            MAC: for (int k = 0; k < N; k++) {
                fixed_t sum = 0;
                hls::vector<fixed_t, dv> mult = attention_row[k] * V_mat[k];
                ROW: for (int j = 0; j < dv; j++) {
                    #pragma HLS unroll
                    output_row[j] += mult[j];
                }
            }
            Output.write(output_row);
        }
    }
}

void store_output_HLS(hls::stream<hls::vector<fixed_t, dv>>& Output_buffer, hls::vector<fixed_t, dv> Output[B][N]) {
    STORE: for (int b = 0; b < B; ++b) {
        STORE_ROW: for (int i = 0; i < N; ++i) {
            // #pragma HLS pipeline II=2
            #pragma HLS performance target_tl=5000
            hls::vector<fixed_t, dv> output_row = Output_buffer.read();
            // OUTPUT: for (int j = 0; j < dv; ++j) {
            //     // #pragma HLS unroll factor=2
            //     Output[b][i][j] = output_row[j];
            // }
            Output[b][i] = output_row;
        }
    }
}

void compute_attention_HLS(hls::vector<fixed_t, dk> Q[B][N], hls::vector<fixed_t, dk> K[B][N], hls::vector<fixed_t, dv> V[B][N], hls::vector<fixed_t, dv> Output[B][N]) {
    #pragma HLS interface m_axi port=Q offset=slave bundle=mem1 channel=0
    #pragma HLS interface m_axi port=K offset=slave bundle=mem1 channel=1
    #pragma HLS interface m_axi port=V offset=slave bundle=mem1 channel=2
    #pragma HLS interface m_axi port=Output offset=slave bundle=mem2

    #pragma HLS interface s_axilite port=return

	#pragma HLS DATAFLOW

    //Q * K^T uses sum over k for Q, so buffer should hold entire row of Q
    hls::stream<hls::vector<fixed_t, dk>, 2> Q_buffer;
    //K us transposed and is used multiple times, so store entire K matrix
    hls::stream<hls::vector<fixed_t, dk>, 100> K_buffer;
    //V used in transpose, so buffer should hold entire row of V (transposed as you load)
    hls::stream<hls::vector<fixed_t, dv>, 100> V_buffer;
    //Attention uses max over N for attention, so buffer should hold entire row of attention
    hls::stream<hls::vector<fixed_t, N>, 2> attention_buffer;
    //Softmax_attention uses sum over N for softmax_attention, so buffer should hold entire row of softmax_attention
    hls::stream<hls::vector<fixed_t, N>, 2> softmax_attention_buffer;
    //No need to use a vector for softmax_attention as only a single value should be created per cycle for it for now
    hls::stream<hls::vector<fixed_t, dv>, 2> Output_buffer;

    //load data
    // load_qkv_HLS(Q, K, V, Q_buffer, K_buffer, VT_buffer);
    load_q_HLS(Q, Q_buffer);
    load_k_HLS(K, K_buffer);
    load_v_HLS(V, V_buffer);

   // Compute Q * K^T
   compute_qkt_HLS(Q_buffer, K_buffer, attention_buffer);

   // Apply softmax
   softmax_HLS(attention_buffer, softmax_attention_buffer);

   // Compute Attention * V^T
   compute_attentionv_HLS(softmax_attention_buffer, V_buffer, Output_buffer);

   //store data
    store_output_HLS(Output_buffer, Output);

}
