#include "dcl.h"

void compute_QK(
    fixed_t Q_line[dk],
    fixed_t K_local[N][dk],
    fixed_t attention[N]
) {
    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);

    //QKCompute_loop_i:
        //for (int i = 0; i < N; ++i) { 
            //#pragma HLS LOOP_FLATTEN off
QKCompute_loop_j:
    for (int j = 0; j < N; ++j) { //j represents row of k
#pragma HLS LOOP_FLATTEN off
        ap_fixed<32, 8> sum_qk = 0;
    QKCompute_loop_k:
        for (int d = 0; d < dk; ++d) {
#pragma HLS UNROLL
            sum_qk += Q_line[d] * K_local[j][d];
        }
        attention[j] = sum_qk * scale;
    }  
}

void softmax(
    fixed_t attention0[N],
    fixed_t attention1[N]
) {
    ap_fixed<32, 8> max_val = attention0[0];
Softmax_max_j:
    for (int j = 1; j < N; ++j) {
#pragma HLS PIPELINE II=1
        if (attention0[j] > max_val) {
            max_val = attention0[j];
        }
    }
    ap_fixed<32, 8> sum_val = 0;
Softmax_exp_j:
    for (int j = 0; j < N; ++j) {
#pragma HLS PIPELINE II=1
        attention1[j] = hls::exp(attention0[j] - max_val);
        sum_val += attention1[j];
    }
    for (int j = 0; j < N; ++j) {
#pragma HLS UNROLL
        attention1[j] = attention1[j] / sum_val;
    }
}


void compute_AV(
    fixed_t attention[N],
    fixed_t V_local[N][dv],
    fixed_t output[dv]
) {
ComputeOut_loop:
    //for (int i = 0; i < N; ++i) {
    for (int j = 0; j < dv; ++j) {
#pragma HLS PIPELINE II=1
        ap_fixed<32, 8> sum_out = 0;
        for (int k = 0; k < N; ++k) {
#pragma HLS UNROLL factor=8
            sum_out += attention[k] * V_local[k][j];
        }
        output[j] = sum_out;
    }

}




//read completed KV matrix of 1 batch
void read_data_KV(
   // fixed_t Q_in[B][N][dk],
    fixed_t K_in[B][N][dk],
    fixed_t V_in[B][N][dv],
   // fixed_t Q_buf[N][dk],
    fixed_t K_buf[N][dk],
    fixed_t V_buf[N][dv],
    int batch_idx
) {
#pragma HLS INLINE off

    Read_loop_i:
    for (int i = 0; i < N; i++) {
    Read_loop_d:
        for (int d = 0; d < dk; d++) {
#pragma HLS PIPELINE II=1
           // Q_buf[i][d] = Q_in[batch_idx][i][d];
            K_buf[i][d] = K_in[batch_idx][i][d];
        }
    Read_loop_dv:
        for (int dv_i = 0; dv_i < dv; dv_i++) {
#pragma HLS PIPELINE II=1
            V_buf[i][dv_i] = V_in[batch_idx][i][dv_i];
        }
    }
}

void read_Q_Line(
    fixed_t Q_in[B][N][dk],
    fixed_t Q_line[dk],
    int Bidx,
    int Q_Lidx
) {
    for (int i = 0; i < dk; ++i) {
        Q_line[i] = Q_in[Bidx][Q_Lidx][i];
    }
}


//----------------------------------------------------------
// write back to DRAM
//----------------------------------------------------------
void write_data_Output(
    fixed_t Output_out[B][N][dv],
    fixed_t Out_buf[dv],
    int batch_idx,
    int Lidx
) {
#pragma HLS INLINE off

    //Write_loop_i:
    //for (int i = 0; i < N; i++) {
    Write_loop_dv:
        for (int dv_i = 0; dv_i < dv; dv_i++) {
#pragma HLS UNROLL
    		Output_out[batch_idx][Lidx][dv_i] = Out_buf[dv_i];
        }
   // }
}

void compute_attention_HLS(
    fixed_t Q[B][N][dk],
    fixed_t K[B][N][dk],
    fixed_t V[B][N][dv],
    fixed_t Output[B][N][dv]
) {
#pragma HLS interface m_axi port=Q      offset=slave bundle=mem1 max_widen_bitwidth=512
#pragma HLS interface m_axi port=K      offset=slave bundle=mem1 max_widen_bitwidth=512
#pragma HLS interface m_axi port=V      offset=slave bundle=mem1 max_widen_bitwidth=512
#pragma HLS interface m_axi port=Output offset=slave bundle=mem2 max_widen_bitwidth=512
#pragma HLS interface s_axilite port=return


    for (int b = 0; b < B; b++) {

        fixed_t K_buff[N][dk], V_buff[N][dv];

#pragma HLS ARRAY_PARTITION variable=K_buff dim=2 factor=64 cyclic
#pragma HLS ARRAY_PARTITION variable=V_buff dim=1 factor=64 cyclic

        read_data_KV(K, V, K_buff, V_buff, b);


        for (int QL = 0; QL < N; ++ QL) {

            fixed_t Q_buff0[dk];//, Q_buff1[dk];
            fixed_t Out_buff0[dv];//, Out_buff1[dv];
            fixed_t attention0[N];//, attention1[N];
            fixed_t attention_sm0[N];//, attention_sm1[N];
#pragma HLS ARRAY_PARTITION variable=Q_buff0 dim=1 factor=64 cyclic
//#pragma HLS ARRAY_PARTITION variable=Q_buff1 dim=1 factor=64 cyclic
#pragma HLS ARRAY_PARTITION variable=Out_buff0 dim=1 factor=64 cyclic
//#pragma HLS ARRAY_PARTITION variable=Out_buff1 dim=1 factor=64 cyclic
#pragma HLS ARRAY_PARTITION variable=attention0 dim=1 factor=64 cyclic
//#pragma HLS ARRAY_PARTITION variable=attention1 dim=1 factor=64 cyclic
#pragma HLS ARRAY_PARTITION variable=attention_sm0 dim=1 factor=64 cyclic
//#pragma HLS ARRAY_PARTITION variable=attention_sm1 dim=1 factor=64 cyclic


#pragma HLS DATAFLOW
#pragma HLS stable variable=K_buff
#pragma HLS stable variable=V_buff
            //int pp = QL % 2;
            //int prev_pp = 1 - pp;
            int pp = 0;
            int prev_pp = 0;

            read_Q_Line(Q, Q_buff0, b, QL);
            compute_QK(Q_buff0, K_buff, attention0);
            softmax(attention0, attention_sm0);
            compute_AV(attention_sm0, V_buff, Out_buff0);
            write_data_Output(Output, Out_buff0, b, QL);
            //load Q row 1
            // if (QL < N) {
            //     if (pp == 0) {
            //         read_Q_Line(Q, Q_buff0, b, QL);
            //     }
            //     else {
            //        // read_Q_Line(Q, Q_buff1, b, QL);
            //     }
            // }

            // compute Q*K^T 2
            // if (QL > 0 && QL <= N) {
            //     if (prev_pp == 0) {
            //         compute_QK(Q_buff0, K_buff, attention0);
            //     }
            //     else {
            //      //   compute_QK(Q_buff1, K_buff, attention1);
            //     }
            // }


            // softmax 3
            // if (QL > 1 && (QL - 2) < N) {
            //    // int Line = QL - 2;
            //     if (pp == 0) {
            //         // pp=0 => prev_pp=1 
            //         softmax(attention0, attention_sm0);
            //     }
            //     else {
            //        // softmax(attention1, attention_sm1);
            //     }
            // }


            // compute Atten*V 4
            // if (QL > 2 && (QL - 3) < N) {
            //     if (prev_pp == 0) {
            //         compute_AV(attention_sm0, V_buff, Out_buff0);
            //     }
            //     else {
            //       //  compute_AV(attention_sm1, V_buff, Out_buff1);
            //     }
            // }

            // write back 5
            // if (QL > 3 && (QL - 4) < N) {
            //     int Line = QL - 4;
            //     if (pp == 0) {
            //         // pp=0 => prev_pp=1 
            //         write_data_Output(Output, Out_buff0, b, Line);
            //     }
            //     else {
            //      //   write_data_Output(Output, Out_buff1, b, Line);
            //     }
            // }

        }
	printf("Batch %d completed \n", b);
    }
}
