#include "dcl.h"
#include "hls_stream.h"


//-----------------------------//
// Definition of structure as a stream
//-----------------------------//
struct QK_block_t {
    // Use this block as a stream to send Q and K at once
    fixed_t Q_vec[dk];
    fixed_t K_vec[dk];
};

struct attn_row_t {
    fixed_t row[N];     // 2D stream to optimize stream
};

//-----------------------------//
// (1) read_input function
//   - Read Q, K, V from DRAM
//   - send Q, K to stream
//   - save V at BRAM
//-----------------------------//
void read_input(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], hls::stream<QK_block_t> &qkStream, fixed_t V_local[B][N][dv]) {
    // (a) Read Q, K and push it as a stream
    //     
    READ_QK_LOOP:
    for(int b=0; b < B; b++) {
        for(int i=0; i < N; i++) {
            QK_block_t pkt;
        #pragma HLS pipeline
            for(int d=0; d < dk; d++) {
                pkt.Q_vec[d] = Q[b][i][d];
                pkt.K_vec[d] = K[b][i][d];
            }
            qkStream.write(pkt);    // send pkt as a stream: 2D 
        }
    }

    // (b) save V at BRAM
    READ_V_LOOP:
    for(int b=0; b < B; b++){
        for(int i=0; i < N; i++){
            for(int d=0; d < dv; d++){
            #pragma HLS pipeline
                V_local[b][i][d] = V[b][i][d];
            }
        }
    }
}

//-----------------------------//
// (2) compute_QK function
//     get qkStream, compute QK, and send as a stream 
//-----------------------------//
void compute_QK(hls::stream<QK_block_t> &qkStream, hls::stream<attn_row_t> &fifo_qk2softmax) {
    // scale
    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);

    for(int b=0; b < B; b++){
        for(int i=0; i < N; i++){
            QK_block_t pkt = qkStream.read();
            attn_row_t outRow;  // 2D stream for softmax function           

            for(int j=0; j < N; j++){
            #pragma HLS pipeline
                // Q*K 
                fixed_t sum = 0;
                for(int d=0; d < dk; d++){
                #pragma HLS unroll factor=8
                    sum += pkt.Q_vec[d] * pkt.K_vec[d]; 
                }
                // scale
                sum = sum * scale;
                outRow.row[j] = sum;    // pack sum in 2D 
            }
            fifo_qk2softmax.write(outRow);  // send 2D stream to softmax function
        }
    }
}

//-----------------------------//
// (3) softmax_stream function
//-----------------------------//
void softmax_stream(hls::stream<attn_row_t> &fifo_qk2softmax, hls::stream<attn_row_t> &fifo_softmax2av) {
    for(int b=0; b < B; b++){
        for(int i=0; i < N; i++){
            attn_row_t tmp = fifo_qk2softmax.read();    // read 2D stream
            ap_fixed<32, 8> max_val = tmp.row[0];
            // find max
            for(int j=1; j < N; j++){
                if(tmp.row[j] > max_val){
                    max_val = tmp.row[j];
                }
            }
            // exp & sum
            ap_fixed<32, 8> sum=0;
            for(int j=0; j < N; j++){
            #pragma HLS pipeline
                tmp.row[j] = hls::exp(tmp.row[j] - max_val);
                sum += tmp.row[j];
            }
            // normalize
            for(int j=0; j < N; j++){
            #pragma HLS unroll factor=5
                tmp.row[j] /= sum;
            }
            fifo_softmax2av.write(tmp); // send result as 2D stream
        }
    }
}

//-----------------------------//
// (4) compute_AV
//-----------------------------//
void compute_AV(
    hls::stream<attn_row_t> &fifo_softmax2av, fixed_t V_local[B][N][dv], fixed_t Output[B][N][dv]) {
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < N; i++) {
            attn_row_t attn = fifo_softmax2av.read();   // read 2D stream
            attn_row_t sum;
            for (int d = 0; d < dv; d++) {
            #pragma HLS pipeline
                fixed_t mult[N];    // array to save N multiplies for each d
                #pragma HLS array_partition variable=mult complete dim=1
                // execute multiply for every k at once
                for (int k = 0; k < N; k++) {
                #pragma HLS unroll factor=8
                    mult[k] = attn.row[k] * V_local[b][k][d];
                }
                // sum by tree based reduction algorithm
                int count = N;
                while (count > 1) {
                    int half = (count + 1) >> 1;
                    for (int p = 0; p < half; p++) {
                    #pragma HLS unroll factor=8
                        if (p + half < count)
                            mult[p] = mult[p] + mult[p + half];
                    }
                    count = half;
                }
                fixed_t sum = mult[0];
                Output[b][i][d] = sum;  // write at DRAM
            }
        }
    }
}



//----------------------------------------------------------------
// Top-level: dataflow
//----------------------------------------------------------------
void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
// separate memory port to avoid conflict
#pragma HLS interface m_axi port=Q      offset=slave bundle=mem_q 
#pragma HLS interface m_axi port=K      offset=slave bundle=mem_k 
#pragma HLS interface m_axi port=V      offset=slave bundle=mem_v 
#pragma HLS interface m_axi port=Output offset=slave bundle=mem_out 
#pragma HLS interface s_axilite port=return

// local buffer
static fixed_t V_local[B][N][dv];

// (2) dataflow
#pragma HLS dataflow
    // stream
    static hls::stream<QK_block_t> qkStream("qkStream");
#pragma HLS stream variable=qkStream depth=1
    static hls::stream<attn_row_t> fifo_qk2softmax("fifo_qk2softmax");
#pragma HLS stream variable=fifo_qk2softmax depth=1
    static hls::stream<attn_row_t> fifo_softmax2av("fifo_softmax2av");
#pragma HLS stream variable=fifo_softmax2av depth=500

    // 1) read_input : Q,K->qkStream / V->V_local
    read_input(Q, K, V, qkStream, V_local);
    // 2) compute_QK  : qkStream -> fifo_qk2softmax
    compute_QK(qkStream, fifo_qk2softmax);
    // 3) softmax     : fifo_qk2softmax -> fifo_softmax2av
    softmax_stream(fifo_qk2softmax, fifo_softmax2av);
    // 4) compute_AV  : fifo_softmax2av + V_local -> Output_local
    compute_AV(fifo_softmax2av, V_local, Output);
}
