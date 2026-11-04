#include "dcl.h"
#include <hls_stream.h>
#include <math.h>
#include <string.h>

static const ap_fixed<32,8> LUT_MIN = -1.0;

static const ap_fixed<32,8> LUT_MAX = 1.0;

static const ap_fixed<32,8> LUT_STEP = 0.125;

static const int LUT_SIZE = 17; // (1.0 - (-1.0)) / 0.125 + 1 = 17

static const ap_fixed<32,8> exp_lut[LUT_SIZE] = {

    0.3679,  // exp(-1.0)

    0.4169,  // exp(-0.875)

    0.4724,  // exp(-0.75)

    0.5353,  // exp(-0.625)

    0.6065,  // exp(-0.5)

    0.6873,  // exp(-0.375)

    0.7788,  // exp(-0.25)

    0.8825,  // exp(-0.125)

    1.0,     // exp(0.0)

    1.1331,  // exp(0.125)

    1.2840,  // exp(0.25)

    1.4550,  // exp(0.375)

    1.6487,  // exp(0.5)

    1.8683,  // exp(0.625)

    2.1170,  // exp(0.75)

    2.3989,  // exp(0.875)

    2.7183   // exp(1.0)

};

ap_fixed<32,8> approx_exp(ap_fixed<32,8> x) {
 
    if (x < LUT_MIN) x = LUT_MIN;

    if (x > LUT_MAX) x = LUT_MAX;

 
    ap_fixed<32,8> pos = x - LUT_MIN;

    ap_fixed<32,8> index_f = pos / LUT_STEP;

    int index = (int)index_f;

    if (index >= LUT_SIZE - 1)

        index = LUT_SIZE - 2;

    ap_fixed<32,8> frac = index_f - (ap_fixed<32,8>)index;

    ap_fixed<32,8> base = exp_lut[index];

    ap_fixed<32,8> next = exp_lut[index + 1];

    return base + (next - base) * frac;

}


typedef struct {
    fixed_t data[N];
} row_t;

//DRAM TO lcoal 
void read_data(fixed_t Q[B][N][dk],
               fixed_t K[B][N][dk],
               fixed_t V[B][N][dv],
               fixed_t Q_local[B][N][dk],
               fixed_t K_local[B][N][dk],
               fixed_t V_local[B][N][dv]) {
  
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < N; i++) {
            #pragma HLS PIPELINE II=1 
            for (int j = 0; j < dk; j++) {
                 #pragma HLS UNROLL factor=8
                Q_local[b][i][j] = Q[b][i][j];
            }
        }
    }

    for (int b = 0; b < B; b++) {
        for (int i = 0; i < N; i++) {
            #pragma HLS PIPELINE II=1 
            for (int j = 0; j < dk; j++) {
                 #pragma HLS UNROLL factor=8
                K_local[b][i][j] = K[b][i][j];
            }
        }
    }
    
    
 
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < N; i++) {
            #pragma HLS PIPELINE II=1
            for (int j = 0; j < dv; j++) {
                 #pragma HLS UNROLL factor=8
                V_local[b][i][j] = V[b][i][j];
            }
        }
    }

}
//Compute attention and write into output stream[b] corresponding to batch b
void compute_attention_matrix_stream(fixed_t Q_local[B][N][dk],
                                     fixed_t K_local[B][N][dk],
                                     hls::stream<row_t> att_stream[B],
                                     ap_fixed<32,8> scale) {
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < N; i++) {
            row_t row;
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                ap_fixed<32,8> sum = 0;
                for (int k = 0; k < dk; k++) {
                    #pragma HLS UNROLL factor=8
                    sum += Q_local[b][i][k] * K_local[b][j][k];
                }
                row.data[j] = sum * scale;
            }
            att_stream[b].write(row);
        }
    }
}

//APPY SOFTMAX ON input stream and wrote into output stream
void softmax_stream(hls::stream<row_t> &att_in, hls::stream<row_t> &att_out) {
    row_t row;
    // Read one row from the stream.
    row = att_in.read();
    ap_fixed<32,8> max_val = row.data[0];
    for (int j = 1; j < N; j++) {
        #pragma HLS PIPELINE II=1
         #pragma HLS UNROLL factor=8
        if (row.data[j] > max_val)
            max_val = row.data[j];
    }
    ap_fixed<32,8> sum = 0;
    for (int j = 0; j < N; j++) {
        #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=8
        row.data[j] = approx_exp(row.data[j] - max_val);
        sum += row.data[j];
    }
    for (int j = 0; j < N; j++) {
        #pragma HLS PIPELINE II=1
         #pragma HLS UNROLL factor=8
        row.data[j] /= sum;
    }
    att_out.write(row);
}

//Multiply with value
void compute_output_stream(hls::stream<row_t> att_stream[B],
                           fixed_t V_local[B][N][dv],
                           fixed_t Output_local[B][N][dv]) {
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < N; i++) {
            row_t att_row = att_stream[b].read();
            for (int j = 0; j < dv; j++) {
                #pragma HLS PIPELINE II=1
                ap_fixed<32,8> sum = 0;
                for (int k = 0; k < N; k++) {
                    #pragma HLS UNROLL factor=8
                    sum += att_row.data[k] * V_local[b][k][j];
                }
                Output_local[b][i][j] = sum;
            }
        }
    }
}

//Local biffer to DRAM
void write_data(fixed_t Output_local[B][N][dv], fixed_t Output[B][N][dv]) {
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < N; i++) {
            #pragma HLS PIPELINE II=1
            for (int j = 0; j < dv; j++) {
                #pragma HLS UNROLL factor=8
                Output[b][i][j] = Output_local[b][i][j];
            }
        }
    }
}


void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk],
                           fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
    


    #pragma HLS interface m_axi port=Q offset=slave bundle=mem1
    #pragma HLS interface m_axi port=K offset=slave bundle=mem1
    #pragma HLS interface m_axi port=V offset=slave bundle=mem1
    #pragma HLS interface m_axi port=Output offset=slave bundle=mem2

    #pragma HLS interface s_axilite port=return
    ap_fixed<32,8> scale = 1.0 / sqrt((float)dk);

    // Local buffers for DRAM data.
    static fixed_t Q_local[B][N][dk];
    static fixed_t K_local[B][N][dk];
    static fixed_t V_local[B][N][dv];
    static fixed_t Output_local[B][N][dv];

    // Cyclic partitioning of local buffers.
    #pragma HLS ARRAY_PARTITION variable=Q_local cyclic factor=8 dim=3
    #pragma HLS ARRAY_PARTITION variable=K_local cyclic factor=8 dim=3
    #pragma HLS ARRAY_PARTITION variable=V_local cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=Output_local block factor=8 dim=3

    // Streams for passing attention rows between stages.
    hls::stream<row_t> att_stream_raw[B];
    hls::stream<row_t> att_stream_soft[B];
    #pragma HLS STREAM variable=att_stream_raw depth=32
    #pragma HLS STREAM variable=att_stream_soft depth=32

    #pragma HLS DATAFLOW

    read_data(Q, K, V, Q_local, K_local, V_local);


    compute_attention_matrix_stream(Q_local, K_local, att_stream_raw, scale);

    for (int b = 0; b < B; b++) {
        for (int i = 0; i < N; i++) {
            #pragma HLS DATAFLOW
            softmax_stream(att_stream_raw[b], att_stream_soft[b]);
        }
    }
    compute_output_stream(att_stream_soft, V_local, Output_local);
    write_data(Output_local, Output);
}
