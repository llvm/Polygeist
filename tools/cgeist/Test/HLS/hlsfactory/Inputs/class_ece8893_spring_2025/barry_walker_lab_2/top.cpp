#include "dcl.h"

struct MatrixRowDK {
    fixed_t data[dk];
};

struct MatrixRowDV {
    fixed_t data[dv];
};

struct MatrixRowN {
    fixed_t data[N];
};

template <typename In, typename Out>
void accumulate_quarter_HLS(In source[], Out &dest, int length) {
    Out sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum12 = 0, sum34 = 0;
    for (int i = 0; i < length/4; i++) {
    #pragma HLS unroll
        sum1 += source[i];
        sum2 += source[i + length/4];
        sum3 += source[i + ((length/4) * 2)];
        sum4 += source[i + ((length/4) * 3)];
    }
    sum12 = sum1 + sum2;
    sum34 = sum3 + sum4;
    dest = sum12 + sum34;
}

// Module for reading Q and K from DRAM
void read_QK_HLS(mem_t Q[B][N][dk/32], mem_t K[B][N][dk/32], hls::stream<MatrixRowDK> &Q_row_buffer, hls::stream<MatrixRowDK> &K_row_buffer, int b) {

    MatrixRowDK Q_row_local;
#pragma HLS array_partition variable=Q_row_local.data complete

    MatrixRowDK K_row_local;
#pragma HLS array_partition variable=K_row_local.data complete

    mem_t data;

    // Read the first row of Q and put it in the buffer
    for (int k = 0; k < dk/32; k++) {
    #pragma HLS pipeline
        data = Q[b][0][k];
        for (int kk = 0; kk < 32; kk++) {
            ap_uint<16> temp = data.range(15 + (kk * 16), 0 + (kk * 16));
            Q_row_local.data[k * 32 + kk].range(15, 0) = temp;
        }
    }
    Q_row_buffer.write(Q_row_local);

    // Read all the rows in K and put them in the buffer
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < dk/32; k++) {
        #pragma HLS pipeline
            data = K[b][i][k];
            for (int kk = 0; kk < 32; kk++) {
                ap_uint<16> temp = data.range(15 + (kk * 16), 0 + (kk * 16));
                K_row_local.data[k * 32 + kk].range(15, 0) = temp;
            }
        }
        K_row_buffer.write(K_row_local);
    }

    // Read the rest of the rows in Q and put them in the buffer
    for (int i = 1; i < N; i++) {
        for (int k = 0; k < dk/32; k++) {
        #pragma HLS pipeline
            data = Q[b][i][k];
            for (int kk = 0; kk < 32; kk++) {
                ap_uint<16> temp = data.range(15 + (kk * 16), 0 + (kk * 16));
                Q_row_local.data[k * 32 + kk].range(15, 0) = temp;
            }
        }
        Q_row_buffer.write(Q_row_local);
    }
}

// Module that computes Q*K^T 
void compute_KQT_HLS(hls::stream<MatrixRowDK> &Q_row_buffer, hls::stream<MatrixRowDK> &K_row_buffer, hls::stream<MatrixRowN> &attention_row_buffer) {

    const fixed_t scale = 1.0 / sqrt((float)dk);

    MatrixRowDK Q_row_local;
#pragma HLS array_partition variable=Q_row_local.data complete

    // Cache for K so it doesn't need to be re-read from memory
    fixed_t K_matrix_local[N][dk];
#pragma HLS array_partition variable=K_matrix_local dim=2 factor=dk cyclic

    MatrixRowDK K_row_local;
#pragma HLS array_partition variable=K_row_local.data complete

    MatrixRowN attention_row_local;
#pragma HLS array_partition variable=attention_row_local.data complete

    for (int i = 0; i < N; ++i) {

        // Retrieve a row of Q
        Q_row_local = Q_row_buffer.read();

        for (int j = 0; j < N; ++j) {
        #pragma HLS pipeline
            // Retrieve a row of K. If on the first row of Q, cache the K row in local storage
            if (i == 0) {
                K_row_local = K_row_buffer.read();
                for (int k = 0; k < dk; k++) {
                #pragma HLS unroll
                    K_matrix_local[j][k] = K_row_local.data[k];
                }
            }

            // Perform the multiplication
            fixed_t mult_local[dk];
            #pragma HLS array_partition variable=mult_local complete
            for (int k = 0; k < dk; k++) {
            #pragma HLS unroll
                mult_local[k] = Q_row_local.data[k] * K_matrix_local[j][k];
            }

            // Perform the accumulation
            accum_t sum = 0;
            accumulate_quarter_HLS<fixed_t,accum_t>(mult_local, sum, dk);
            fixed_t small_sum = sum;

            // Scale the sum and store it back in the row
            attention_row_local.data[j] = small_sum * scale;
        }

        // Put the attention row in the buffer
        attention_row_buffer.write(attention_row_local);
    }
}

// Module that performs the softmax on rows of attention
void softmax_HLS(hls::stream<MatrixRowN> &attention_row_buffer, fixed_t attention[N][N]) {

    MatrixRowN attention_row_local;
#pragma HLS array_partition variable=attention_row_local.data complete

    // Lookup table of precalculated exponential values
    const ap_ufixed<8,0> exp_LUT[128] = {
        0.99609375, 0.98437500, 0.96875000, 0.95312500, 0.94140625, 0.92578125, 0.91406250, 0.89843750, 0.88671875, 0.87109375, 0.85937500, 0.84765625, 0.83203125, 0.82031250, 0.80859375, 0.79687500, 0.78515625, 0.77343750, 0.76171875, 0.75000000, 0.73828125, 0.72656250, 0.71875000, 0.70703125, 0.69531250, 0.68750000, 0.67578125, 0.66406250, 0.65625000, 0.64453125, 0.63671875, 0.62500000, 0.61718750, 0.60937500, 0.59765625, 0.58984375, 0.58203125, 0.57421875, 0.56250000, 0.55468750, 0.54687500, 0.53906250, 0.53125000, 0.52343750, 0.51562500, 0.50781250, 0.50000000, 0.49218750, 0.48437500, 0.47656250, 0.47265625, 0.46484375, 0.45703125, 0.44921875, 0.44531250, 0.43750000, 0.42968750, 0.42578125, 0.41796875, 0.41015625, 0.40625000, 0.39843750, 0.39453125, 0.38671875, 0.38281250, 0.37500000, 0.37109375, 0.36328125, 0.35937500, 0.35546875, 0.34765625, 0.34375000, 0.33984375, 0.33203125, 0.32812500, 0.32421875, 0.32031250, 0.31250000, 0.30859375, 0.30468750, 0.30078125, 0.29687500, 0.29296875, 0.28515625, 0.28125000, 0.27734375, 0.27343750, 0.26953125, 0.26562500, 0.26171875, 0.25781250, 0.25390625, 0.25000000, 0.24609375, 0.24218750, 0.23828125, 0.23437500, 0.23046875, 0.23046875, 0.22656250, 0.22265625, 0.21875000, 0.21484375, 0.21093750, 0.21093750, 0.20703125, 0.20312500, 0.19921875, 0.19531250, 0.19531250, 0.19140625, 0.18750000, 0.18359375, 0.18359375, 0.17968750, 0.17578125, 0.17578125, 0.17187500, 0.16796875, 0.16796875, 0.16406250, 0.16015625, 0.16015625, 0.15625000, 0.15625000, 0.15234375, 0.14843750, 0.14843750
    };
#pragma HLS array_partition variable=exp_LUT complete

    for (int i = 0; i < N; ++i) {
    #pragma HLS pipeline

        // Retrieve an attention row
        attention_row_local = attention_row_buffer.read();

        // Find the max value in the row
        fixed_t max_val = attention_row_local.data[0];
        for (int j = 1; j < N; j++) {
        #pragma HLS unroll
            if (attention_row_local.data[j] > max_val) {
                max_val = attention_row_local.data[j];
            }
        }

        // Perform exp using the LUT
        for (int j = 0; j < N; j++) {
        #pragma HLS unroll
            ap_uint<7> index;
            index.range(6,0) = fixed_t(max_val - attention_row_local.data[j]).range(11,5);
            attention_row_local.data[j] = exp_LUT[index];
        }

        // Perform the accumulation
        accum_t sum = 0;
        accumulate_quarter_HLS<fixed_t, accum_t>(attention_row_local.data, sum, N);

        // Calculate the reciprocal of the sum
        fixed_t inv_sum = hls::recip(sum);

        // Perform the division using the reciprocal and multiplication and store in attention matrix
        for (int j = 0; j < N; j++) {
        #pragma HLS unroll
            attention[i][j] = attention_row_local.data[j] * inv_sum;
        }
    }
}

// The first dataflow of the module
void dataflow1_HLS(mem_t Q[B][N][dk/32], mem_t K[B][N][dk/32], fixed_t attention[N][N], int b) {
#pragma HLS dataflow

    // Shared buffer for rows of Q
    hls::stream<MatrixRowDK> Q_row_buffer;
#pragma HLS STREAM variable=Q_row_buffer depth=N

    // Shared buffer rows of K
    hls::stream<MatrixRowDK> K_row_buffer;
#pragma HLS STREAM variable=K_row_buffer depth=8

    // Buffer for rows of attention
    hls::stream<MatrixRowN> attention_row_buffer;
#pragma HLS STREAM variable=attention_row_buffer depth=8

    read_QK_HLS(Q, K, Q_row_buffer, K_row_buffer, b);
    compute_KQT_HLS(Q_row_buffer, K_row_buffer, attention_row_buffer);
    softmax_HLS(attention_row_buffer, attention);
}

// Module for reading V from DRAM
void read_V_HLS(mem_t V[B][N][dv/32], hls::stream<MatrixRowDV> &V_row_buffer, int b) {

    MatrixRowDV V_row_local;
#pragma HLS array_partition variable=V_row_local.data factor=32 cyclic

    mem_t data;

    // Read all rows of V
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < dv/32; k++) {
        #pragma HLS pipeline
            data = V[b][i][k];
            for (int kk = 0; kk < 32; kk++) {
                ap_uint<16> temp = data.range(15 + (kk * 16), 0 + (kk * 16));
                V_row_local.data[k * 32 + kk].range(15, 0) = temp;
            }
        }
        V_row_buffer.write(V_row_local);
    }
}

// Module for computing the Attention * V after attention's softmaxing
void compute_AtV_HLS(fixed_t attention[N][N], hls::stream<MatrixRowDV> &V_row_buffer, hls::stream<MatrixRowDV> &output_row_buffer) {

    MatrixRowDV V_row_local;
#pragma HLS array_partition variable=V_row_local.data complete

    // Matrix for storing the partial sums
    accum_t output_matrix_local[N][dv];
#pragma HLS array_partition variable=output_matrix_local dim=2 factor=dv cyclic

    // Compressed output row to send to buffer
    MatrixRowDV output_rowB_local;
#pragma HLS array_partition variable=output_rowB_local.data complete

    fixed_t attention_val_local;

    for (int i = 0; i < N; i++) {

        // Retrieve a row of V
        V_row_local = V_row_buffer.read();

        for (int j = 0; j < N; j++) {
        #pragma HLS pipeline
            
            // Grab the value from the attention matrix
            attention_val_local = attention[j][i];

            accum_t mult_local[dv];
            #pragma HLS array_partition variable=mult_local complete
            // Perform the row multiplications and store intermediate sums in the output matrix
            for (int k = 0; k < dv; k++) {
            #pragma HLS unroll
                mult_local[k] = attention_val_local * V_row_local.data[k];
            }
            for (int k = 0; k < dv; k++) {
            #pragma HLS unroll
                if (i == 0) {
                    output_matrix_local[j][k] = mult_local[k];
                } else {
                    output_matrix_local[j][k] += mult_local[k];
                }
            }

            if (i == N-1) {
                // On last row iteration, send to output buffer.
                for (int h = 0; h < dv; h++) {
                #pragma HLS unroll
                    output_rowB_local.data[h] = output_matrix_local[j][h];
                }
                output_row_buffer.write(output_rowB_local);
            }
        }
    }
}

// Module for writing the output to DRAM
void write_output_HLS(mem_t Output[B][N][dv/32], hls::stream<MatrixRowDV> &output_row_buffer, int b) {
    MatrixRowDV output_row_local;
#pragma HLS array_partition variable=output_row_local.data complete

    mem_t data;

    for (int i = 0; i < N; ++i) {

        // Retrieve an output row
        output_row_local = output_row_buffer.read();

        // Pack it into wide data types and send
        for (int k = 0; k < dv/32; k++) {
        #pragma HLS pipeline
            for (int kk = 0; kk < 32; kk++) {
                ap_uint<16> temp = output_row_local.data[k * 32 + kk].range(15, 0);
                data.range(15 + (kk * 16), 0 + (kk * 16)) = temp;
            }
            Output[b][i][k] = data;
        }
    }
}

// The second dataflow for the module.
void dataflow2_HLS(fixed_t attention[N][N], mem_t V[B][N][dv/32], mem_t Output[B][N][dv/32], int b) {
#pragma HLS dataflow

    // Buffer for rows of V
    hls::stream<MatrixRowDV> V_row_buffer; 
#pragma HLS STREAM variable=V_row_buffer depth=N

    // Buffer for rows of output
    hls::stream<MatrixRowDV> output_row_buffer;
#pragma HLS STREAM variable=output_row_buffer depth=N

    read_V_HLS(V, V_row_buffer, b);
    compute_AtV_HLS(attention, V_row_buffer, output_row_buffer);
    write_output_HLS(Output, output_row_buffer, b);
}

// Top level module
void compute_attention_HLS(mem_t Q[B][N][dk/32], mem_t K[B][N][dk/32], mem_t V[B][N][dv/32], mem_t Output[B][N][dv/32]) {
#pragma HLS interface m_axi port=Q offset=slave bundle=mem1
#pragma HLS interface m_axi port=K offset=slave bundle=mem1
#pragma HLS interface m_axi port=V offset=slave bundle=mem1
#pragma HLS interface m_axi port=Output offset=slave bundle=mem2

#pragma HLS interface s_axilite port=return

    // Attention matrix that is transferred between the two dataflows
    fixed_t attention[N][N];
#pragma HLS array_partition variable=attention dim=2 factor=N cyclic

    // Dataflows are repeated for each batch
    for (int b = 0; b < B; b++) {
        dataflow1_HLS(Q, K, attention, b);
        dataflow2_HLS(attention, V, Output, b);
    }
}