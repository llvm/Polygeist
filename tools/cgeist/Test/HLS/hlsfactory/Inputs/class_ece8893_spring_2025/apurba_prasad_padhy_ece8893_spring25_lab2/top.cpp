#include "dcl.h"
#include "hls_stream.h"

// Stream-based softmax: reads N values from in_stream, computes softmax, and writes N softmax values into out_stream.
void softmax_stream(hls::stream<fixed_t> &in_stream, hls::stream<fixed_t> &out_stream) {
    fixed_t localA[N];
    #pragma HLS ARRAY_PARTITION variable=localA complete

    fixed_t localB[N];
    #pragma HLS ARRAY_PARTITION variable=localB complete

    fixed_t localC[N];
    #pragma HLS ARRAY_PARTITION variable=localC complete


    // Find maximum for numerical stability
    // Read the entire attention vector from the stream
    fixed_t max_val = localA[0];
    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        localA[i] = in_stream.read();
        if (localA[i] > max_val) max_val = localA[i];
    }
    
    // Compute exponentials and accumulate the sum
    ap_fixed<16, 8> exp_sum = 0;
    for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=25
        localB[i] = hls::exp(localA[i] - max_val);
        exp_sum += localB[i];
    }

    // Normalization
    for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL 
        localC[i] = localB[i] / exp_sum;
    }

    // Write back to stream
    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        out_stream.write(localC[i]);
    }
}


void compute_attention_HLS(fixed_t Q[B][N][dk],
                           fixed_t K[B][N][dk],
                           fixed_t V[B][N][dv],
                           fixed_t Output[B][N][dv]) {
#pragma HLS interface m_axi port=Q offset=slave bundle=mem1 depth=51200
#pragma HLS interface m_axi port=K offset=slave bundle=mem2 depth=51200
#pragma HLS interface m_axi port=V offset=slave bundle=mem1 depth=51200
#pragma HLS interface m_axi port=Output offset=slave bundle=mem2 depth=51200
#pragma HLS interface s_axilite port=return
#pragma HLS Dataflow


    // Local BRAM buffers for one batch; partition inner arrays for parallel access.
    ap_fixed<16, 2> Q_local[N][dk];
    ap_fixed<16, 2> K_local[N][dk];
    ap_fixed<16, 2> V_local[N][dv];

    #pragma HLS ARRAY_PARTITION variable=Q_local cyclic dim=2 factor=128
    #pragma HLS ARRAY_PARTITION variable=K_local cyclic dim=2 factor=128
    #pragma HLS ARRAY_PARTITION variable=V_local cyclic dim=1 factor=100

    // Process one batch at a time
    for (int b = 0; b < B; b++) {
        // Load Q from external memory into local buffers
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < dk; j++) {
                #pragma HLS PIPELINE II=1
                Q_local[i][j] = Q[b][i][j];
            }
        }

        // Load K from external memory into local buffers
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < dk; j++) {
                #pragma HLS PIPELINE II=1
                K_local[i][j] = K[b][i][j];
            }
        }

        // Load V from external memory into local buffers
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < dk; j++) {
                #pragma HLS PIPELINE II=1
                V_local[i][j] = V[b][i][j];
            }
        }

        // Scaling factor for dot product normalization
        ap_fixed<16, 2> scale = 1.0 / sqrt((float)dk);

        // Process each query vector (each row of Q_local)
        for (int i = 0; i < N; i++) {
            // Declare streams to hold the dot-product results and softmax output
            hls::stream<fixed_t> dot_stream("dot_stream");
            #pragma HLS STREAM variable=dot_stream depth=N
            hls::stream<fixed_t> softmax_stream_out("softmax_stream_out");
            #pragma HLS STREAM variable=softmax_stream_out depth=N

            // Stage 1: Compute dot products (Q_local[i] * K_local[j]^T) for all j,
            // writing each scaled dot product to the stream.
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                ap_fixed<24, 8> dot[dk] = {0};
                ap_fixed<24, 8> attention = 0;
                #pragma HLS ARRAY_PARTITION variable=dot complete

                for (int k = 0; k < dk; k++) {
                    #pragma HLS UNROLL
                    dot[k] = Q_local[i][k] * K_local[j][k];
                }

                for (int k = 0; k < dk; k++) {
                    #pragma HLS UNROLL
                    attention += dot[k];
                }

                dot_stream.write(attention * scale);
            }


            // Stage 2: Perform softmax on the streamed attention values.
            softmax_stream(dot_stream, softmax_stream_out);

            // Since the softmax output will be used in an inner loop (for each output channel), store it in a temporary local buffer.
            fixed_t softmax_vals[N];
            #pragma HLS ARRAY_PARTITION variable=softmax_vals complete

            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                softmax_vals[j] = softmax_stream_out.read();
            }

            // Stage 3: Compute the weighted sum with V_local using the softmax values.
            for (int j = 0; j < dv; j++) {
                #pragma HLS PIPELINE II=1
                ap_fixed<24, 8> sum[N] = {0};
                ap_fixed<24, 8> output = 0;
                #pragma HLS ARRAY_PARTITION variable=sum complete

                for (int k = 0; k < N; k++) {
                    #pragma HLS UNROLL
                    sum[k] = softmax_vals[k] * V_local[k][j];
                }

                for (int k = 0; k < N; k++) {
                    #pragma HLS UNROLL
                    output += sum[k];
                }
                Output[b][i][j]= output;
            }
        }
    }
}
