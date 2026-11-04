#include "dcl.h"
#include <hls_stream.h>

// Wrap each row into a struct so it can be streamed.
struct RowQ {
    fixed_t data[dk];
};

struct RowK {
    fixed_t data[dk];
};

struct RowV {
    fixed_t data[dv];
};

struct RowOut {
    fixed_t data[dv];
};

static void get_softmax_single_row(fixed_t attention[N][N]) {
#pragma HLS INLINE
    for (int i = 0; i < N; i++) {
        ap_fixed<32,8> max_val = attention[i][0];
        for (int j = 1; j < N; j++) {
        #pragma HLS UNROLL factor = 4
            if (attention[i][j] > max_val)
                max_val = attention[i][j];
        }
        ap_fixed<32,8> sum_val = 0;
        for (int j = 0; j < N; j++) {
        #pragma HLS UNROLL factor = 4
            attention[i][j] = hls::exp(attention[i][j] - max_val);
            sum_val += attention[i][j];
        }
        for (int j = 0; j < N; j++) {
        #pragma HLS UNROLL factor = 4
            attention[i][j] = attention[i][j] / sum_val;
        }
    }
}

static void get_compute_attention_single_row(
    fixed_t Q_local[N][dk],
    fixed_t K_local[N][dk],
    fixed_t V_local[N][dv],
    fixed_t Out_local[N][dv]
) {
#pragma HLS INLINE
    static fixed_t attention[N][N];
#pragma HLS ARRAY_PARTITION variable=attention complete dim=2

    ap_fixed<32,8> scale = 1.0 / sqrt((float)dk);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            ap_fixed<32,8> sum_val = 0;
            for (int k = 0; k < dk; k++) {
                sum_val += Q_local[i][k] * K_local[j][k];
            }
            attention[i][j] = sum_val * scale;
        }
    }
    get_softmax_single_row(attention);
    for (int i = 0; i < N; i++) {
        for (int dv_i = 0; dv_i < dv; dv_i++) {
#pragma HLS PIPELINE II=1
            ap_fixed<32,8> sum_val = 0;
            for (int n_i = 0; n_i < N; n_i++) {
                sum_val += attention[i][n_i] * V_local[n_i][dv_i];
            }
            Out_local[i][dv_i] = sum_val;
        }
    }
}

// Load task: Reads one row at a time from global memory and writes to the FIFOs.
void load_task(
    fixed_t Q_in[B][N][dk],
    fixed_t K_in[B][N][dk],
    fixed_t V_in[B][N][dv],
    hls::stream<RowQ> &q_fifo,
    hls::stream<RowK> &k_fifo,
    hls::stream<RowV> &v_fifo
) {
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < N; i++) {
            RowQ row_q;
            RowK row_k;
            RowV row_v;
            // Read one row for Q and K
            for (int j = 0; j < dk; j++) {
#pragma HLS PIPELINE II=1
                row_q.data[j] = Q_in[b][i][j];
            }
            for (int j = 0; j < dk; j++) {
#pragma HLS PIPELINE II=1
                row_k.data[j] = K_in[b][i][j];
            }
            // Read one row for V
            for (int j = 0; j < dv; j++) {
#pragma HLS PIPELINE II=1
                row_v.data[j] = V_in[b][i][j];
            }
            q_fifo.write(row_q);
            k_fifo.write(row_k);
            v_fifo.write(row_v);
        }
    }
}

// Compute task: Reconstructs full matrices by reading rows sequentially,
// computes attention, then writes out the result rows.
void compute_task(
    hls::stream<RowQ> &q_fifo,
    hls::stream<RowK> &k_fifo,
    hls::stream<RowV> &v_fifo,
    hls::stream<RowOut> &out_fifo
) {
    fixed_t Q_local[N][dk];
    fixed_t K_local[N][dk];
    fixed_t V_local[N][dv];
    fixed_t Out_local[N][dv];

    for (int b = 0; b < B; b++) {
        // Merge stream reads into one loop per row
        for (int i = 0; i < N; i++) {
            RowQ rQ;
            RowK rK;
            RowV rV;
            // Sequentially read each row from all three streams
            q_fifo.read(rQ);
            k_fifo.read(rK);
            v_fifo.read(rV);
            // Store the row into local buffers
            for (int j = 0; j < dk; j++) {
            #pragma HLS PIPELINE II=1
                Q_local[i][j] = rQ.data[j];
            }
            
            for (int j = 0; j < dk; j++) {
            #pragma HLS PIPELINE II=1
            K_local[i][j] = rK.data[j];
            }
            
            for (int j = 0; j < dv; j++) {
            #pragma HLS PIPELINE II=1
                V_local[i][j] = rV.data[j];
            }
        }
        // Compute attention for the full batch.
        get_compute_attention_single_row(Q_local, K_local, V_local, Out_local);
        // Write out each result row.
        for (int i = 0; i < N; i++) {
            RowOut rOut;
            for (int j = 0; j < dv; j++) {
            #pragma HLS PIPELINE II=1
                rOut.data[j] = Out_local[i][j];
            }
            out_fifo.write(rOut);
        }
    }
}

// Store task: Reads result rows from the FIFO and writes them to global memory.
void store_task(
    hls::stream<RowOut> &out_fifo,
    fixed_t Out_global[B][N][dv]
) {
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < N; i++) {
            RowOut rOut;
            out_fifo.read(rOut);
            for (int j = 0; j < dv; j++) {
#pragma HLS PIPELINE II=1
                Out_global[b][i][j] = rOut.data[j];
            }
        }
    }
}

// Top-level function using DATAFLOW with row-based FIFO streams.
void compute_attention_HLS(
    fixed_t Q[B][N][dk],
    fixed_t K[B][N][dk],
    fixed_t V[B][N][dv],
    fixed_t Output[B][N][dv]
)
{
#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=mem1
#pragma HLS INTERFACE m_axi port=K offset=slave bundle=mem1
#pragma HLS INTERFACE m_axi port=V offset=slave bundle=mem1
#pragma HLS INTERFACE m_axi port=Output offset=slave bundle=mem2
#pragma HLS INTERFACE s_axilite port=return

    hls::stream<RowQ> q_fifo;
    hls::stream<RowK> k_fifo;
    hls::stream<RowV> v_fifo;
    hls::stream<RowOut> out_fifo;
#pragma HLS STREAM variable=q_fifo depth=128
#pragma HLS STREAM variable=k_fifo depth=128
#pragma HLS STREAM variable=v_fifo depth=128
#pragma HLS STREAM variable=out_fifo depth=128

#pragma HLS DATAFLOW
    load_task(Q, K, V, q_fifo, k_fifo, v_fifo);
    compute_task(q_fifo, k_fifo, v_fifo, out_fifo);
    store_task(out_fifo, Output);
}
