#include "dcl.h"

void compute_attention_on_q_row(fixed_t Q_local_row[dk], fixed_t K_local[N][dk], fixed_t attention[N]) {
// partition these arrays in these dimensions to allow for the inner-most loop to be unrolled
#pragma HLS array_partition variable=Q_local_row dim=1 complete
#pragma HLS array_partition variable=K_local dim=2 complete
    const ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);
    for (int j = 0; j < N; ++j) {
        ap_fixed<32, 8> sum_local[dk]; // made this to solve DSP utilization issues in this unroll
        #pragma HLS array_partition variable=sum_local dim=1 complete
        for (int k = 0; k < dk; ++k) {
        #pragma HLS unroll
            sum_local[k] = Q_local_row[k] * K_local[j][k]; 
        }

        // again, splitting of the summation to save DSP usage
        ap_fixed<32, 8> sum = 0;
        for(int k =0; k < dk; k++) {
        #pragma HLS unroll
            sum+=sum_local[k];
        }

        attention[j] = sum * scale;
    }
}

void compute_softmax_on_row(fixed_t attention[N], fixed_t attention_flow[N]) {
// increasing this number increases usage in c synth, but not yet in 
// verilog synth, however it does decrease the co-sim clock cycles
#define exp_unroll_factor 6 
#pragma HLS array_partition variable=attention dim=1 complete
    ap_fixed<32, 8> max_val = attention[0];
    for (int j = 1; j < N; ++j) {
    #pragma HLS pipeline // pipeline because loop iterations are dependent. already was auto-pipelined by HLS
        if (attention[j] > max_val) {
            max_val = attention[j];
        }
    }

    ap_fixed<32, 8> sum = 0;
    // need to be careful here since we are using around 11% of the DSPs here, and we are at 74% usage on DSPs
    // so in theory we can add another 22% and be fine. it appears, however, that parallelizing this did not 
    // increase the final DSP resource usage (~14%) so im doubling this again to 6. 
    for (int j = 0; j < N; j+=exp_unroll_factor) {
        #pragma HLS pipeline // this should already be pipelined, but being explicit
        for(int jj=0; jj<exp_unroll_factor; jj++) {
            #pragma HLS unroll // partial loop unroll
            attention_flow[j+jj] = hls::exp(attention[j+jj] - max_val);
            sum += attention_flow[j+jj];
        }
    }

    // trivial unroll, no dependencies between loop iterations.
    for (int j = 0; j < N; ++j) {
    #pragma HLS unroll
        attention_flow[j] /= sum;
    }
}

void compute_attention_v_vec_mul_on_row(fixed_t attention[N], fixed_t V_local[N][dv], fixed_t output_vec_local[dv]) {
// this partitioning allows for the unrolling of the inner-most loop
#pragma HLS array_partition variable=attention dim=1 complete
#pragma HLS array_partition variable=V_local dim=1 complete
    for (int j = 0; j < dv; ++j) {
        ap_fixed<32, 8> sum_local[N]; // made this to solve DSP utilization issues in this unroll
        #pragma HLS array_partition variable=sum_local dim=1 complete
        // the summation is carried out in a seperately unrolled step to ensure DSP usage stays low
        for (int k = 0; k < N; ++k) {
        #pragma HLS unroll
            sum_local[k] = attention[k] * V_local[k][j];
        }
        
        // it is interesting that even though this would seem to have loop dependent, this still unrolls
        ap_fixed<32, 8> sum = 0;
        for(int k =0; k < N; k++) {
        #pragma HLS unroll
            sum+=sum_local[k];
        }
        output_vec_local[j] = sum;
    }
}

// saves holding of multiple attention vectors in BRAM, now only one needs to exist
void handle_row_operations(fixed_t Q_local[N][dk], 
    fixed_t K_local[N][dk], 
    fixed_t V_local[N][dv], 
    fixed_t output_matrix_local[N][dv]) {
    for(int i=0; i<N; i++) {
    #pragma HLS dataflow // this helped, but not that much (~30k cycles in C synthesis estimate of cycles)
        fixed_t attention_out[N];
        fixed_t attention_flow[N]; // added for dataflow idealization
        compute_attention_on_q_row(Q_local[i], K_local, attention_out);
        compute_softmax_on_row(attention_out, attention_flow);
        compute_attention_v_vec_mul_on_row(attention_flow, V_local, output_matrix_local[i]);
    }
}

void load_q_k_v(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Q_local[N][dk], fixed_t K_local[N][dk], fixed_t V_local[N][dv], size_t B_index) {
    for(size_t j = 0; j < N; j++)
    {
        for(size_t k = 0; k < dk; k++)
        {
            Q_local[j][k] = Q[B_index][j][k];
            K_local[j][k] = K[B_index][j][k];
        }

        for(size_t kk = 0; kk < dv; kk++)
        {
            V_local[j][kk] = V[B_index][j][kk];
        }
    }
}

void output_mat(fixed_t output_local[N][dv], fixed_t Output[B][N][dv], size_t B_index) {
    for(size_t j = 0; j < N; j++) {
        for(size_t kk = 0; kk < dv; kk++)
        {
            Output[B_index][j][kk] = output_local[j][kk];
        }
    }
}

void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
#pragma HLS interface m_axi port=Q offset=slave bundle=mem1
#pragma HLS interface m_axi port=K offset=slave bundle=mem1
#pragma HLS interface m_axi port=V offset=slave bundle=mem1
#pragma HLS interface m_axi port=Output offset=slave bundle=mem2

#pragma HLS interface s_axilite port=return

    // first-things-first, we need to copy the args to BRAM from DRAM:
    // next, allocate our local output that we will copy to the back out

    for(size_t i = 0; i < B; i++) {
    #pragma HLS dataflow 
    // our BRAM right now is fine in the final synthesis so we should also be able to data flow here
        fixed_t Q_local[N][dk];
        fixed_t K_local[N][dk];
        fixed_t V_local[N][dv];
        load_q_k_v(Q, K, V, Q_local, K_local, V_local, i);
        fixed_t output_local[N][dv];
        handle_row_operations(Q_local, K_local, V_local, output_local);
        output_mat(output_local, Output, i);
    }
}
