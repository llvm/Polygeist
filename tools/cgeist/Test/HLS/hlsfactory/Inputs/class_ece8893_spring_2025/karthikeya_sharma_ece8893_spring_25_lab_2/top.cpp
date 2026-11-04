// ECE8893 Parallel Programming for FPGA Lab-2 : Efficient Implementation of Attention Computation
// Author: Karthikeya Sharma
// Description : This code contains the implementation for compute_attention_HLS kernel. To achieve a desirable performance, concepts such as Task Level Parallelism and Streaming have been implemented. However, there is a scope of improvement by adding hls::vector to optimize the computations.
// Last Modified : 2/28/2025

#include "dcl.h"

// @param matrix : The attention matrix fed per batch.
void softmax_HLS(fixed_t matrix[N][N])
{
    //Partitioning the attention array on BRAM.
    #pragma HLS ARRAY_PARTITION variable=matrix type=cyclic dim=2 factor=N/2

    for (int i = 0; i < N; ++i)
    {
    #pragma HLS PIPELINE off

        //Reducing mantissa-bit precision helps in reducing DSP utilization leaving room for optimizing any other bottlenecks.
        ap_fixed<18, 8> max_val = matrix[i][0];
        ap_fixed<18, 8> sum = 0;

        //Declaring an exponent array to reduce the dependencies of the for loops below i.e., every for loop depends on the previous "for" loop to use results like "max_val" and "sum".
        ap_fixed<18, 8> exp[N];
        
        //Partitioning the array by a factor of 4
        #pragma HLS ARRAY_PARTITION variable=exp type=cyclic dim=1 factor=N/25

        //Replaced the "if" condition with hls::max operation to reduce MUX's on the critical path providing an efficient implementation.
        for (int j = 1; j < N; ++j)
        {
        #pragma HLS UNROLL factor=N/2
        	max_val = hls::max(max_val, static_cast<ap_fixed<18, 8>>(matrix[i][j]));
        }

        //Loop Tiling + Unrolling for faster accumulation.
        for (int j = 0; j < N; j+=N/25)
        {
        #pragma HLS PIPELINE
            for (int jj = 0; jj < N/25; jj++)
            {
        #pragma HLS UNROLL
                exp[j+jj] = hls::exp(matrix[i][j+jj] - max_val);
                sum += exp[j+jj];
            }
        }

        //Converted the division operation "exp[j]/sum" to multiplication. This enables mapping to DSP, and cause not to blow the LUT resource usage out of proportion.
        ap_fixed<18, 8> inv_sum = ap_fixed<18, 8>(1.0)/sum;

        //Loop Unrolling for parallelism.
        for (int j = 0; j < N; j++)
        {
        #pragma HLS UNROLL factor=N/2
            matrix[i][j] = exp[j] * inv_sum;
        }
    }
}

// @param hls::stream<fixed_t> &QStream : The Query Stream.
// @param hls::stream<fixed_t> &KStream : The Key Stream.
// @param hls::stream<fixed_t> &VStream : The Value Stream.
// @param fixed_t Output[B][N][dv] : The final Attention output.

void computeAttention(hls::stream<fixed_t> &QStream, hls::stream<fixed_t> &KStream, hls::stream<fixed_t> &VStream, fixed_t Output[B][N][dv])
{
    //Declaring arrays to accumulate the streamed array values into the BRAM.
    static fixed_t Q_BRAM[N][dk];
    static fixed_t K_BRAM[N][dk];
    static fixed_t V_BRAM[N][dv];
    static fixed_t attention_BRAM[N][N];

    //Partitioning the Q,K,V matrics for memory parallism.
    #pragma HLS ARRAY_PARTITION variable=Q_BRAM type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=K_BRAM type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=V_BRAM type=cyclic factor=N/2 dim=1

    //Precomputed 1/sqrt(dk) to save time and resource.
    ap_fixed<32, 8> scale = 0.08838834764831843;

    //Start processing the data in batches
    for (int b = 0; b < B; b++)
    {
        //Copy data into BRAM
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < dk; j++)
            {
                QStream >> Q_BRAM[i][j];
                KStream >> K_BRAM[i][j];
                VStream >> V_BRAM[i][j];
            }
        }

        // Compute Q * K^T
        //Flattened and pipelined the original outer loops.

        for (int i = 0; i < N*N; ++i)
        {
    #pragma HLS PIPELINE
            ap_fixed<32, 8> sum = 0;
            for (int k = 0; k < dk; ++k)
            {
    #pragma HLS UNROLL factor=N/2
                sum += Q_BRAM[i/N][k] * K_BRAM[i%N][k];
            }

            attention_BRAM[i/N][i%N] = sum * scale;
        }

        // Apply softmax
        softmax_HLS(attention_BRAM);

        // Compute Attention * V
        //Flattened and pipelined the original outer loops.
        for (int i = 0; i < N*dv; ++i)
        {
        #pragma HLS PIPELINE
            ap_fixed<32, 8> sum = 0;

            for (int k = 0; k < N; ++k)
            {
            #pragma HLS UNROLL factor=N/2
                sum += attention_BRAM[i/dv][k] * V_BRAM[k][i%dv];
            }

            Output[b][i/dv][i%dv] = sum;
        }
    }
}

// @param fixed_t Q[B][N][dk] : The Query Matrix.
// @param hls::stream<fixed_t> &QStream : The Query Stream.

void loadQuery(fixed_t Q[B][N][dk], hls::stream<fixed_t> &Q_Stream)
{
#pragma HLS INLINE off
    for (int b = 0; b < B; b++)
    {
        for (int i = 0; i < N*dk; i++)
        {
            Q_Stream << Q[b][i/dk][i%dk];
        }
    }
}

// @param fixed_t K[B][N][dk] : The Key Matrix.
// @param hls::stream<fixed_t> &KStream : The Key Stream.

void loadKey(fixed_t K[B][N][dk], hls::stream<fixed_t> &K_Stream)
{
#pragma HLS INLINE off
    for (int b = 0; b < B; b++)
    {
        for (int i = 0; i < N*dk; i++)
        {
            K_Stream << K[b][i/dk][i%dk];
        }
    }
}

// @param fixed_t V[B][N][dv] : The Value Matrix.
// @param hls::stream<fixed_t> &VStream : The Value Stream.

void loadValue(fixed_t V[B][N][dk], hls::stream<fixed_t> &V_Stream)
{
#pragma HLS INLINE off
    for (int b = 0; b < B; b++)
    {
        for (int i = 0; i < N*dv; i++)
        {
            V_Stream << V[b][i/dv][i%dv];
        }
    }
}

// @param fixed_t Q[B][N][dk] : The Query Matrix.
// @param fixed_t K[B][N][dk] : The Key Matrix.
// @param fixed_t V[B][N][dv] : The Value Matrix.
// @param fixed_t Output[B][N][N] : The Output Matrix.

void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv])
{
#pragma HLS interface m_axi port=Q offset=slave bundle=memQ max_read_burst_length=256
#pragma HLS interface m_axi port=K offset=slave bundle=memK max_read_burst_length=256
#pragma HLS interface m_axi port=V offset=slave bundle=memV max_read_burst_length=256
#pragma HLS interface m_axi port=Output offset=slave bundle=memO max_read_burst_length=256
#pragma HLS interface s_axilite port=return
#pragma HLS DATAFLOW

/////////////////////////// The DATAFLOW Region ///////////////////////////

    //Create HLS Streams
    hls::stream<fixed_t,dk> Q_Stream;
    hls::stream<fixed_t,dk> K_Stream;
    hls::stream<fixed_t,dk> V_Stream;

    //load arrays into local BRAMs
    loadQuery(Q, Q_Stream);
    loadKey(K, K_Stream);
    loadValue(V, V_Stream);

    // Compute Q * K^T
    computeAttention(Q_Stream, K_Stream, V_Stream, Output);
}