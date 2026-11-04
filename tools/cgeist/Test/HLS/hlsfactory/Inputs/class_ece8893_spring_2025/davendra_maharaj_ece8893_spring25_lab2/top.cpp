#include "dcl.h"
#include <hls_math.h>
#include <hls_stream.h>

static const ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);

// Function to load data into stream
void loadData(fixed_t in[B][N][dk], hls::stream<fixed_t> &inStream)
{
#pragma HLS INLINE off
    load_batch: for (int b = 0; b < B; b++)
    {
        load_row: for (int i = 0; i < N; i++)
        {
            load_col: for (int j = 0; j < dk; j++)
            {
                inStream << in[b][i][j];
            }
        }
    }
}


void softmax_HLS(fixed_t matrix[N][N])
{
#pragma HLS INLINE

    soft_QK_row: for ( int row = 0; row < N; ++row )
    {
#pragma HLS PIPELINE off
        // Reduce precision
        ap_fixed<16, 5> max_val = matrix[row][0];

        // Reduce unrolling
        QK_row_max: for ( int i = 0, dim_base = 0; i < N/50; i++, dim_base+=50 )
        {
#pragma HLS PIPELINE
            QK_row_max_offset: for ( int dim_offset = 0 ; dim_offset < 50; dim_offset++ )
            {
#pragma HLS UNROLL
                // int dim = dim_base + dim_offset;
                max_val = hls::max(max_val, matrix[row][dim_base + dim_offset]);
            }
        }

        ap_fixed<16, 8> sum = 0;
        ap_fixed<16, 8> exp_val[N];
#pragma HLS ARRAY_PARTITION variable = exp_val cyclic factor = 50 dim = 1

        QK_row_sum: for ( int i = 0, dim_base = 0; i < N/10; i++, dim_base+=10 )
        {
#pragma HLS PIPELINE
            QK_row_sum_offset: for ( int dim_offset = 0; dim_offset < 10; dim_offset++ )
            {
#pragma HLS UNROLL
                // int dim = dim_base + dim_offset;
                ap_fixed<16, 8> temp = matrix[row][dim_base + dim_offset] - max_val;
                exp_val[dim_base + dim_offset] = hls::exp(temp);
                sum += exp_val[dim_base + dim_offset];
            }
        }

        ap_fixed<16, 8> sum_recip = ap_fixed<16, 8>(1) / sum;

        QK_row_exp: for ( int i = 0, dim_base = 0; i < N/50; i++, dim_base+=50 )
        {
#pragma HLS PIPELINE
            QK_row_exp_offset: for ( int dim_offset = 0; dim_offset < 50; dim_offset++ )
            {
#pragma HLS UNROLL
                // int dim = dim_base + dim_offset;
                matrix[row][dim_base + dim_offset] = exp_val[dim_base + dim_offset] * sum_recip;
            }
        }
    }
}

// // Function to compute Attention = softmax(Q*K^T)*V from streams and send to output array port
void computeAttention(
    hls::stream<fixed_t> &QStream, 
    hls::stream<fixed_t> &KStream,
    hls::stream<fixed_t> &VStream,
    hls::stream<fixed_t> &attentionStream
    // fixed_t out[B][N][dv]
)
{
#pragma HLS INLINE off

    // Declare arrays to store Q, K, and QK
    static fixed_t Q[N][dk];
    static fixed_t K[N][dk];
    static fixed_t V[N][dv];
    static fixed_t QK[N][N];
#pragma HLS ARRAY_PARTITION variable = Q cyclic factor = 64 dim = 2
#pragma HLS ARRAY_PARTITION variable = K cyclic factor = 64 dim = 2
// #pragma HLS ARRAY_PARTITION variable = Q complete dim = 2
// #pragma HLS ARRAY_PARTITION variable = K complete dim = 2
#pragma HLS ARRAY_PARTITION variable = V cyclic factor = 50 dim = 1
#pragma HLS ARRAY_PARTITION variable = QK cyclic factor = 50 dim = 2

    // Process B batches
    stream_batch_compute: for ( int b = 0; b < B; ++b )
    {
        // Step 1: Copy data from stream to arrays
        stream_row_compute: for ( int i = 0; i < N; ++i )
        {
            stream_col_compute: for ( int j = 0; j < dk; ++j )
            {
                QStream >> Q[i][j];
                KStream >> K[i][j];
                VStream >> V[i][j];
            }
        }

        // Step 2: Compute Q*K^T
        QK_row: for ( int i = 0; i < N; ++i )
        {
            QK_col: for ( int j = 0; j < N; ++j )
            {
#pragma HLS PIPELINE
                ap_fixed<16, 8> sum = 0;
                QK_inner_base: for ( int k = 0, dim_base = 0; k < dk/64; ++k, dim_base+=64 )
                {
#pragma HLS PIPELINE
                    QK_inner_offset: for ( int offset = 0; offset < 64; offset++ )
                    {
#pragma HLS UNROLL
                        int dim = dim_base + offset;
                        sum += Q[i][dim] * K[j][dim];
                    }
                }
                QK[i][j] = sum * scale;
            }
        }

        // Step 3: Apply softmax(Q*K^T)
        softmax_HLS(QK);

        // Step 4: Compute Attention = softmax(Q*K^T)*V 
        attn_row: for ( int i = 0; i < N; ++i )
        {
            attn_col: for ( int j = 0; j < dv; ++j )
            {
#pragma HLS PIPELINE
                ap_fixed<24, 8> sum_val = 0;
                attn_inner_base: for ( int k = 0, dim_base = 0; k < N/50; ++k, dim_base+=50 )
                {
#pragma HLS PIPELINE
                    attn_inner_offset: for ( int offset = 0; offset < 50; offset++ )
                    {
#pragma HLS UNROLL
                        int dim = dim_base + offset;
                        sum_val += QK[i][dim] * V[dim][j];
                    }
                }
                attentionStream << sum_val;
            }
        }
    }
}

// Function to write data from stream to array
void writeData(hls::stream<fixed_t> &outStream, fixed_t out[B][N][dv])
{
#pragma HLS INLINE off

    for ( int b = 0; b < B; ++b )
    {
        for ( int i = 0; i < N; ++i )
        {
            for ( int j = 0; j < dv; ++j )
            {
                fixed_t val;
                outStream >> val;
                out[b][i][j] = val;
            }
        }
    }
}

// Top-level function to compute attention
void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv])
{
#pragma HLS interface m_axi port=Q offset=slave bundle=memQ
#pragma HLS interface m_axi port=K offset=slave bundle=memK
#pragma HLS interface m_axi port=V offset=slave bundle=memV
#pragma HLS interface m_axi port=Output offset=slave bundle=memO
#pragma HLS interface s_axilite port=return

#pragma HLS DATAFLOW

    hls::stream<fixed_t> QStream;
    hls::stream<fixed_t> KStream;
    hls::stream<fixed_t> VStream;
    hls::stream<fixed_t> attentionStream;

#pragma HLS stream variable=QStream depth=256
#pragma HLS stream variable=KStream depth=256
#pragma HLS stream variable=VStream depth=256
#pragma HLS stream variable=attentionStream depth=256

    // Load Q, K, and V tensors into streams
    loadData(Q, QStream);
    loadData(K, KStream);
    loadData(V, VStream);

    // Compute Attention
    computeAttention(QStream, KStream, VStream, attentionStream);

    // Write output to Output array
    writeData(attentionStream, Output);
}