#include "dcl.h"


void compute_attention(fixed_t Q[N][dk], fixed_t K[N][dk], fixed_t V[N][dv], fixed_t Output[N][dv])
{

    // Compute by batch
    fixed_t Q_l_row[dk];
    fixed_t K_l[N][dk];
    fixed_t V_l[N][dv];
    fixed_t attention_row[N];
    #pragma HLS array_partition variable=Q_l_row cyclic factor=Fac
    #pragma HLS array_partition variable=attention_row cyclic factor=Fac_N
    #pragma HLS array_partition variable=K_l dim=2 cyclic factor=Fac
    #pragma HLS array_partition variable=V_l dim=1 cyclic factor=Fac_N


    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);


    // Wider bus transaction to read K matrix
    ap_uint<512>* K_wide = reinterpret_cast<ap_uint<512>*>(K);
    for (int i = 0; i < N; i++)
    {
    #pragma HLS PIPELINE II=1
        for (int j = 0; j < DFac; j++)
        {
        // #pragma HLS PIPELINE II=1
            ap_uint<512> temp = K_wide[i * DFac + j];
            for (int jj = 0; jj < Fac; jj++)
            {
                #pragma HLS UNROLL
                ap_fixed<16, 5> slice;
                // Bit-to-bit transaction
                slice.range(15, 0) = temp.range(16 * (jj + 1) - 1, 16 * jj);

                K_l[i][j * Fac + jj] = slice;
            }
        }
    }

    // Since V is not transposed in SW, we cannot use wider bus.
    for (int j = 0; j < D; ++j)
    {
    #pragma HLS PIPELINE
        for (int i = 0; i < N; ++i)
        {
            V_l[i][j] = V[i][j];
        }
    }

    // Wider bus transaction for Q row vector and Output
    ap_uint<512>* Q_wide = reinterpret_cast<ap_uint<512>*>(Q);
    for (int i = 0; i < N; ++i)
    {
        LoadQ: for (int k = 0; k < DFac; ++k)
        {
        #pragma HLS PIPELINE
            ap_uint<512> temp = Q_wide[i * DFac + k];
            for (int kk = 0; kk < Fac; ++kk)
            {
                #pragma HLS UNROLL
                ap_fixed<16, 5> slice;
                slice.range(15, 0) = temp.range(16 * (kk + 1) - 1, 16 * kk);
                Q_l_row[k * Fac + kk] = slice;
            }
        }
        
        ap_fixed<32, 8> max_val = -1e20;
        ComputeAttnScore:
        {
            for (int j = 0; j < N; ++j)
            {
                ap_fixed<32, 8> sum = 0;
            #pragma HLS PIPELINE   
                for (int k = 0; k < dk; ++k)
                {
                // #pragma HLS PIPELINE
                    sum += Q_l_row[k] * K_l[j][k];
                }
                attention_row[j] = sum * scale;

                if (attention_row[j] > max_val) max_val = attention_row[j];

            }
        }
        

        // Apply softmax
        SoftMax:
        {   
            ap_fixed<32, 8> sum = 0;

            for (int j = 0; j < NFac_N; ++j)
            {
            #pragma HLS PIPELINE II=1
                for (int jj = 0; jj < Fac_N; ++jj)
                {
                    attention_row[j*Fac_N+jj] = hls::exp(attention_row[j*Fac_N+jj] - max_val);
                    sum += attention_row[j*Fac_N+jj];
                }
            }

            ap_fixed<32, 8> inv_sum = 1.0 / (float)sum;

            for (int j = 0; j < NFac_N; ++j)
            {
            #pragma HLS PIPELINE II=1
                for (int jj = 0; jj < Fac_N; ++jj)
                {
                    attention_row[j*Fac_N+jj] *= inv_sum;
                }
            }
        }


        ComputeOutput: for (int j = 0; j < dv; ++j)
        {
        #pragma HLS PIPELINE II=1
            ap_fixed<32, 8> sum = 0;
            for (int k = 0; k < NFac_N; ++k)
            {
                for (int kk = 0; kk < Fac_N; ++kk)
                {
                    sum += attention_row[k*Fac_N+kk] * V_l[k*Fac_N+kk][j];
                }
            }
            // Will automatically burst on write
            Output[i][j] = sum;
        }
    }
}


void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv])
{
#pragma HLS interface m_axi port = Q offset = slave bundle = mem0 max_widen_bitwidth=512
#pragma HLS interface m_axi port = K offset = slave bundle = mem1 max_widen_bitwidth=512
#pragma HLS interface m_axi port = V offset = slave bundle = mem2 max_widen_bitwidth=512
#pragma HLS interface m_axi port = Output offset = slave bundle = mem3 max_widen_bitwidth=512

#pragma HLS interface s_axilite port = return

    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);

    for (int b = 0; b < B; ++b)
    {
        compute_attention(Q[b], K[b], V[b], Output[b]);
    }
}