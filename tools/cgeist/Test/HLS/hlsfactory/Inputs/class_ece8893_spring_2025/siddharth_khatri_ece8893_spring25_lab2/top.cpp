#include <hls_math.h>

#include <hls_stream.h>

#include <cstring>

#include "dcl.h"
 

#define TILE_SIZE 2 
 

typedef struct {

    fixed_t row_d[N];

} row_t;
 
void copy_kv_to_bram(
    fixed_t input[B][N][dk],
    fixed_t output[B][N][dk])
{
     for (int b = 0; b < B; b++) {

      for (int i = 0; i < N; i++) {

        for (int j = 0; j < dk; j++) {

#pragma HLS PIPELINE II=1

          output[b][i][j] = input[b][i][j];

        }

      }

    }
}
  	
void process_tile(

    int tile,  

    fixed_t Q_tile[B][TILE_SIZE][dk],

    const fixed_t K_local[B][N][dk],

    const fixed_t V_local[B][N][dv],

    fixed_t Output_tile[B][TILE_SIZE][dv])

{

    hls::stream<row_t> qk_out("qk_out");

    hls::stream<row_t> softmax_out("softmax_out");

#pragma HLS stream variable=qk_out depth=8

#pragma HLS stream variable=softmax_out depth=8
 
#pragma HLS DATAFLOW
 

    for (int b = 0; b < B; b++) {

      for (int i = 0; i < TILE_SIZE; i++) {

        row_t tmp;


        const ap_fixed<32,8> scale = 1.0/sqrt((float)dk);

        for (int j = 0; j < N; j++) {

#pragma HLS PIPELINE II=1

          ap_fixed<32,8> dot = 0;

          for (int k = 0; k < dk; k++) {

#pragma HLS UNROLL factor=16

            dot += Q_tile[b][i][k] * K_local[b][j][k];

          }

          tmp.row_d[j] = dot * scale;

        }

        qk_out.write(tmp);

      }

    }
 

    for (int b = 0; b < B; b++) {

      for (int i = 0; i < TILE_SIZE; i++) {

        row_t temp = qk_out.read();

        ap_fixed<32,8> max_val = temp.row_d[0];

        for (int j = 1; j < N; j++) {

#pragma HLS PIPELINE II=1

          if (temp.row_d[j] > max_val) {

            max_val = temp.row_d[j];

          }

        }

        ap_fixed<32,8> exp_sum = 0;

        for (int j = 0; j < N; j++) {

#pragma HLS PIPELINE II=1

          temp.row_d[j] = hls::exp(temp.row_d[j] - max_val);

          exp_sum += temp.row_d[j];

        }


        for (int j = 0; j < N; j++) {

#pragma HLS PIPELINE II=1

          temp.row_d[j] = temp.row_d[j] / exp_sum;

        }

        softmax_out.write(temp);

      }

    }
 

    for (int b = 0; b < B; b++) {

      for (int i = 0; i < TILE_SIZE; i++) {

        row_t softmax_row = softmax_out.read();

        for (int j = 0; j < dv; j++) {

#pragma HLS PIPELINE II=1

          ap_fixed<32,8> result = 0;

          for (int k = 0; k < N; k++) {

#pragma HLS UNROLL factor=10

            result += softmax_row.row_d[k] * V_local[b][k][j];

          }

          Output_tile[b][i][j] = result;

        }

      }

    }

}
 

void compute_attention_HLS(

    fixed_t Q[B][N][dk],

    fixed_t K_in[B][N][dk],

    fixed_t V_in[B][N][dv],

    fixed_t Output[B][N][dv])

{

#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=qmem

#pragma HLS INTERFACE m_axi port=K_in offset=slave bundle=kmem

#pragma HLS INTERFACE m_axi port=V_in offset=slave bundle=vmem

#pragma HLS INTERFACE m_axi port=Output offset=slave bundle=omem

#pragma HLS INTERFACE s_axilite port=return
 
    fixed_t K_local[B][N][dk];

    fixed_t V_local[B][N][dv];
 
    copy_kv_to_bram(K_in,K_local);
    copy_kv_to_bram(V_in,V_local);
 
    
    fixed_t Q_tile[B][TILE_SIZE][dk];

    fixed_t Output_tile[B][TILE_SIZE][dv];
 

    for (int tile = 0; tile < N; tile += TILE_SIZE) {


      for (int b = 0; b < B; b++) {

        for (int i = 0; i < TILE_SIZE && (tile + i < N); i++) {

#pragma HLS PIPELINE II=1

          for (int k = 0; k < dk; k++) {

            Q_tile[b][i][k] = Q[b][tile + i][k];

          }

        }

      }
 

      process_tile(tile, Q_tile, K_local, V_local, Output_tile);
 

      for (int b = 0; b < B; b++) {

        for (int i = 0; i < TILE_SIZE && (tile + i < N); i++) {

#pragma HLS PIPELINE II=1

          for (int j = 0; j < dv; j++) {

            Output[b][tile + i][j] = Output_tile[b][i][j];

          }

        }

      }

    }

}

 
