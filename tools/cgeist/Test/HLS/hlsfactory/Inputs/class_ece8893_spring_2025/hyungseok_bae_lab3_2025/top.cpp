#include "dcl.h"

// VECTOR_SIZE는 한 번에 처리할 non-zero element 수

void load_A(
    const vec_data_t values_A[], const vec_idx_t column_indices_A[], const int row_ptr_A[],
    hls::stream<vec_data_t>& val_stream,
    hls::stream<vec_idx_t>& idx_stream,
    hls::stream<int>& len_stream)
{
LOAD_ROWS:
    for (int i = 0; i < N; i++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
        int row_len = row_ptr_A[i + 1] - row_ptr_A[i];
        len_stream.write(row_len);

        int start = row_ptr_A[i];
        int end = row_ptr_A[i + 1];
        int idx = start;

        for (int b = 0; b < (row_len + VECTOR_SIZE - 1) / VECTOR_SIZE; b++) {
#pragma HLS PIPELINE II=1
            vec_data_t tmp_val;
            vec_idx_t tmp_idx;

            for (int v = 0; v < VECTOR_SIZE; v++) {
                tmp_val[v] = 0;
                tmp_idx[v] = 0;
            }
            
            for (int v = 0; v < VECTOR_SIZE; v++) {
                if (idx < end) {
                    int global_vec_idx = idx / VECTOR_SIZE;
                    int offset = idx % VECTOR_SIZE;

                    tmp_val[v] = values_A[global_vec_idx][offset];
                    tmp_idx[v] = column_indices_A[global_vec_idx][offset];

                    idx++;
                } else {
                    tmp_val[v] = 0;
                    tmp_idx[v] = 0;
                }
            }

            val_stream.write(tmp_val);
            idx_stream.write(tmp_idx);
        }
    }
}


void load_B(
    const vec_data_t values_B[], const vec_idx_t row_indices_B[],
    vec_data_t local_B[], vec_idx_t local_row_idx_B[])
{
LOAD_B:
    for (int i = 0; i < (M*K + VECTOR_SIZE - 1) / VECTOR_SIZE; i++) {
#pragma HLS PIPELINE II=1
        local_B[i] = values_B[i];
        local_row_idx_B[i] = row_indices_B[i];
    }
}

    



// Compute stage with Multiply-Merge Pipeline applied
void compute(
    hls::stream<hls::vector<data_t, VECTOR_SIZE>>& val_stream,
    hls::stream<hls::vector<int, VECTOR_SIZE>>& idx_stream,
    hls::stream<int>& len_stream,
    const vec_data_t local_B[], const vec_idx_t local_row_idx_B[], const int col_ptr_B[],
    hls::stream<hls::vector<data_t, K>>& c_stream)
{
#pragma HLS INLINE off

COMPUTE_ROWS:
    for (int i = 0; i < N; i++) {
        data_t local_row_C[K] = {0};
#pragma HLS ARRAY_PARTITION variable=local_row_C complete dim=1

        int row_len = len_stream.read();

        MULT_MERGE_PIPELINE:
        for (int b = 0; b < (row_len + VECTOR_SIZE - 1) / VECTOR_SIZE; b++) {
#pragma HLS PIPELINE II=1
            hls::vector<data_t, VECTOR_SIZE> val_vec = val_stream.read();
            hls::vector<int, VECTOR_SIZE> idx_vec = idx_stream.read();

            int valid_elems = (b < row_len / VECTOR_SIZE) ? VECTOR_SIZE : (row_len % VECTOR_SIZE);

            MULT_ACCUM_LOOP:
            for (int e = 0; e < valid_elems; e++) {
#pragma HLS UNROLL
                int k = idx_vec[e];
                data_t val_a = val_vec[e];

                int col_start = col_ptr_B[k];
                int col_end = col_ptr_B[k + 1];

                int vec_idx = -1;
                vec_data_t vecB;
                vec_idx_t vecRowB;

                INNER_MULTIPLY_LOOP:
                for (int idx_B = col_start; idx_B < col_end; idx_B++) {
#pragma HLS PIPELINE II=1
                    int curr_vec_idx = idx_B / VECTOR_SIZE;
                    int offset = idx_B % VECTOR_SIZE;

                    if (curr_vec_idx != vec_idx) {
                        vecB = local_B[curr_vec_idx];
                        vecRowB = local_row_idx_B[curr_vec_idx];
                        vec_idx = curr_vec_idx;
                    }

                    int j = vecRowB[offset];
                    data_t val_b = vecB[offset];

                    // 곧바로 accumulation
                    data_t prod;
#pragma HLS BIND_OP variable=prod op=mul impl=DSP
                    prod = val_a * val_b;
                    local_row_C[j] += prod;
                }
            }
        }

        // 결과를 곧바로 stream으로 출력
        hls::vector<data_t, K> c_row_vec;
        OUTPUT_VEC_LOOP:
        for (int col = 0; col < K; col++) {
#pragma HLS UNROLL
            c_row_vec[col] = local_row_C[col];
        }

        c_stream.write(c_row_vec);
    }
}






// Store stage
void store_result(hls::stream<hls::vector<data_t, K>>& c_stream, hls::vector<data_t, K> C_vec[N]) {
    STORE_ROWS:
        for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
            C_vec[i] = c_stream.read();
        }
    }
    



// Top function
void sparse_matrix_multiply_HLS(
    const vec_data_t values_A[], const vec_idx_t column_indices_A[], const int row_ptr_A[],
    const vec_data_t values_B[], const vec_idx_t row_indices_B[], const int col_ptr_B[],
    hls::vector<data_t, K> C_vec[N])
{
#pragma HLS interface m_axi port=values_A offset=slave bundle=mem1 depth=((N*M+VECTOR_SIZE-1)/VECTOR_SIZE)
#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem2 depth=((N*M+VECTOR_SIZE-1)/VECTOR_SIZE)
#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem3 depth=(N+1)

#pragma HLS interface m_axi port=values_B offset=slave bundle=mem4 depth=((M*K+VECTOR_SIZE-1)/VECTOR_SIZE)
#pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem5 depth=((M*K+VECTOR_SIZE-1)/VECTOR_SIZE)
#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem6 depth=(M+1)

#pragma HLS interface m_axi port=C_vec offset=slave bundle=mem7 depth=N max_write_burst_length=64

#pragma HLS interface s_axilite port=return

#pragma HLS dataflow

    
    static vec_data_t local_B[M*K/VECTOR_SIZE];
    static vec_idx_t local_row_idx_B[M*K/VECTOR_SIZE];

    #pragma HLS ARRAY_PARTITION variable=local_B cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_row_idx_B cyclic factor=8 dim=1



    hls::stream<hls::vector<data_t, VECTOR_SIZE>> val_stream;
    hls::stream<hls::vector<int, VECTOR_SIZE>> idx_stream;
    hls::stream<int> len_stream;
    hls::stream<hls::vector<data_t, K>> c_stream;  // 새롭게 추가된 stream
#pragma HLS STREAM variable=val_stream depth=24
#pragma HLS STREAM variable=idx_stream depth=24
#pragma HLS STREAM variable=len_stream depth=20
#pragma HLS STREAM variable=c_stream depth=2
    
    load_A(values_A, column_indices_A, row_ptr_A, val_stream, idx_stream, len_stream);
    load_B(values_B, row_indices_B, local_B, local_row_idx_B);
    compute(val_stream, idx_stream, len_stream, local_B, local_row_idx_B, col_ptr_B, c_stream);
    store_result(c_stream, C_vec);
}

