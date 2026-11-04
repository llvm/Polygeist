#include <ap_fixed.h>
#include "hls_stream.h"
#include "dcl.h"

#define MAX_NONZEROS (N * M)

// For partial vector processing of B (no unroll, just pipeline):
#define VEC_SIZE 64

// Data structures for streaming between stages
struct A_elem {
    int row;      // Row index in A (and in C)
    int col;      // Column index in A -> used to find matching column in B
    data_t value; // Nonzero value from A
};

struct Mult_elem {
    int row;      // Row index for C
    int col;      // Column index for C
    data_t mult;  // Partial product (A[i][k] * B[k][j])
};

void load_A(
    data_t values_A[MAX_NONZEROS],
    int    column_indices_A[MAX_NONZEROS],
    int    row_ptr_A[N+1],
    hls::stream<A_elem> &a_stream)
{
    load_row_loop: for (int i = 0; i < N; i++) {
        #pragma HLS pipeline II=1
        int start = row_ptr_A[i];
        int end   = row_ptr_A[i+1];
        load_nz_loop: for (int idx = start; idx < end; idx++) {
            #pragma HLS pipeline II=1
            A_elem a_elem;
            a_elem.row   = i;
            a_elem.col   = column_indices_A[idx];
            a_elem.value = values_A[idx];
            a_stream.write(a_elem);
        }
    }
}

void compute_AxB(
    hls::stream<A_elem> &a_stream,
    data_t values_B[M*K],
    int    row_indices_B[M*K],
    int    col_ptr_B[M+1],
    hls::stream<Mult_elem> &mult_stream)
{
    // We'll assume up to MAX_NONZEROS elements come from a_stream.
    compute_loop: for (int i = 0; i < MAX_NONZEROS; i++) {
        #pragma HLS pipeline II=1
        if (!a_stream.empty()) {
            A_elem a_elem = a_stream.read();
            int k = a_elem.col;
            int b_start = col_ptr_B[k];
            int b_end   = col_ptr_B[k+1];
            int nnzB = b_end - b_start;

            // Process B’s elements in chunks of VEC_SIZE
            // with pipelining, no loop unrolling.
            b_chunk_loop: for (int b = 0; b < nnzB; b += VEC_SIZE) {
                #pragma HLS pipeline II=1
                // We iterate VEC_SIZE times, but do not unroll it
                b_vec_loop: for (int v = 0; v < VEC_SIZE; v++) {
                    #pragma HLS pipeline II=1
                    int idxB = b_start + b + v;
                    if ((b + v) < nnzB) {
                        int col = row_indices_B[idxB];
                        data_t b_val = values_B[idxB];

                        Mult_elem m_elem;
                        m_elem.row  = a_elem.row;
                        m_elem.col  = col;
                        m_elem.mult = a_elem.value * b_val;
                        mult_stream.write(m_elem);
                    }
                }
            }
        }
    }
}

void accumulate(
    hls::stream<Mult_elem> &mult_stream,
    data_t C[N][K])
{
    data_t tile_C[N][K];
    #pragma HLS array_partition variable=tile_C complete dim=2

    // Initialize the tile_C to zero
    init_loop_i: for (int i = 0; i < N; i++) {
        #pragma HLS pipeline II=1
        init_loop_j: for (int j = 0; j < K; j++) {
            #pragma HLS pipeline II=1
            tile_C[i][j] = 0;
        }
    }

    accumulate_loop: for (int i = 0; i < MAX_NONZEROS; i++) {
        #pragma HLS pipeline II=1
        if (!mult_stream.empty()) {
            Mult_elem m_elem = mult_stream.read();
            tile_C[m_elem.row][m_elem.col] += m_elem.mult;
        }
    }

    // Write the final results back to C
    write_back_i: for (int i = 0; i < N; i++) {
        #pragma HLS pipeline II=1
        write_back_j: for (int j = 0; j < K; j++) {
            #pragma HLS pipeline II=1
            C[i][j] = tile_C[i][j];
        }
    }
}

//---------------------------------------------------------
// Top-level: Dataflow-based, Aggressive Pipelining, No Unrolling
//---------------------------------------------------------

void sparse_matrix_multiply_HLS(
    data_t values_A[MAX_NONZEROS],
    int    column_indices_A[MAX_NONZEROS],
    int    row_ptr_A[N+1],
    data_t values_B[M*K],
    int    row_indices_B[M*K],
    int    col_ptr_B[M+1],
    data_t C[N][K])
{
    #pragma HLS interface m_axi port=values_A         offset=slave bundle=mem1
    #pragma HLS interface m_axi port=column_indices_A   offset=slave bundle=mem2
    #pragma HLS interface m_axi port=row_ptr_A          offset=slave bundle=mem3
    #pragma HLS interface m_axi port=values_B           offset=slave bundle=mem4
    #pragma HLS interface m_axi port=row_indices_B      offset=slave bundle=mem5
    #pragma HLS interface m_axi port=col_ptr_B          offset=slave bundle=mem6
    #pragma HLS interface m_axi port=C                  offset=slave bundle=mem7
    #pragma HLS interface s_axilite port=return

    // Dataflow: run load, compute, and accumulate concurrently
    #pragma HLS dataflow

    hls::stream<A_elem>    a_stream("A_stream");
    hls::stream<Mult_elem> mult_stream("Mult_stream");
    #pragma HLS STREAM variable=a_stream     depth=64
    #pragma HLS STREAM variable=mult_stream  depth=64

    load_A(values_A, column_indices_A, row_ptr_A, a_stream);
    compute_AxB(a_stream, values_B, row_indices_B, col_ptr_B, mult_stream);
    accumulate(mult_stream, C);
}