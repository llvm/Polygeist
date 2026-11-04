
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <tuple>

#include <ap_fixed.h>
#include <hls_math.h>
#include <hls_math.h>
#include <hls_vector.h>
#include <hls_stream.h>
#include <hls_task.h>

#include <stdlib.h>
#include <cstdint>

#define N 64  // Rows of A and C
#define M 64  // Columns of A and Rows of B
#define K 64  // Columns of B and C
#define SPARSITY 0.8  // Percentage of non-zero elements

typedef ap_fixed<16, 5> data_t;
typedef ap_uint<16> index_t;

struct C_entry
{
    data_t to_sum;
    index_t ith_index;
    index_t jth_index;
};

struct ent
{
    data_t val;
    index_t index;
};

struct b_ent
{
    ent a_ent;
    data_t val;
    index_t index;
};

struct cs_ent
{
    index_t lhs;
    index_t rhs;
};

struct PartialSum {
    int col;      // column index in matrix C
    data_t value; // partial product value
};

using A_ent_stream_type = hls::stream<ent, N>;
using B_ent_stream_type = hls::stream<hls::vector<b_ent, M>, K>;

using control_stream = hls::stream<cs_ent, N+1>;
using ind_pair_stream = hls::stream<std::tuple<index_t, index_t, data_t>>;

void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], 
                                data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t C[N][K]);



// IDEAS:
// I could use a stream per each row of C and in stream itself it will have the column index and value from which to accumulate into
// the row / col id
    // choosing rows because we are already going through those in the first place i guess


// read_B_cols(A_stream, col_ptr_B, values_B, row_indices_B, B_stream);
// Perform Sparse x Sparse Multiplication
// for (int i = 0; i < N; i++) { //
//     // #pragma HLS unroll
//     for (int idx_A = row_ptr_A[i]; idx_A < row_ptr_A[i + 1]; idx_A++) {
//         int  = local_column_indices_A[idx_A]; // Column index of A (k) -> its not really "column" per-se, it is 
//         data_t value_A = local_values_A[idx_A];

//         // Iterate over columns of B corresponding to row k
            //
//         for (int idx_B = col_ptr_B[k]; idx_B < col_ptr_B[k + 1]; idx_B++) {
//             
//                 int j = local_row_indices_B[idx_B]; // Row index of B
//             data_t value_B = local_values_B[idx_B];
        
        // there can be multiple iterations of this loop at each i , j pair, 
        // however within each loop iteration here, i,j is unique. how can we parallelize this?
            // im thinking a vector containing the A*B values and the i,j indexes within a stream 
                // what about a stream containing just the A*B vals and the i,j indexes, we can use the start / stop idx_B counts
                // to set the length of how long we need to read from the stream for we can keep 


    // one problem im thinking about is the stream read contention / out of order value streaming
        // if the a value, C index pair and B value is streamed in as a copy we should be ok
            // however, how do we get the inner count number of iterations? 
                // dont care, will use an hls::task
//             // Accumulate the product into C[i][j]
//             local_C[i][j] += value_A * value_B;
//         }
//     }
// }