// sparse_matrix_multiply_hls.cpp
#include "dcl.h"
#include <ap_int.h>

void load_matrix_data16(data_t target_matrix[], const int num_words, data_t out[])
{
    ap_uint<512>* target_matrix_wide = reinterpret_cast<ap_uint<512>*>(target_matrix);
    for (int j = 0; j < num_words; j++)
    {
    #pragma HLS pipeline
        ap_uint<512> temp = target_matrix_wide[j];
        for (int jj = 0; jj < 32; jj++)
        {
            #pragma HLS unroll
            data_t slice;
            // Bit-to-bit transaction
            slice.range(15, 0) = temp.range(16 * (jj + 1) - 1, 16 * jj);
            out[j * 32 + jj] = slice;
        }
    }
}

void load_matrix_int32(int target_matrix[], const int num_words, data_int out[])
{
    ap_uint<512>* target_matrix_wide = reinterpret_cast<ap_uint<512>*>(target_matrix);
    for (int j = 0; j < num_words; j++)
    {
    #pragma HLS pipeline
        ap_uint<512> temp = target_matrix_wide[j];
        for (int jj = 0; jj < 16; jj++)
        {
            #pragma HLS unroll
            data_int slice;
            // Bit-to-bit transaction
            slice.range(31, 0) = temp.range(32 * (jj + 1) - 1, 32 * jj);
            out[j * 16 + jj] = slice;
        }
    }
}

void load_all_matrices(
    data_t values_A[N*M], int column_indices_A[N*M], int row_ptr_A[N+1],
    data_t values_B[M*K], int row_indices_B[M*K], int col_ptr_B[M+1],

    data_t       buf_A_vals[N*M],
    data_int     buf_A_cols[N*M],
    data_int     buf_A_ptr[N+1],
    data_t       buf_B_vals[M*K],
    data_int     buf_B_rows[M*K],
    data_int     buf_B_ptr[M+1]
) {
#pragma HLS dataflow
    // --- Bulk transfer from DRAM to on-chip buffers using wide bus transactions ---
    load_matrix_data16(values_A, num_A_words, buf_A_vals);
    load_matrix_data16(values_B, num_B_words, buf_B_vals);
    load_matrix_int32(column_indices_A, num_A_int_words, buf_A_cols);
    load_matrix_int32(row_indices_B, num_B_int_words, buf_B_rows);
    load_matrix_int32(row_ptr_A, num_A_ptr_words, buf_A_ptr);
    load_matrix_int32(col_ptr_B, num_B_ptr_words, buf_B_ptr);
}

///////////////////////////
// Main HLS function
///////////////////////////
void sparse_matrix_multiply_HLS(
    data_t values_A[N*M], int column_indices_A[N*M], int row_ptr_A[N+1],
    data_t values_B[M*K], int row_indices_B[M*K], int col_ptr_B[M+1],
    data_t C[N][K])
{
    // --- Define wide bus AXI interfaces ---
#pragma HLS interface m_axi port=values_A         offset=slave bundle=mem10 max_widen_bitwidth=512 
#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem11 max_widen_bitwidth=512 
#pragma HLS interface m_axi port=row_ptr_A        offset=slave bundle=mem12 max_widen_bitwidth=512 

#pragma HLS interface m_axi port=values_B         offset=slave bundle=mem20 max_widen_bitwidth=512  
#pragma HLS interface m_axi port=row_indices_B    offset=slave bundle=mem21 max_widen_bitwidth=512  
#pragma HLS interface m_axi port=col_ptr_B        offset=slave bundle=mem22 max_widen_bitwidth=512  

#pragma HLS interface m_axi port=C                offset=slave bundle=mem3 max_widen_bitwidth=512 
#pragma HLS interface s_axilite port=return

    // --- On-chip buffers in BRAM ---
    static data_t       buf_A_vals[N*M];
    static data_int     buf_A_cols[N*M];
    static data_int     buf_A_ptr[N+1];
    static data_t       buf_B_vals[M*K];
    static data_int     buf_B_rows[M*K];
    static data_int     buf_B_ptr[M+1];

#pragma HLS array_partition variable=buf_A_ptr complete
#pragma HLS array_partition variable=buf_B_ptr complete
#pragma HLS array_partition variable=buf_A_vals cyclic factor=32
#pragma HLS array_partition variable=buf_B_vals cyclic factor=32
#pragma HLS array_partition variable=buf_A_cols cyclic factor=16
#pragma HLS array_partition variable=buf_B_rows cyclic factor=16

    load_all_matrices(
        values_A, column_indices_A, row_ptr_A,
        values_B, row_indices_B, col_ptr_B,
    
        buf_A_vals,
        buf_A_cols,
        buf_A_ptr,
        buf_B_vals,
        buf_B_rows,
        buf_B_ptr
    );
      
    // Handle the last element of ptr
    buf_A_ptr[N] = row_ptr_A[N];
    buf_B_ptr[M] = col_ptr_B[M];

// debug
    // Print loaded CSR A (first 5 rows)
    printf("Kernel: Buffered CSR A (first 5 rows)\n");
    for(int i = 0; i < 5; i++){
        printf("row %2d: ", i);
        for(int idx = buf_A_ptr[i]; idx < buf_A_ptr[i+1]; idx++){
            printf("(%2d,%7.4f) ", buf_A_cols[idx], buf_A_vals[idx].to_float());
        }
        printf("\n");
    }

    // Print loaded CSC B (first 5 cols)
    printf("\nKernel: Buffered CSC B (first 5 cols)\n");
    for(int j = 0; j < 5; j++){
        printf("col %2d: ", j);
        for(int idx = buf_B_ptr[j]; idx < buf_B_ptr[j+1]; idx++){
            printf("(%2d,%7.4f) ", buf_B_rows[idx], buf_B_vals[idx].to_float());
        }
        printf("\n");
    }
    printf("\n");

// --- Compute the result matrix C ---
    Row_Loop:
    for (int i = 0; i < N; i++) {
#pragma HLS pipeline II=1
        data_t C_row[K];
#pragma HLS array_partition variable=C_row complete

    // Initialize the accumulation buffer
    Init_Row:
        for (int j = 0; j < K; j++) {
#pragma HLS unroll
            C_row[j] = 0;
        }

    A_Loop:
        for (int idxA = buf_A_ptr[i]; idxA < buf_A_ptr[i+1]; idxA++) {
#pragma HLS pipeline II=1
            int k = buf_A_cols[idxA];
            data_t a_val = buf_A_vals[idxA];

        B_Loop:
            for (int idxB = buf_B_ptr[k]; idxB < buf_B_ptr[k+1]; idxB++) {
#pragma HLS pipeline II=1
                int j = buf_B_rows[idxB];
                data_t b_val = buf_B_vals[idxB];
                C_row[j] += a_val * b_val;
            }
        }

        ap_uint<512>* C_wide = reinterpret_cast<ap_uint<512>*>(C[i]);        

    Write_Row:
        for(int w = 0; w < num_C_row_words; w++) {
#pragma HLS pipeline II=1       
            ap_uint<512> wide_out;
            for (int i = 0; i < 32; i++) {
#pragma HLS unroll
                    // Bit‑cast data_t → 16‑bit unsigned without changing bits
                    ap_uint<16> bits = *reinterpret_cast<ap_uint<16>*>(&C_row[w*32 + i]);
                    wide_out.range((i+1)*16 - 1, i*16) = bits;
                }

            C_wide[w] = wide_out;
        }
        
    }
}
