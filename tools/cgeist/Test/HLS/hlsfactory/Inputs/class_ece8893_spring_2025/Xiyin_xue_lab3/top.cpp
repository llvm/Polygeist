#include "dcl.h"
typedef ap_uint<512> wide_bus_t;
#define ELEMS_PER_BUS (512 / 8 / sizeof(data_t))
#define ELEMS_PER_BUS_INT (512 / 8 / sizeof(int))


//-----------------Load Data arrays related to Sparse Matrix A-----------------
void load_value_A(data_t va_in[N * M], data_t va_buff[N * M]) {
    // Loop label for the pipeline process
    LOOP_Value_A:
    for (int i = 0; i < N * M; i += ELEMS_PER_BUS) { // Loop over the input array in chunks of ELEMS_PER_BUS
#pragma HLS PIPELINE II=1            // HLS pragma to pipeline the loop with an initiation interval of 1
        wide_bus_t temp = *reinterpret_cast<wide_bus_t*>(&va_in[i]); // Reinterpret the chunk of Q as a wide bus type
        for (int ii = 0; ii < ELEMS_PER_BUS; ii++) { // Loop over each element in the chunk
#pragma HLS UNROLL                // HLS pragma to unroll the loop for parallel processing
            if ((i + ii) < N * M) {              // Check if the current index is within bounds
                ap_uint<16> bits = temp.range(16 * (ii + 1) - 1, 16 * ii); // Extract 16 bits from the wide bus
                va_buff[i + ii] = *reinterpret_cast<data_t*>(&bits); // Reinterpret the bits as a fixed-point number and store in the buffer
            }
        }
    }
}


void load_ptr_A(int pa_in[N +1], int pa_buff[N +1]) {
    // Loop label for the pipeline process
LOOP_PTR_A:
    for (int i = 0; i < N +1; i += ELEMS_PER_BUS_INT) { // Loop over the input array in chunks of ELEMS_PER_BUS
#pragma HLS PIPELINE II=1            // HLS pragma to pipeline the loop with an initiation interval of 1
        wide_bus_t temp = *reinterpret_cast<wide_bus_t*>(&pa_in[i]); // Reinterpret the chunk of Q as a wide bus type
        for (int ii = 0; ii < ELEMS_PER_BUS_INT; ii++) { // Loop over each element in the chunk
#pragma HLS UNROLL                // HLS pragma to unroll the loop for parallel processing
            if ((i + ii) < N +1) {              // Check if the current index is within bounds
                ap_uint<32> bits = temp.range(32 * (ii + 1) - 1, 32 * ii); // Extract 16 bits from the wide bus
                pa_buff[i + ii] = *reinterpret_cast<int*>(&bits); // Reinterpret the bits as a fixed-point number and store in the buffer
            }
        }
    }
}


void load_Cidx_A(int ca_in[N * M], int ca_buff[N * M]) {
    // Loop label for the pipeline process
LOOP_Column_A:
    for (int i = 0; i < N * M; i += ELEMS_PER_BUS_INT) { // Loop over the input array in chunks of ELEMS_PER_BUS
#pragma HLS PIPELINE II=1            // HLS pragma to pipeline the loop with an initiation interval of 1
        wide_bus_t temp = *reinterpret_cast<wide_bus_t*>(&ca_in[i]); // Reinterpret the chunk of Q as a wide bus type
        for (int ii = 0; ii < ELEMS_PER_BUS_INT; ii++) { // Loop over each element in the chunk
#pragma HLS UNROLL                // HLS pragma to unroll the loop for parallel processing
            if ((i + ii) < N * M) {              // Check if the current index is within bounds
                ap_uint<32> bits = temp.range(32 * (ii + 1) - 1, 32 * ii); // Extract 16 bits from the wide bus
                ca_buff[i + ii] = *reinterpret_cast<int*>(&bits); // Reinterpret the bits as a fixed-point number and store in the buffer
            }
        }
    }
}
//-----------------------------------------------------------------------------


//-----------------Load Data arrays related to Sparse Matrix B-----------------
void load_value_B(data_t vb_in[K * M], data_t vb_buff[K * M]) {
    // Loop label for the pipeline process
LOOP_Value_B:
    for (int i = 0; i < K * M; i += ELEMS_PER_BUS) { // Loop over the input array in chunks of ELEMS_PER_BUS
#pragma HLS PIPELINE II=1            // HLS pragma to pipeline the loop with an initiation interval of 1
        wide_bus_t temp = *reinterpret_cast<wide_bus_t*>(&vb_in[i]); // Reinterpret the chunk of Q as a wide bus type
        for (int ii = 0; ii < ELEMS_PER_BUS; ii++) { // Loop over each element in the chunk
#pragma HLS UNROLL                // HLS pragma to unroll the loop for parallel processing
            if ((i + ii) < K * M) {              // Check if the current index is within bounds
                ap_uint<16> bits = temp.range(16 * (ii + 1) - 1, 16 * ii); // Extract 16 bits from the wide bus
                vb_buff[i + ii] = *reinterpret_cast<data_t*>(&bits); // Reinterpret the bits as a fixed-point number and store in the buffer
            }
        }
    }
}


void load_ptr_B(int pb_in[M + 1], int pb_buff[M + 1]) {
    // Loop label for the pipeline process
LOOP_PTR_B:
    for (int i = 0; i < N + 1; i += ELEMS_PER_BUS_INT) { // Loop over the input array in chunks of ELEMS_PER_BUS
#pragma HLS PIPELINE II=1            // HLS pragma to pipeline the loop with an initiation interval of 1
        wide_bus_t temp = *reinterpret_cast<wide_bus_t*>(&pb_in[i]); // Reinterpret the chunk of Q as a wide bus type
        for (int ii = 0; ii < ELEMS_PER_BUS_INT; ii++) { // Loop over each element in the chunk
#pragma HLS UNROLL                // HLS pragma to unroll the loop for parallel processing
            if ((i + ii) < N + 1) {              // Check if the current index is within bounds
                ap_uint<32> bits = temp.range(32 * (ii + 1) - 1, 32 * ii); // Extract 16 bits from the wide bus
                pb_buff[i + ii] = *reinterpret_cast<int*>(&bits); // Reinterpret the bits as a fixed-point number and store in the buffer
            }
        }
    }
}



void load_Ridx_B(int cb_in[K * M], int cb_buff[K * M]) {
    // Loop label for the pipeline process
LOOP_ROW_B:
    for (int i = 0; i < K * M; i += ELEMS_PER_BUS_INT) { // Loop over the input array in chunks of ELEMS_PER_BUS
#pragma HLS PIPELINE II=1            // HLS pragma to pipeline the loop with an initiation interval of 1
        wide_bus_t temp = *reinterpret_cast<wide_bus_t*>(&cb_in[i]); // Reinterpret the chunk of Q as a wide bus type
        for (int ii = 0; ii < ELEMS_PER_BUS_INT; ii++) { // Loop over each element in the chunk
#pragma HLS UNROLL                // HLS pragma to unroll the loop for parallel processing
            if ((i + ii) < K * M) {              // Check if the current index is within bounds
                ap_uint<32> bits = temp.range(32 * (ii + 1) - 1, 32 * ii); // Extract 16 bits from the wide bus
                cb_buff[i + ii] = *reinterpret_cast<int*>(&bits); // Reinterpret the bits as a fixed-point number and store in the buffer
            }
        }
    }
}
//-----------------------------------------------------------------------------

//-----------------Store Data To DRAM---------------------------

void store_c(
    data_t c_buff[N][K],  // Input array representing the line buffer
    data_t c[N][K]      // Output array representing the wider bus storage
)
{
    // Loop to process chunks of the output line buffer
LOOP_STORE_C:
	for (int i = 0; i < N; i++){
    for (int j = 0; j < K; j += ELEMS_PER_BUS) {
#pragma HLS PIPELINE // Enable pipelining to improve throughput
        wide_bus_t temp; // Temporary variable to hold the wide bus data
        // Loop to process each element within the current chunk
        for (int jj = 0; jj < ELEMS_PER_BUS; jj++) {
#pragma HLS UNROLL // Unroll the loop to process elements in parallel
            // Reinterpret the fixed_t data as a 16-bit unsigned integer
            ap_uint<16> bits = *reinterpret_cast<ap_uint<16>*>(&c_buff[i][j + jj]);
            // Assign the 16-bit data to the appropriate range in the temporary wide bus
            temp.range(16 * (jj + 1) - 1, 16 * jj) = bits;
        }
        // Store the temporary wide bus data into the output array
        *reinterpret_cast<wide_bus_t*>(&c[i][j]) = temp;
    }
	}
}

//---------------------------------------------------------------

// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], 
                             data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t C[N][K]) 
{
#pragma HLS interface m_axi port=values_A offset=slave bundle=mem1 max_widen_bitwidth=512
#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1 max_widen_bitwidth=512
#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1 max_widen_bitwidth=512

#pragma HLS interface m_axi port=values_B offset=slave bundle=mem2 max_widen_bitwidth=512
#pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2 max_widen_bitwidth=512
#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2 max_widen_bitwidth=512

#pragma HLS interface m_axi port=C offset=slave bundle=mem3 max_widen_bitwidth=512

#pragma HLS interface s_axilite port=return

    data_t va_buff[N * M], vb_buff[K * M];
    int ca_buff[N * M], rb_buff[K * M];
    int rp_a[N + 1], cp_b[M + 1];
    data_t c_buff[N][K] = {0};



#pragma HLS ARRAY_PARTITION variable=va_buff factor=128 cyclic
#pragma HLS ARRAY_PARTITION variable=ca_buff factor=128 cyclic
#pragma HLS ARRAY_PARTITION variable=vb_buff factor=128 cyclic
#pragma HLS ARRAY_PARTITION variable=rb_buff factor=128 cyclic
#pragma HLS ARRAY_PARTITION variable=rp_a factor=32 cyclic
#pragma HLS ARRAY_PARTITION variable=cp_b factor=32 cyclic

#pragma HLS ARRAY_PARTITION variable=c_buff dim=2 factor=128 cyclic

//#pragma HLS DATAFLOW
    load_value_A(values_A, va_buff);
    load_Cidx_A(column_indices_A, ca_buff);
    load_value_B(values_B, vb_buff);
    load_Ridx_B(row_indices_B, rb_buff);
    load_ptr_A(row_ptr_A, rp_a);
    load_ptr_B(col_ptr_B, cp_b);

    // Perform Sparse x Sparse Multiplication
Loop_Main_00:
//#pragma HLS UNROLL off = true
//#pragma HLS UNROLL off = true 
    for (int i = 0; i < N; i++) {
//#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL off = true
#pragma HLS LOOP_FLATTEN off = true
LOOP_Main_01:
        for (int idx_A = rp_a[i]; idx_A < rp_a[i + 1]; idx_A++) {
            int k = ca_buff[idx_A]; // Column index of A
            data_t value_A = va_buff[idx_A];
            // Iterate over columns of B corresponding to row k
//#pragma HLS PIPELINE
#pragma HLS UNROLL off = true
#pragma HLS LOOP_FLATTEN off = true
LOOP_Main_02:
            for (int idx_B = cp_b[k]; idx_B < cp_b[k + 1]; idx_B++) {
           //#pragma HLS UNROLL factor = 2
                int j = rb_buff[idx_B]; // Column index of B
                data_t value_B = vb_buff[idx_B];

                // Accumulate the product into C[i][j]
                c_buff[i][j] += value_A * value_B;
            }
        }
    }

        store_c(c_buff, C);
    


}
