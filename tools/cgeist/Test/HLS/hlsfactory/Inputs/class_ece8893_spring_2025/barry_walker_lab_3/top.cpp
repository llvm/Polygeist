#include "dcl.h"

#define CORES 16
#define BATCH_SIZE 8

struct Entry {
    data_t val;
    ap_uint<16> index;
};
struct Batch {
    int i;
    data_t val_A;
    data_t val_B[BATCH_SIZE];
    int j[BATCH_SIZE];
};
struct WriteBatch {
    int i;
    data_t val[BATCH_SIZE];
};

// Function responsible for reading values from memory and storing them locally
void read_mem_HLS(mem_t values_A[MPDA], mem_t column_indices_A[MPDA], ap_uint<16> row_ptr_A[N + 1],
                  mem_t values_B[MPDB], mem_t row_indices_B[MPDB], ap_uint<16> col_ptr_B[M + 1],
                  Entry A_local[N * M], ap_uint<16> row_ptr_A_local[N + 1],
                  Entry B_local[M * K], ap_uint<16> col_ptr_B_local[M + 1])
{   
    // Loading in the row and col pointer arrays
    for (int i = 0; i < N + 1; i++) {
    #pragma HLS pipeline
        row_ptr_A_local[i] = row_ptr_A[i];
    }
    for (int i = 0; i < M + 1; i++) {
    #pragma HLS pipeline
        col_ptr_B_local[i] = col_ptr_B[i];
    }

    //Reading in the values and column indicies of A via wide bus transactions, merging them into entry structs.
    bool done = false;
    for (int i = 0; i < MPDA; i++) {
    #pragma HLS pipeline
        mem_t val_mem = values_A[i];
        mem_t ind_mem = column_indices_A[i];
        for (int j = 0; j < DPM; j++) {
            int index = i * DPM + j;
            if (index < row_ptr_A_local[N]) {
                Entry entry;
                entry.val.range(15, 0) = val_mem.range(15 + (j * 16), 0 + (j * 16));
                entry.index.range(15, 0) = ind_mem.range(15 + (j * 16), 0 + (j * 16));
                A_local[index] = entry;
            } else {
                done = true;
            }
        }
        // If finished with all non-zero data, do not fetch anymore
        if (done) {
            break;
        }
    }

    //Reading in the values and row indicies of B via wide bus transactions, merging them into entry structs.
    done = false;
    for (int i = 0; i < MPDB; i++) {
    #pragma HLS pipeline
        mem_t val_mem = values_B[i];
        mem_t ind_mem = row_indices_B[i];
        for (int j = 0; j < DPM; j++) {
            int index = i * DPM + j;
            if (index < col_ptr_B_local[M]) {
                Entry entry;
                entry.val.range(15, 0) = val_mem.range(15 + (j * 16), 0 + (j * 16));
                entry.index.range(15, 0) = ind_mem.range(15 + (j * 16), 0 + (j * 16));
                B_local[index] = entry;
            } else {
                done = true;
            }
        }
        // If finished with all non-zero data, do not fetch anymore
        if (done) {
            break;
        }
    }
}

// Responsible for iterating over the arrays and scheduling computations
void scheduler_HLS(Entry A_local[N * M], ap_uint<16> row_ptr_A_local[N + 1],
                   Entry B_local[M * K], ap_uint<16> col_ptr_B_local[M + 1],
                   hls::stream<Batch> mult_buffers[]) {
    Batch batch;
    int id = 0;
    // For each row...
    for (int i = 0; i < N; i++) {
        // Reset for cyclically scheduling rows to cores
        if (id == CORES) {
            id = 0;
        }

        // Iterating over the values of the row
        for (int idx_A = row_ptr_A_local[i]; idx_A < row_ptr_A_local[i + 1]; idx_A++) {
            Entry A_entry = A_local[idx_A];
            data_t value_A = A_entry.val;
            int col = A_entry.index;

            int col_end = col_ptr_B_local[col + 1];
            
            // Iterating over the values of the column corresponding to the row value
            for (int idx_B = col_ptr_B_local[col]; idx_B < col_end; idx_B += BATCH_SIZE) {
            #pragma HLS pipeline
                batch.i = i;
                batch.val_A = value_A;
                for (int be = 0; be < BATCH_SIZE; be++) {
                    int index = idx_B + be;
                    if (index < col_end) {
                        Entry b_entry = B_local[index];
                        batch.val_B[be] = b_entry.val;
                        batch.j[be] = b_entry.index;
                    } else {
                        batch.j[be] = -1;   // Setting the value to -1 so the subsequent modules know this is unused
                    }
                }
                mult_buffers[id].write(batch);
            }
        }

        id++;
    }

    // Writing a terminating packet so the multipliers know when to stop.
    batch.i = -1;
    for (int id = 0; id < CORES; id++) {
        mult_buffers[id].write(batch);
    }
}

// A multiply process element
void multiply_PE_HLS(hls::stream<Batch> &mult_buffer, hls::stream<Batch> &accum_buffer) {
    Batch batch = mult_buffer.read();

    // Wait until a termination batch is received
    while (batch.i != -1) {
        for (int b = 0; b < BATCH_SIZE; b++) {
        #pragma HLS unroll
            if (batch.j[b] == -1) {
                break;
            }
            batch.val_B[b] = batch.val_A * batch.val_B[b];
        }
        accum_buffer.write(batch);
        batch = mult_buffer.read();
    }

    accum_buffer.write(batch);
}

// Initializes the multipliers
void multipliers_HLS(hls::stream<Batch> mult_buffers[], hls::stream<Batch> accum_buffers[])
{
    for (int id = 0; id < CORES; id++) {
    #pragma HLS unroll
        multiply_PE_HLS(mult_buffers[id], accum_buffers[id]);
    }
}

// Accumulator process element
void accumulator_PE_HLS(hls::stream<Batch> &accum_buffer, hls::stream<WriteBatch> &write_buffer){

    // Row length local buffer
    data_t local_buffer[K];
    #pragma HLS array_partition variable=local_buffer factor=BATCH_SIZE cyclic

    // Initialize the buffer
    for (int k = 0; k < K; k++) {
    #pragma HLS unroll factor=BATCH_SIZE
        local_buffer[k] = 0;
    }

    Batch batch = accum_buffer.read();
    int row_index = batch.i;
    WriteBatch send_batch;
    
    // Wait until a termination batch is received
    while (row_index != -1) {
        // While operating on the same row
        while (batch.i == row_index) {
            for (int b = 0; b < BATCH_SIZE; b++) {
            #pragma HLS pipeline
                if (batch.j[b] == -1) {
                    break;
                }
                local_buffer[batch.j[b]] += batch.val_B[b];
            }
            batch = accum_buffer.read();
        }

        // When the first batch of the next row is received, send the current row to the write module
        send_batch.i = row_index;
        for (int j = 0; j < K; j += BATCH_SIZE) {
        #pragma HLS pipeline
            for (int b = 0; b < BATCH_SIZE; b++) {
                send_batch.val[b] = local_buffer[j + b];
            }
            write_buffer.write(send_batch);
        }

        // Initialize the buffer for the next row
        row_index = batch.i;
        for (int k = 0; k < K; k++) {
        #pragma HLS unroll factor=BATCH_SIZE
            local_buffer[k] = 0;
        }
    }

    send_batch.i = -1;
    write_buffer.write(send_batch);
}

// Initializes the accumulators
void accumulators_HLS(hls::stream<Batch> accum_buffers[], hls::stream<WriteBatch> write_buffers[]) {
    for (int id = 0; id < CORES; id++) {
    #pragma HLS unroll
        accumulator_PE_HLS(accum_buffers[id], write_buffers[id]);
    }
}

// Module writes the rows into memory
void mem_write_HLS(hls::stream<WriteBatch> write_buffers[], mem_t C[N][K/DPM]) {
    // Initialize structures to keep track of terminated cores
    bool terminated[CORES];
    int term_count = 0;
    for (int id = 0; id < CORES; id++) {
        terminated[id] = false;
    }

    // Local row cache for reading write batches
    data_t local_cache[K];
    #pragma HLS array_partition variable=local_cache factor=BATCH_SIZE cyclic
    int row_index = 0;

    // While all cores are not terminated
    while (term_count < CORES) {
        // Iterate over all the cores
        for (int id = 0; id < CORES; id++) {
            // If the core is still active
            if (!terminated[id]) {
                // Read from FIFO
                for (int j = 0; j < K; j += BATCH_SIZE) {
                #pragma HLS pipeline
                    WriteBatch batch = write_buffers[id].read();
                    row_index = batch.i;
                    if (row_index == -1) {
                        break;
                    }
                    for (int b = 0; b < BATCH_SIZE; b++) {
                        local_cache[j + b] = batch.val[b];
                    }
                }

                // If terminate batch read, mark the core as terminated
                if (row_index == -1) {
                    term_count++;
                    terminated[id] = true;
                } else {
                    // Write the row to memory
                    for (int k = 0; k < K/DPM; k++) {
                    #pragma HLS pipeline
                        mem_t data = 0;
                        for (int kk = 0; kk < 32; kk++) {
                            data.range(15 + (kk * 16), 0 + (kk * 16)) = local_cache[k * 32 + kk].range(15, 0);
                        }
                        C[row_index][k] = data;
                    }
                }
            }
        } 
    }
}

// Main dataflow for the module
void dataflow_HLS(Entry A_local[N * M], ap_uint<16> row_ptr_A_local[N + 1],
                  Entry B_local[M * K], ap_uint<16> col_ptr_B_local[M + 1],
                  mem_t C[N][K/DPM])
{
#pragma HLS dataflow

    // Buffers for the multiply modules
    hls::stream<Batch, 2> mult_buffers[CORES];
    #pragma HLS array_partition variable=mult_buffers complete

    // Buffers for the accumulation modules
    hls::stream<Batch, 256> accum_buffers[CORES];
    #pragma HLS array_partition variable=accum_buffers complete

    // Buffers for writes to memory
    hls::stream<WriteBatch, 10> write_buffers[CORES];
    #pragma HLS array_partition variable=write_buffers complete

    scheduler_HLS(A_local, row_ptr_A_local, B_local, col_ptr_B_local, mult_buffers);

    multipliers_HLS(mult_buffers, accum_buffers);

    accumulators_HLS(accum_buffers, write_buffers);

    mem_write_HLS(write_buffers, C);
}

// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(mem_t values_A[MPDA], mem_t column_indices_A[MPDA], ap_uint<16> row_ptr_A[N + 1],
                                mem_t values_B[MPDB], mem_t row_indices_B[MPDB], ap_uint<16> col_ptr_B[M + 1],
                                mem_t C[N][K/DPM])
{
#pragma HLS interface m_axi port = values_A offset = slave bundle = mem1
#pragma HLS interface m_axi port = column_indices_A offset = slave bundle = mem1
#pragma HLS interface m_axi port = row_ptr_A offset = slave bundle = mem1

#pragma HLS interface m_axi port = values_B offset = slave bundle = mem2
#pragma HLS interface m_axi port = row_indices_B offset = slave bundle = mem2
#pragma HLS interface m_axi port = col_ptr_B offset = slave bundle = mem2

#pragma HLS interface m_axi port = C offset = slave bundle = mem3

#pragma HLS interface s_axilite port = return

    // Local cache for A
    Entry A_local[N * M];

    // Local cache for A pointers
    ap_uint<16> row_ptr_A_local[N + 1];
    #pragma HLS array_partition variable=row_ptr_A_local factor=2 cyclic

    // Local cache for B
    Entry B_local[M * K];
    #pragma HLS array_partition variable=B_local factor=BATCH_SIZE*2 cyclic
    
    // Local cache for B pointers
    ap_uint<16> col_ptr_B_local[M + 1];
    #pragma HLS array_partition variable=col_ptr_B_local factor=2 cyclic

    read_mem_HLS(values_A, column_indices_A, row_ptr_A, values_B, row_indices_B, col_ptr_B, A_local, row_ptr_A_local, B_local, col_ptr_B_local);

    dataflow_HLS(A_local, row_ptr_A_local, B_local, col_ptr_B_local, C);

}
