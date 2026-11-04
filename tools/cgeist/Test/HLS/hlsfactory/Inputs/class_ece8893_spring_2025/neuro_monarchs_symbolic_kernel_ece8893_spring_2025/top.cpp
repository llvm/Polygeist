#include "dcl.h"

struct MemPacket {
    float data[MS];
};

struct AccPacket {
    float data[UF];
};

// Reads from memory
void read_mem_HLS(mem_t A_mem[BS][K][L/MS], mem_t B_mem[BS][K][L/MS],
                  hls::stream<MemPacket> A_buffers[], hls::stream<MemPacket> B_buffers[])
{
    MemPacket A_packet;
    #pragma HLS array_partition variable=A_packet.data complete
    MemPacket B_packet;
    #pragma HLS array_partition variable=B_packet.data complete
    
    int id = 0;
    mem_t mem_data;

    for (int b = 0; b < BS; b++) {
        for (int k = 0; k < K; k++) {
            if (id == PE) {
                id = 0;
            }
            // Read a row of A and write into buffer
            for (int l = 0; l < L / MS; l++) {
            #pragma HLS pipeline
                mem_data = A_mem[b][k][l];
                for (int i = 0; i < MS; i++) {
                    ap_uint<32> tmp = mem_data.range(31 + (i * 32), 0 + (i * 32));
                    memcpy(&A_packet.data[i], &tmp, sizeof(float));
                }
                A_buffers[id].write(A_packet);
            }
            // Read a row of B and write into buffer
            for (int l = 0; l < L / MS; l++) {
            #pragma HLS pipeline
                mem_data = B_mem[b][k][l];
                for (int i = 0; i < MS; i++) {
                    ap_uint<32> tmp = mem_data.range(31 + (i * 32), 0 + (i * 32));
                    memcpy(&B_packet.data[i], &tmp, sizeof(float));
                }
                B_buffers[id].write(B_packet);
            }
            id++;
        }
    }
}

// Organizes the local row caches and performs multiplication
void org_mult_HLS(hls::stream<MemPacket> &A_buffer, hls::stream<MemPacket> &B_buffer, hls::stream<AccPacket> &acc_buffer)
{
    float A_row_padded[L];
    #pragma HLS array_partition variable=A_row_padded type=cyclic factor=8 dim=1

    float B_row_fliproll[L];
    #pragma HLS array_partition variable=B_row_fliproll type=cyclic factor=8 dim=1

    MemPacket A_packet;
    #pragma HLS array_partition variable=A_packet.data complete
    MemPacket B_packet;
    #pragma HLS array_partition variable=B_packet.data complete
    AccPacket acc_packet;
    #pragma HLS array_partition variable=acc_packet.data complete

    for (int row = 0; row < RPPE; row++) {

        // Read and pad row of A into local
        for (int i = 0; i < L; i += MS) {
        #pragma HLS pipeline
            A_packet = A_buffer.read();
            for (int j = 0; j < MS; ++j) {
                A_row_padded[i + j] = A_packet.data[j];
            }
        }

        // Read and flip + roll B into local
        for (int i = 0; i < L; i += MS) {
        #pragma HLS pipeline
            B_packet = B_buffer.read();
            for (int j = 0; j < MS; ++j) {
                int index = i + j;
                int flipped_index = L - 1 - index;
                int rolled_index = (flipped_index + 1 == L) ? 0 : flipped_index + 1;
                B_row_fliproll[rolled_index] = B_packet.data[j];
            }
        }

        // Perform multiplications
        for (int i = 0; i < L; i++) {
            float sum = 0.0;
            for (int j = 0; j < L; j += UF) {
            #pragma HLS pipeline
                for (int jj = 0; jj < UF; jj++) {
                    int index = (i + j + jj) % L;
                    acc_packet.data[jj] = A_row_padded[index] * B_row_fliproll[j + jj];
                }
                acc_buffer.write(acc_packet);
            }
        }
    }
}

// Initializes the multipliers
void init_mults_HLS(hls::stream<MemPacket> A_buffers[], hls::stream<MemPacket> B_buffers[], hls::stream<AccPacket> acc_buffers[])
{   
    for (int id = 0; id < PE; id++) {
    #pragma HLS unroll
        org_mult_HLS(A_buffers[id], B_buffers[id], acc_buffers[id]);
    }
}

// Performs binary tree-style accumulation over multiplied elements
void acc_HLS(hls::stream<AccPacket> &acc_buffer, hls::stream<MemPacket> &C_buffer)
{
    AccPacket acc_packet_in;
    #pragma HLS array_partition variable=acc_packet_in.data complete
    MemPacket C_packet;
    #pragma HLS array_partition variable=C_packet.data complete

    // Local accumulation buffers to mitigate floating point accumulation II violations
    float locals[4];
    #pragma HLS array_partition variable=locals complete

    for (int i = 0; i < RPPE * L / MS; i++) {
        for (int ii = 0; ii < MS; ii++) {
            for (int l = 0; l < 4; l++) {
            #pragma HLS unroll
                locals[l] = 0.0;
            }

            // If changing UF, this loop needs to be modified by hand to ensure appropriate pair additions
            for (int j = 0; j < L / UF; j++) {
            #pragma HLS pipeline
                acc_buffer.read(acc_packet_in);

                // Stage 1: Add pairs
                float stage1[UF / 2];
                #pragma HLS array_partition variable=stage1 complete
                for (int a = 0; a < UF / 2; a++) {
                #pragma HLS unroll
                    stage1[a] = acc_packet_in.data[2*a] + acc_packet_in.data[2*a + 1];
                }

                // Stage 2: Add pairs
                float stage2[UF / 4];
                #pragma HLS array_partition variable=stage2 complete
                for (int a = 0; a < UF / 4; a++) {
                #pragma HLS unroll
                    stage2[a] = stage1[2*a] + stage1[2*a + 1];
                }

                // Stage 3: Add Final Pairs
                float stage3 = stage2[0] + stage2[1];

                // Stage 4: Accumulate in local cache
                int locals_index = j % 4;
                float to_add = locals[locals_index];
                float new_sum = stage3 + to_add;
                locals[locals_index] = new_sum;
            }

            float final1 = locals[0] + locals[1];
            float final2 = locals[2] + locals[3];
            C_packet.data[ii] = final1 + final2;
        }
        C_buffer.write(C_packet);
    }
}

// Initializes the full accumulators
void init_accs_HLS(hls::stream<AccPacket> acc_buffers[], hls::stream<MemPacket> C_buffers[])
{
    for (int id = 0; id < PE; id++) {
    #pragma HLS unroll
        acc_HLS(acc_buffers[id], C_buffers[id]);
    }
}

// Writes finished rows into memory
void write_mem_HLS(hls::stream<MemPacket> C_buffers[], mem_t C_mem[BS][K][L/MS])
{   
    MemPacket C_packet;
    #pragma HLS array_partition variable=C_packet.data complete

    int id = 0;
    mem_t mem_data;

    for (int b = 0; b < BS; b++) {
        for (int k = 0; k < K; k++) {
            if (id == PE) {
                id = 0;
            }

            // Write a row of C into the output
            for (int l = 0; l < L / MS; l++) {
            #pragma HLS pipeline
                C_packet = C_buffers[id].read();
                for (int i = 0; i < MS; i++) {
                    ap_uint<32> tmp;
                    memcpy(&tmp, &C_packet.data[i], sizeof(float));
                    mem_data.range(31 + (i * 32), 0 + (i * 32)) = tmp;
                }
                C_mem[b][k][l] = mem_data;
            }

            id++;
        }
    }
}

// Main dataflow
void dataflow_HLS(mem_t A_mem[BS][K][L/MS], mem_t B_mem[BS][K][L/MS], mem_t C_mem[BS][K][L/MS])
{
#pragma HLS dataflow

    hls::stream<MemPacket, 16> A_buffers[PE];
    #pragma HLS array_partition variable=A_buffers complete
    hls::stream<MemPacket, 16> B_buffers[PE];
    #pragma HLS array_partition variable=B_buffers complete

    hls::stream<AccPacket> acc_buffers[PE];
    #pragma HLS array_partition variable=acc_buffers complete

    hls::stream<MemPacket, 16> C_buffers[PE];
    #pragma HLS array_partition variable=C_buffers complete

    read_mem_HLS(A_mem, B_mem, A_buffers, B_buffers);

    init_mults_HLS(A_buffers, B_buffers, acc_buffers);

    init_accs_HLS(acc_buffers, C_buffers);

    write_mem_HLS(C_buffers, C_mem);
}

// Top module
void symbolic_circular_conv_HLS(mem_t A_mem[BS][K][L/MS], mem_t B_mem[BS][K][L/MS], mem_t C_mem[BS][K][L/MS])
{
#pragma HLS interface m_axi port = A_mem offset = slave bundle = mem1
#pragma HLS interface m_axi port = B_mem offset = slave bundle = mem1
#pragma HLS interface m_axi port = C_mem offset = slave bundle = mem2

#pragma HLS interface s_axilite port = return

    dataflow_HLS(A_mem, B_mem, C_mem);

}
