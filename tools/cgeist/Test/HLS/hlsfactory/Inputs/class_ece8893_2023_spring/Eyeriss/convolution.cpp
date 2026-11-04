#include "convolution.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

// Structs for each PE to maintain some local data
struct PE
{
    int local_cycle_count;    // Keeps count of local active cycles for each PE
    int psum_count;           // To help in local data reuse - shift register functionality
    int psum_buffer_internal; // For MAC
};

// First dimension is for identifying which PE's local buffer it is, second dimension is the local buffer size
int PE_IFM_Buffer[PE_LENGTH * PE_BREADTH][BUFFER_SIZE];
int PE_Wt_Buffer[PE_LENGTH * PE_BREADTH][BUFFER_SIZE];

PE pes[PE_LENGTH][PE_BREADTH]; // Total number of PEs in this systolic array

int global_cycles;
int count_test = 0;

/*
The below function supports resetting the systolic array,
clearing data for logical PE folding (increased data size)
i.e ROW size > systolic array size
*/

void reset()
{
    for (int i = 0; i < PE_LENGTH; i++)
        for (int j = 0; j < PE_LENGTH; j++)
        {
            pes[i][j].local_cycle_count = 0;
            pes[i][j].psum_count = 0;
            pes[i][j].psum_buffer_internal = 0;
        }
    global_cycles = 0;
    count_test = 0;
}

/* The below function reads data the local buffers from DRAM
assigning rows of inputs/weights to specific PEs for row reuse
*/

void read_data_DRAM(int DRAM_ip_data[IF_LENGTH][IF_WIDTH], int DRAM_Wt_data[WT_LENGTH][WT_WIDTH], int &horizontal_ip_0_0, int &horizontal_ip_1_0,
                    int &horizontal_ip_2_0, int &diagonal_ip_0_0, int &diagonal_ip_1_0, int &diagonal_ip_2_0,
                    int &diagonal_ip_2_1, int &diagonal_ip_2_2)
{
#pragma HLS INLINE OFF
    global_cycles++;

    if (global_cycles <= BUFFER_SIZE + PE_BREADTH - 1) // to account for initial cycles where data movement is continuous
    {

        if (global_cycles <= BUFFER_SIZE)
        {
            // 0-0 PE

            diagonal_ip_0_0 = DRAM_ip_data[0][global_cycles - 1];
            horizontal_ip_0_0 = DRAM_Wt_data[0][global_cycles - 1];

            // 1-0 PE

            diagonal_ip_1_0 = DRAM_ip_data[1][global_cycles - 1];
            horizontal_ip_1_0 = DRAM_Wt_data[1][global_cycles - 1];

            // 2-0 PE

            diagonal_ip_2_0 = DRAM_ip_data[2][global_cycles - 1];
            horizontal_ip_2_0 = DRAM_Wt_data[2][global_cycles - 1];
        }

        // 2-1 PE
        if (1 < global_cycles && global_cycles <= BUFFER_SIZE + 1)
        {
            diagonal_ip_2_1 = DRAM_ip_data[3][global_cycles - 2];
        }

        // 2-2 PE
        if (2 < global_cycles && global_cycles <= BUFFER_SIZE + 2)
        {
            diagonal_ip_2_2 = DRAM_ip_data[4][global_cycles - 3];
        }
    }

    if (global_cycles > 4) // further cycles - cyclic data movement
    {
        if ((global_cycles + 1) % BUFFER_SIZE == 0)
        {

            diagonal_ip_0_0 = DRAM_ip_data[0][PE_LENGTH - 2 + ((global_cycles + 1) / 3)];
            diagonal_ip_1_0 = DRAM_ip_data[1][PE_LENGTH - 2 + ((global_cycles + 1) / 3)];
            diagonal_ip_2_0 = DRAM_ip_data[2][PE_LENGTH - 2 + ((global_cycles + 1) / 3)];
        }
        else if (global_cycles % BUFFER_SIZE == 0)
        {
            diagonal_ip_2_1 = DRAM_ip_data[3][PE_LENGTH - 2 + ((global_cycles - 1 + 1) / 3)];
        }
        else if ((global_cycles - 1) % BUFFER_SIZE == 0)
        {
            diagonal_ip_2_2 = DRAM_ip_data[4][PE_LENGTH - 2 + ((global_cycles - 2 + 1) / 3)];
        }
    }
}

// The below function for simulating the external PEs that produce summation of PSUMs

void sum_of_psums(hls::stream<int> &psum_DRAM0, hls::stream<int> &psum_DRAM1, hls::stream<int> &psum_DRAM2, hls::stream<int> &psum_buffer0, hls::stream<int> &psum_buffer1, hls::stream<int> &psum_buffer2)
{

    if (psum_buffer0.size() == SUM_BUFFER_SIZE)
    {

        int psum0 = 0;
        for (int i = 0; i < SUM_BUFFER_SIZE; i++)
        {
            psum0 += psum_buffer0.read();
        }
        psum_DRAM0.write(psum0);
    }
    if (psum_buffer1.size() == SUM_BUFFER_SIZE)
    {
        int psum1 = 0;
        for (int i = 0; i < SUM_BUFFER_SIZE; i++)
        {
            psum1 += psum_buffer1.read();
        }
        psum_DRAM1.write(psum1);
    }
    if (psum_buffer2.size() == SUM_BUFFER_SIZE)
    {
        int psum2 = 0;
        for (int i = 0; i < SUM_BUFFER_SIZE; i++)
        {
            psum2 += psum_buffer2.read();
        }
        psum_DRAM2.write(psum2);
    }
}

// Below functions writes the final data to DRAM

void write_data_DRAM(int DRAM_op_data[OP_LENGTH][OP_WIDTH], hls::stream<int> &psum_DRAM0, hls::stream<int> &psum_DRAM1, hls::stream<int> &psum_DRAM2)
{
#pragma HLS INLINE OFF

    if (psum_DRAM0.size() == 1)
    {
        DRAM_op_data[0][count_test] = psum_DRAM0.read();
    }
    if (psum_DRAM1.size() == 1)
    {
        DRAM_op_data[1][count_test] = psum_DRAM1.read();
    }
    if (psum_DRAM2.size() == 1)
    {
        DRAM_op_data[2][count_test] = psum_DRAM2.read();
        count_test++;
    }
}

// The below function simulates all the operations performed by a PE (data movement and MAC)

void processingelement(hls::stream<int> &psum_buffer0, hls::stream<int> &psum_buffer1, hls::stream<int> &psum_buffer2, int &horizontal_ip, int &horizontal_op, int &vertical_ip, int &vertical_op, int &diagonal_ip, int &diagonal_op, int i, int j, int k)
{
#pragma HLS INLINE

    if (global_cycles >= j + 1 && pes[i][j].local_cycle_count < 1 + PE_BREADTH + (OP_WIDTH * 3)) // start and stop condition for PEs
    {
        pes[i][j].local_cycle_count++;

        if (pes[i][j].psum_count == BUFFER_SIZE)
        {
            // to pop out data for incoming data - shift register functionality
            for (int i = 0; i < BUFFER_SIZE; i++)
            {
                if (i != (BUFFER_SIZE - 1))
                    PE_IFM_Buffer[k][i] = PE_IFM_Buffer[k][i + 1];
                else
                    PE_IFM_Buffer[k][i] = 0;
            }

            pes[i][j].psum_count = 0;

            // write data to PE outside systolic array for final summation
            if (i == 0)
            {

                if (j == 0)
                {
                    psum_buffer0.write(pes[i][j].psum_buffer_internal);
                }
                else if (j == 1)
                {
                    psum_buffer1.write(pes[i][j].psum_buffer_internal);
                }
                else if (j == 2)
                {
                    psum_buffer2.write(pes[i][j].psum_buffer_internal);
                }
            }
            else
            {
                vertical_op = pes[i][j].psum_buffer_internal;
            }
            pes[i][j].psum_buffer_internal = 0;
        }
        else
        {
            // vertical movement of PSUMs
            if (i == 0 && pes[i][j].local_cycle_count >= PE_BREADTH + BUFFER_SIZE) // 6 is initial latency of the design
            {
                if (j == 0)
                {
                    psum_buffer0.write(vertical_ip);
                }
                else if (j == 1)
                {
                    psum_buffer1.write(vertical_ip);
                }
                else if (j == 2)
                {
                    psum_buffer2.write(vertical_ip);
                }
            }
            else if (pes[i][j].local_cycle_count >= PE_BREADTH + BUFFER_SIZE) // 6 is initial latency of the design
            {
                vertical_op = vertical_ip;
            }
        }

        // MAC operation with shift register
        if (pes[i][j].local_cycle_count > 1)
        {
            pes[i][j].psum_buffer_internal += PE_IFM_Buffer[k][pes[i][j].psum_count] * PE_Wt_Buffer[k][pes[i][j].psum_count];
            pes[i][j].psum_count++;
        }

        // Moving data to neighbouring PEs - intial cycles to fill pipeline
        if ((pes[i][j].local_cycle_count > 1 && pes[i][j].local_cycle_count <= BUFFER_SIZE + 1))
        {

            // Diagonal
            if (i > 0 && j < (PE_LENGTH - 1)) // check for bounds
            {
                // pass input feature map diagonally to PE i-1, j+1
                diagonal_op = PE_IFM_Buffer[k][pes[i][j].local_cycle_count - 2];
            }

            // Horizontal
            if (j < (PE_LENGTH - 1)) // check for bounds
            {
                // pass weights horizontally to PE i, j+1
                horizontal_op = PE_Wt_Buffer[k][pes[i][j].local_cycle_count - 2];
            }
        }
        // Moving data to neighbouring PEs - following cycles
        else if ((pes[i][j].local_cycle_count) % BUFFER_SIZE == 0)
        {

            // Diagonal
            if (i > 0 && (j < (PE_LENGTH - 1)) && pes[i][j].local_cycle_count <= (((PE_BREADTH * PE_LENGTH) + ((IF_WIDTH - (PE_BREADTH + PE_BREADTH - 1)) * BUFFER_SIZE))))
            {
                // pass input feature map diagonally to PE i-1, j+1
                diagonal_op = PE_IFM_Buffer[k][2];
            }

            // Horizontal
            if ((j < (PE_LENGTH - 1)) && pes[i][j].local_cycle_count <= (BUFFER_SIZE + 1))
            {
                // pass weights horizontally to PE i, j+1
                horizontal_op = PE_Wt_Buffer[k][2];
            }
        }

        // Load input to internal buffer
        if ((pes[i][j].local_cycle_count <= BUFFER_SIZE)) // inital cycles to fill pipeline
        {
            PE_IFM_Buffer[k][pes[i][j].local_cycle_count - 1] = diagonal_ip;
        }
        else if ((pes[i][j].local_cycle_count + 1) % BUFFER_SIZE == 0) // follow up cycles
        {
            if (pes[i][j].local_cycle_count <= ((((PE_BREADTH * PE_LENGTH) - 1) + ((IF_WIDTH - (PE_BREADTH + PE_BREADTH - 1)) * BUFFER_SIZE))))
            {
                PE_IFM_Buffer[k][2] = diagonal_ip;
            }
        }

        // Loading weight to internal buffer
        if (pes[i][j].local_cycle_count <= BUFFER_SIZE)
        {
            PE_Wt_Buffer[k][pes[i][j].local_cycle_count - 1] = horizontal_ip;
        }
    }
}

void convolution(int DRAM_ip_data[BUFFER_SIZE - 1 + WT_LENGTH][IF_WIDTH], int DRAM_Wt_data[WT_LENGTH][WT_WIDTH], int DRAM_op_data[BUFFER_SIZE][OP_WIDTH])
{
    //--------------------------------------------------------------------------
    // Defines interface IO ports for HLS.
    //--------------------------------------------------------------------------

#pragma HLS INTERFACE m_axi depth = 1 port = DRAM_ip_data bundle = ifm
#pragma HLS INTERFACE m_axi depth = 1 port = DRAM_Wt_data bundle = ifm
#pragma HLS INTERFACE m_axi depth = 1 port = DRAM_op_data bundle = ifm

#pragma HLS INTERFACE s_axilite register port = return

    hls::stream<int, 3> psum_accumulation[3]; // depth 3
    hls::stream<int, 1> psum_DRAM[3];         // depth 1

#pragma HLS STREAM variable = psum_accumulation[0] type = fifo depth = 3
#pragma HLS STREAM variable = psum_accumulation[1] type = fifo depth = 3
#pragma HLS STREAM variable = psum_accumulation[2] type = fifo depth = 3
#pragma HLS STREAM variable = psum_DRAM type = fifo depth = 1

    int horizontal_pipes[3][3];
    int vertical_pipes[3][3];
    int diagonal_pipes[3][3];

#pragma HLS array_partition variable = horizontal_pipes dim = 0 complete
#pragma HLS array_partition variable = vertical_pipes dim = 0 complete
#pragma HLS array_partition variable = diagonal_pipes dim = 0 complete

#pragma HLS array_partition variable = PE_IFM_Buffer dim = 0 complete
#pragma HLS array_partition variable = PE_Wt_Buffer dim = 0 complete

    /*

    #pragma HLS bind_storage variable=horizontal_pipes type=RAM_T2P impl=bram
    #pragma HLS bind_storage variable=vertical_pipes type=RAM_T2P impl=bram
    #pragma HLS bind_storage variable=diagonal_pipes type=RAM_T2P impl=bram

    #pragma HLS bind_storage variable=PE_IFM_Buffer type=RAM_T2P impl=bram
    #pragma HLS bind_storage variable=PE_Wt_Buffer type=RAM_T2P impl=bram

    */

ITERATION_LOOP:
    for (int cycles = 0; cycles < active_cycles; cycles++)
    {
#pragma HLS pipeline

        // reads neccessary data from DRAM
        read_data_DRAM(DRAM_ip_data, DRAM_Wt_data, horizontal_pipes[0][0], horizontal_pipes[1][0],
                       horizontal_pipes[2][0], diagonal_pipes[0][0], diagonal_pipes[1][0], diagonal_pipes[2][0],
                       diagonal_pipes[2][1], diagonal_pipes[2][2]);

        // call the PEs unrolled PE number of times
    PE_LOOP:

        for (int j = 0; j < 3; j++)
        {
#pragma HLS unroll

            for (int i = 0; i < 3; i++)
            {

#pragma HLS unroll

                // int i = k % 3;
                // int j = k / 3;

                int k = i * 3 + j;
                // the below function handles data movement to neighbouring PE's , to external PSUM PE, and also the MAC operation
                processingelement(psum_accumulation[0], psum_accumulation[1], psum_accumulation[2], horizontal_pipes[i][j], horizontal_pipes[i][j + 1], vertical_pipes[i][j],
                                  vertical_pipes[i - 1][j], diagonal_pipes[i][j], diagonal_pipes[i - 1][j + 1], i, j, k);
            }
        }

        // PSUM outside systolic array
        sum_of_psums(psum_DRAM[0], psum_DRAM[1], psum_DRAM[2], psum_accumulation[0], psum_accumulation[1], psum_accumulation[2]);

        // write to DRAM
        write_data_DRAM(DRAM_op_data, psum_DRAM[0], psum_DRAM[1], psum_DRAM[2]);
    }
}