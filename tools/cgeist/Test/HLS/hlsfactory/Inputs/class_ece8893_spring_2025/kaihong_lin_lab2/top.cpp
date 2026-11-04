
#include "dcl.h"
#include <hls_math.h>
typedef ap_uint<512> wide_bus_t;
#define ELEMS_PER_BUS (512 / 8 / sizeof(fixed_t))



// Function to load Key (K) and Value (V) matrices from wide bus data into buffer matrices
void LOAD_KV(
    fixed_t K[N][dk],          // Input Key matrix of size N x dk
    fixed_t V[N][dv],          // Input Value matrix of size N x dv
    fixed_t K_buf[N][dk],      // Output buffer for Key matrix
    fixed_t V_buf[N][dv])      // Output buffer for Value matrix
{
    // Loop to load Key matrix data
    LOAD_K_LOOP:
    for (int i = 0; i < N * dk; i += ELEMS_PER_BUS) // Iterate over elements in steps of ELEMS_PER_BUS
    {
    #pragma HLS PIPELINE II = 1 // Pipeline the loop to improve performance, II = 1 means initiate a new iteration every clock cycle
        wide_bus_t temp = *reinterpret_cast<wide_bus_t *>(&K[0][i]); // Reinterpret the address of K[0][i] as a wide bus type
        for (int ii = 0; ii < ELEMS_PER_BUS; ii++) // Unroll loop to process ELEMS_PER_BUS elements per iteration
        {
    #pragma HLS UNROLL // Unroll the loop to process elements in parallel
            int index = i + ii; // Calculate the current element index
            if (index < N * dk) // Check if the index is within the bounds of the matrix
            {
                int row = index / dk; // Calculate the row index
                int col = index % dk; // Calculate the column index
                ap_uint<16> bits_16 = temp.range(16 * (ii + 1) - 1, 16 * ii); // Extract 16 bits from the wide bus
                K_buf[row][col] = *reinterpret_cast<fixed_t *>(&bits_16); // Reinterpret the 16-bit value as fixed_t and store in buffer
            }
        }
    }
    // Loop to load Value matrix data
    LOAD_V_LOOP:
    for (int i = 0; i < N * dv; i += ELEMS_PER_BUS) // Iterate over elements in steps of ELEMS_PER_BUS
    {
    #pragma HLS PIPELINE II = 1 // Pipeline the loop to improve performance, II = 1 means initiate a new iteration every clock cycle
        wide_bus_t temp = *reinterpret_cast<wide_bus_t *>(&V[0][i]); // Reinterpret the address of V[0][i] as a wide bus type
        for (int ii = 0; ii < ELEMS_PER_BUS; ii++) // Unroll loop to process ELEMS_PER_BUS elements per iteration
        {
    #pragma HLS UNROLL // Unroll the loop to process elements in parallel
            int index = i + ii; // Calculate the current element index
            if (index < N * dv) // Check if the index is within the bounds of the matrix
            {
                int row = index / dv; // Calculate the row index
                int col = index % dv;
                ap_uint<16> bits_16 = temp.range(16 * (ii + 1) - 1, 16 * ii); // Extract 16 bits from the wide bus
                V_buf[row][col] = *reinterpret_cast<fixed_t *>(&bits_16); // Reinterpret the 16-bit value as fixed_t and store in buffer
            }
        }
    }
}

// Function to load a line of data from a fixed-point array Q to a buffer Q_line_buf
void LOAD_Q_LINE(
    fixed_t Q[dk],              // Input array of fixed-point numbers
    fixed_t Q_line_buf[dk]      // Output buffer to store the loaded line of data
)
{
    // Loop label for the pipeline process
    LINE_LOAD:
    for(int i=0; i<dk; i+=ELEMS_PER_BUS){ // Loop over the input array in chunks of ELEMS_PER_BUS
    #pragma HLS PIPELINE II=1            // HLS pragma to pipeline the loop with an initiation interval of 1
        wide_bus_t temp = *reinterpret_cast<wide_bus_t*>(&Q[i]); // Reinterpret the chunk of Q as a wide bus type
        for(int ii=0; ii<ELEMS_PER_BUS; ii++){ // Loop over each element in the chunk
        #pragma HLS UNROLL                // HLS pragma to unroll the loop for parallel processing
            if((i+ii) < dk){              // Check if the current index is within bounds
                ap_uint<16> bits = temp.range(16*(ii+1)-1,  16*ii); // Extract 16 bits from the wide bus
                Q_line_buf[i+ii] = *reinterpret_cast<fixed_t*>(&bits); // Reinterpret the bits as a fixed-point number and store in the buffer
            }
        }
    }
}


// Function to store output data from a line buffer to a wider bus
void STORE_OUTPUT(
    fixed_t output_line[dv],  // Input array representing the line buffer
    fixed_t Output[dv]        // Output array representing the wider bus storage
)
{
    // Loop to process chunks of the output line buffer
    STORE_LINE:
    for(int j=0; j<dv; j+=ELEMS_PER_BUS){
    #pragma HLS PIPELINE // Enable pipelining to improve throughput
        wide_bus_t temp; // Temporary variable to hold the wide bus data
        // Loop to process each element within the current chunk
        for(int jj=0; jj<ELEMS_PER_BUS; jj++){
        #pragma HLS UNROLL // Unroll the loop to process elements in parallel
            // Reinterpret the fixed_t data as a 16-bit unsigned integer
            ap_uint<16> bits = *reinterpret_cast<ap_uint<16>*>(&output_line[j+jj]);
            // Assign the 16-bit data to the appropriate range in the temporary wide bus
            temp.range(16*(jj+1)-1,  16*jj) = bits;
        }
        // Store the temporary wide bus data into the output array
        *reinterpret_cast<wide_bus_t*>(&Output[j]) = temp;
    }
}
void COMPUTE_QK_LINE(
    fixed_t Q_line[dk],       // Input array representing the query vector
    fixed_t K_buf[N][dk],     // Input 2D array representing the key vectors
    fixed_t attn_line[N])     // Output array to store the computed attention scores
{
    // Calculate the scaling factor for the dot product
    const fixed_t scale = (fixed_t)(1.0/sqrt((float)dk));
    
    // Loop over each key vector to compute the attention score
    QK_LINE_LOOP:
    for(int j=0; j<N; j++){
    // #pragma HLS UNROLL off = true
    // #pragma HLS LOOP_FLATTEN off = true
    #pragma HLS PIPELINE // Enable pipelining to improve performance
        ap_fixed<32, 8> sum = 0; // Initialize the sum for the dot product
        // Loop over the dimensions of the query and key vectors in chunks of 128
        for(int k=0; k<dk; k+=128){
        // #pragma HLS UNROLL factor=2
        #pragma HLS PIPELINE // Enable pipelining within the inner loop
            ap_fixed<32, 8> sum_local[128]; // Local sum array to accumulate partial sums
            // Compute the dot product for the current chunk
            for(int kk=0; kk<128; kk++){
            #pragma HLS UNROLL // Unroll the loop to improve performance
            sum_local[kk] = Q_line[k+kk] * K_buf[j][k+kk];
            }
            // Accumulate the partial sums into the total sum
            for(int kk=0; kk<128; kk++){
            #pragma HLS UNROLL // Unroll the loop to improve performance
                sum += sum_local[kk];
            }
            // sum += Q_line[k] * K_buf[j][k]; // Original line commented out
        }
        // Scale the sum and store it in the attention line array
        attn_line[j] = sum * scale;
    }
}
void SOFTMAX_LINE(
    fixed_t attn_line_in[N], // Input array of attention scores
    fixed_t attn_line_out[N]) // Output array of softmax normalized attention scores
{
    ap_fixed<32, 8> max_val = attn_line_in[0]; // Initialize max value with the first element of the input array
    // Block-wise maximum value calculation
    MAX_BLOCK:
    for(int j=0; j<N; j+=100){ // Process 100 elements each time
    #pragma HLS PIPELINE // HLS directive: pipeline execution, improve parallelism
        ap_fixed<32,8> block_max = attn_line_in[j]; // Initialize the max value of the current block with the first element of the block
        for(int jj=1; jj<100; jj++){ // Traverse the rest elements of the current block
            #pragma HLS UNROLL // HLS directive: fully unroll the loop, improve parallelism
            if(attn_line_in[j+jj] > block_max) // If the current element is greater than the max value in the block
                block_max = attn_line_in[j+jj]; // Update the max value in the block
        }
        if(block_max > max_val) // If the max value of the current block is greater than the global max value
            max_val = block_max; // Update the global max value
    }
    ap_fixed<32,8> sum = 0; // Initialize the sum variable
    // Exponential sum
    EXP_SUM:
    for (int j = 0; j < N; j += 1) // Traverse all input elements
    {
        #pragma HLS UNROLL factor=25 // HLS directive: partially unroll the loop, improve parallelism
        #pragma HLS PIPELINE // HLS directive: pipeline execution, improve parallelism
        attn_line_out[j] = hls::exp(attn_line_in[j] - max_val); // Calculate the exponential value of each element and subtract the max value
        sum += attn_line_out[j]; // TODO: OPTIMIZE HERE MAYBE
    }
    // Normalization
    NORMALIZE:
    for (int j = 0; j < N; j += 1)
    {
        #pragma HLS UNROLL
        attn_line_out[j] = attn_line_out[j] / sum;
    }
}

// Function to apply attention to a line of data
void APPLY_ATTN_LINE(
    fixed_t attn_line[N],       // Input attention weights for N elements
    fixed_t V_buf[N][dv],
    fixed_t output_line[dv])
{
    // Temporary buffer to store intermediate results
    ap_fixed<32, 8> output_line_temp[dv];
    // Partition the temporary buffer for parallel processing
    #pragma HLS ARRAY_PARTITION variable=output_line_temp cyclic factor=64
    // Initialize the temporary buffer to zero
    APPLY_ATTN_LINE1:for(int j=0; j<dv; j++){
        #pragma HLS UNROLL
        output_line_temp[j] = 0;
    }
    // Compute the weighted sum of V_buf using attention weights
    APPLY_ATTN_LINE2: 
    for(int k=0; k<N; k++){
        // Disable unrolling and loop flattening for this loop
        #pragma HLS UNROLL off = true
        #pragma HLS LOOP_FLATTEN off = true
        // Pipeline the loop to improve performance
        #pragma HLS PIPELINE II=1
        for(int j=0; j<dv; j++){
            // Unroll the inner loop for parallel processing
            #pragma HLS UNROLL factor = 128
            output_line_temp[j] += attn_line[k] * V_buf[k][j];
        }
    }
    // Copy the results from the temporary buffer to the output buffer
    APPLY_ATTN_LINE3: for(int j=0; j<dv; j++){
        #pragma HLS UNROLL factor = 128
        output_line[j] = output_line_temp[j];
    }
}



void compute_attention_HLS(
    fixed_t Q[B][N][dk],
    fixed_t K[B][N][dk],
    fixed_t V[B][N][dv],
    fixed_t Output[B][N][dv])
{
#pragma HLS INTERFACE m_axi port=Q bundle=mem1 offset=slave
#pragma HLS INTERFACE m_axi port=K bundle=mem1 offset=slave
#pragma HLS INTERFACE m_axi port=V bundle=mem1 offset=slave
#pragma HLS INTERFACE m_axi port=Output bundle=mem2 offset=slave
// #pragma HLS INTERFACE s_axilite port = return
#pragma HLS INTERFACE ap_ctrl_hs port=return
       
BATCH_LOOP:
    for(int b=0; b<B; b++){
            #pragma HLS UNROLL off = true
            #pragma HLS LOOP_FLATTEN off = true
            #pragma HLS PIPELINE off = true
    
            // Global Buffer
            fixed_t K_buf[N][dk];
            fixed_t V_buf[N][dv];
            #pragma HLS ARRAY_PARTITION variable=K_buf cyclic factor=128 dim=2
            #pragma HLS ARRAY_PARTITION variable=V_buf cyclic factor=128 dim=2
            LOAD_KV(K[b],V[b],K_buf,V_buf);


        Q_LINE_LOOP:
        for(int i=0; i<N; i++){
            #pragma HLS DATAFLOW        
            #pragma HLS stable variable=K_buf
            #pragma HLS stable variable=V_buf
            
            // Local Buffer    
            fixed_t Q_line[dk];
            #pragma HLS ARRAY_PARTITION variable=Q_line cyclic factor=128
            fixed_t attn_line1[N];
            #pragma HLS ARRAY_PARTITION variable=attn_line1 cyclic factor=100
            fixed_t attn_line2[N];
            #pragma HLS ARRAY_PARTITION variable=attn_line2 cyclic factor=100
            fixed_t output_line[dv];
            #pragma HLS ARRAY_PARTITION variable=output_line cyclic factor=128

            // Phase 1: Load a single row of Q
            LOAD_Q_LINE(Q[b][i], Q_line);

            // Phase 2: Compute QK^T row
            COMPUTE_QK_LINE(Q_line, K_buf, attn_line1);

            // Phase 3: Row-wise Softmax
            SOFTMAX_LINE(attn_line1, attn_line2);

            // Phase 4: Apply attention
            APPLY_ATTN_LINE(attn_line2, V_buf, output_line);

            // Phase 5: Store the result row
            STORE_OUTPUT(output_line, Output[b][i]);
        }
    }
}
