#include "dcl.h"

typedef hls::stream<fixed_t> my_data_fifo;
#define UNROLL_FACTOR 8
#define UNROLL_FACTOR_2 10
#define Matrix_Unroll_Factor 25

/*
Loads a single row of Q or K into BRAM through as 512 bit bus. Also unpacks the data.

@param Q_local: Local buffer for one row
@param Q: Input 2d matrix
@param i: Specifies which row to load
*/
void load_variable_local_wide(fixed_t Q_local[dk], wide_bus_t Q[N][dk/32], int i) {
    #pragma HLS INLINE
    for (int k = 0; k < dk/32; ++k) {
        wide_bus_t packed_data = Q[i][k];  // Load 512-bit word into a local variable
        for (int slice = 0; slice < 32; ++slice) {
            Q_local[k * 32 + slice] = fixed_t(packed_data.range((slice + 1) * 16 - 1, slice * 16));
        }
    }
}

/*
Loads a 2d Matrix of V into BRAM through as 512 bit bus. Also unpacks the data.

@param V: A 512 bit 2d matrix
@param V_local: 2d matrix to be stored in BRAM
*/
void load_V_to_BRAM_wide(wide_bus_t V[N][dv/32], fixed_t V_local[N][dv]) {
    #pragma HLS INLINE
    load_memory_to_buffer_matrix_multiply_loop2:
    for (int k = 0; k < N; ++k) {
        load_memory_to_buffer_matrix_multiply_loop3:
        for (int j = 0; j < dv/32; ++j) {
            // #pragma HLS PIPELINE II=1
            wide_bus_t local_wide_packet=   V[k][j];  
            loop_to_unpack:
            for (int slice = 0; slice < 32; ++slice) {
                V_local[k][j * 32 + slice] = fixed_t(local_wide_packet.range((slice + 1) * 16 - 1, slice * 16));
            }
        }
    }
}


/*
Calculates softmax for a given 2d Matrix
@param matrix_fifo: Input FIFO from dot product computation
@param output_fifo: Output FIFO for softmax. To be fed to Matrix Multiplication
*/
void softmax_HLS(my_data_fifo &matrix_fifo, my_data_fifo &output_fifo) {

    fixed_t matrix_row[N];  // Local buffer to be fed from FIFO. Storing into a buffer for unrolling
    #pragma HLS ARRAY_PARTITION variable=matrix_row cyclic factor=UNROLL_FACTOR_2
    softmax_loop2:
    for (int i = 0; i < N; ++i) {
        //Computes maximum element in each row
        matrix_row[0] = matrix_fifo.read(); 
        ap_fixed<32, 8> max_val;
        softmax_loop3A:
        for (int j = 1; j < N; ++j) {
            #pragma HLS PIPELINE II=1
            matrix_row[j] = matrix_fifo.read(); 
            if (matrix_row[j] > max_val)
                { max_val = matrix_row[j];  // Find max
                }
            }
    
        fixed_t exponentials[N]; //local buffer to store exponentials.
        #pragma HLS ARRAY_PARTITION variable=exponentials cyclic factor=UNROLL_FACTOR_2
        ap_fixed<32, 8> sum = 0;
        softmax_exponential_approximation_loop3B:
        for (int j = 0; j < N; ++j) {
            #pragma HLS UNROLL factor=UNROLL_FACTOR_2 skip_exit_check
            fixed_t x = matrix_row[j] - max_val;
            //Do Taylor expansion for exponentials
            exponentials[j]=2^x;
            sum += exponentials[j];
        }
        
        softmax_divide_by_exponetial_loop_3D:
        for (int j = 0; j < N; ++j) {
            #pragma HLS PIPELINE II=1
            output_fifo << (exponentials[j] / sum);  // Write to output FIFO
        }
    }
}

/*
Computes the dot product of two rows. Internally called by Dot_product
*/
void compute_dot_product(fixed_t Q_local[dk], fixed_t K_local[dk], ap_fixed<32, 8> &sum) {
   #pragma HLS INLINE
    sum = 0;
    for (int k = 0; k < dk; ++k) {
        sum+= Q_local[k]* K_local[k];
    }
}

/*
Calculates the dot product of 2d matrix of Q and K. Stores the result into a fifo.

@param Q: 2d matrix of Q
@param K: 2d matrix of K
@param attention_fifo: FIFO to store the result
@param scale: Scaling factor

*/
void Dot_product(wide_bus_t Q[N][dk/32], wide_bus_t K[N][dk/32], my_data_fifo &attention_fifo, fixed_t scale) {

    Dot_Product_Loop2:
    for (int i = 0; i < N; ++i) {
        fixed_t Q_local[dk];  // stores Q[b][i][k]
        #pragma HLS ARRAY_PARTITION variable=Q_local cyclic factor=UNROLL_FACTOR dim=1
        //load_variable_local(Q_local, Q, i);
        load_variable_local_wide(Q_local, Q, i);
        
        dot_product_loop4: 
        for (int j = 0; j < N; ++j) {
            //Botlleneck Loop. Increase II for performance, but huge impact on resource
            #pragma HLS PIPELINE II=4
            ap_fixed<32, 8> sum = 0;
            fixed_t K_local[dk];
            #pragma HLS ARRAY_PARTITION variable=K_local cyclic factor=UNROLL_FACTOR dim=1
            load_variable_local_wide(K_local, K, j); //loads a row of K
            compute_dot_product(Q_local, K_local, sum); //Computes dot product
            attention_fifo << (sum * scale);  //Write to FIFO
        }
    }
}

/*
Computes the matrix multiplication of V and the attention matrix from Softmax.
@param V: 2d matrix of V
@param attention_fifo: FIFO which feeds the matrix multiplication function
@param Output_FIFO: FIFO to store the result
*/
void Matrix_Multiply(wide_bus_t V[N][dv/32], my_data_fifo &attention_fifo, my_data_fifo &Output_FIFO) {

    fixed_t V_local[N][dv];
    #pragma HLS ARRAY_PARTITION variable=V_local cyclic factor=Matrix_Unroll_Factor dim=2
    load_V_to_BRAM_wide(V, V_local);

    compute_matrix_multiply_loop2B:
    for (int i = 0; i < N; ++i) {
        fixed_t attention_row[N];
        #pragma HLS ARRAY_PARTITION variable=attention_row cyclic factor=UNROLL_FACTOR_2 dim=1
        for (int k = 0; k < N; ++k) {
            #pragma HLS PIPELINE II=1
            attention_row[k] = attention_fifo.read();
        }
        
        compute_matrix_multiply_loop3:
        for (int j = 0; j < dv; ++j) {
            //Critical Loop for performance
            //#pragma HLS PIPELINE II=1
            ap_fixed<32, 8> sum = 0;
            compute_matrix_multiply_loop4:
            for (int k = 0; k < N; ++k) {
                #pragma HLS UNROLL factor=Matrix_Unroll_Factor skip_exit_check
                sum += attention_row[k] * V_local[k][j];
            }
            Output_FIFO << sum;
        }
    }
}

/*
Reads from the FIFO and stores the result into DRAM
@param outFifo: FIFO being fed from Matrix Multiplication function
@param output: 2d Matrix of Output (DRAM)
*/
void store_output( my_data_fifo &outFifo, fixed_t output[N][dv]) {

        write_fifo_to_memory_loop_1:
        for (int i = 0; i < N; ++i) {
            write_fifo_to_memory_loop_2:
            for (int j = 0; j < dv; ++j) {
                #pragma HLS PIPELINE II=1
                outFifo >> output[i][j];
            }
        }
}


//main HLS kernel
void compute_attention_HLS(wide_bus_t Q[B][N][dk/32], wide_bus_t K[B][N][dk/32], wide_bus_t V[B][N][dv/32], fixed_t Output[B][N][dv]) {
    #pragma HLS interface m_axi port=Q offset=slave bundle=mem1 
    #pragma HLS interface m_axi port=K offset=slave bundle=mem2  
    #pragma HLS interface m_axi port=V offset=slave bundle=mem3
    #pragma HLS interface m_axi port=Output offset=slave bundle=mem4  
    #pragma HLS interface s_axilite port=return
    
    ap_fixed<32, 8> scale_const = 1.0 / sqrt(dk);
    //Compute batch wise
    for (int b = 0; b < B; ++b) {
        #pragma HLS DATAFLOW
        ap_fixed<32, 8> scale = scale_const;
        my_data_fifo attention_fifo; //FIFO to store output from dot product
        my_data_fifo attention_softmax_fifo; //FIFO to store output from softmax
        my_data_fifo Output_FIFO; //FIFO to store output from matrix multiplication
        #pragma HLS STREAM variable=attention_fifo depth=37
        #pragma HLS STREAM variable=attention_softmax_fifo depth=100
        #pragma HLS STREAM variable=Output_FIFO depth=2

        //Main Functions:
        //Step 1: Dot Product
        Dot_product(Q[b], K[b], attention_fifo, scale);
        //Step 2: Softmax
        softmax_HLS(attention_fifo, attention_softmax_fifo);
        //Step 3: Matrix Multiply
        Matrix_Multiply(V[b], attention_softmax_fifo, Output_FIFO);
        //Step4: Store Output
        store_output(Output_FIFO, Output[b]);

    }
}