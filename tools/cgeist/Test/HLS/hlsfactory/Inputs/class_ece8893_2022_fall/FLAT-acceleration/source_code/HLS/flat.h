#ifndef FLAT_H_
#define FLAT_H_

#include <vector>
#include <ap_fixed.h>

//--------------------------------------------------------------------------
// Compiler Defines
//--------------------------------------------------------------------------
#ifdef HLS_SIM
    #include "config.h"
#endif

//--------------------------------------------------------------------------
// Type Conversions
//--------------------------------------------------------------------------
#ifdef  CSIM_DEBUG
    typedef float data_t;
#else
    typedef ap_fixed<16,3> data_t;
#endif 

typedef ap_uint<1024> MEM_TYPE;
    
#define IF_ROW 0 //Length Granularity
#define IF_HEAD 1 //Head Granularity
#define IF_BARCH 0 //Batch Granularity
#define BATCH_B 64 //How many batches we want to process at a time?
#define KEY_LENGTH_T 64 //How many words we want to process at a time?
#define QUERY_LENGTH_F 64 //How many words we want to process at a time?
#define VALUE_LENGTH_T 64 //How many words we want to process at a time?
#define NUM_HEAD_N 16 //How many heads we want to process at a time?
#define HEAD_DIM_H 64 //How many context scores per head we want to process at a time?

///////////////////////////////////////////
// Adjustable parameters :: How we want to set up the test?
// 1. FILE 
// 2/ Large piece of memory : Manually set length and batch we want to process at one time
///////////////////////////////////////////
#define FILE_TOTAL 10 //How many files we want to process? [Load TEN 64 * 64 * 16 * 64 bin file]

void GenerateKey(std::vector<data_t>& key_matrix);
void GenerateValue(std::vector<data_t>& value_matrix);
void GenerateQuery(std::vector<data_t>& query_matrix);
void GenerateBias(std::vector<data_t>& input_matrix);

/**
 * @brief Load tile from DRAM block
 * Apply Double Buffering if needed (ONE Block for data loading and the Other for Computation)
 * Not Provided index for loading for now. Assume only one block in DRAM to load
 */
void Load_Query_from_DRAM(int b, int n, data_t query_buffer[QUERY_LENGTH_F][HEAD_DIM_H], MEM_TYPE query[576][64][16]);
void Load_Key_from_DRAM(int b, int n, data_t key_buffer[KEY_LENGTH_T][HEAD_DIM_H], MEM_TYPE key[576][64][16]);
void Load_Value_from_DRAM(int b, int n, data_t value_buffer[KEY_LENGTH_T][HEAD_DIM_H], MEM_TYPE value[576][64][16]);
void Load_Bias_from_DRAM(int b, int n, data_t bias_buffer[QUERY_LENGTH_F][KEY_LENGTH_T], MEM_TYPE bias[64][16][64]);
void Write_Attention_Back(int b, int n, data_t attention_out[576][64][16][64], data_t attention_out_buffer[QUERY_LENGTH_F][HEAD_DIM_H]);

void Load_Query_ROW_Gran(int b, int n, data_t query_row_gran[QUERY_LENGTH_F][HEAD_DIM_H], data_t query[576][64][16][64]);
void Load_Key_ROW_Gran(int b, int n, data_t key_row_gran[KEY_LENGTH_T][HEAD_DIM_H], data_t key[576][64][16][64]);
void Load_Value_ROW_Gran(int b, int n, data_t value_row_gran[KEY_LENGTH_T][HEAD_DIM_H], data_t value[576][64][16][64]);
void Load_Bias_ROW_Gran(int b, int n,  data_t bias_row_gran[QUERY_LENGTH_F][KEY_LENGTH_T], data_t bias[576][16][64][64]);

//void Write_Attention_Back(int b, int n, data_t attention_out_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H], data_t attention_out_row_gran[QUERY_LENGTH_F][HEAD_DIM_H]);

void Fused_Logit_Operator(data_t query_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H], data_t key_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H], data_t bias_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T], data_t out_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T]);
void Softmax(data_t logit_out[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T], data_t softmanx_out[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T]);
void Fused_Attention_Operator(data_t softmax_out[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T], data_t value_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H],
                    data_t attention_out_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H]);

void computeLogit(data_t query_matrix[QUERY_LENGTH_F][HEAD_DIM_H], data_t key_matrix[KEY_LENGTH_T][HEAD_DIM_H],
					data_t bias_matrix[QUERY_LENGTH_F][KEY_LENGTH_T], data_t logit[QUERY_LENGTH_F][KEY_LENGTH_T], data_t max[QUERY_LENGTH_F]);
void computeAttention(data_t softmax[QUERY_LENGTH_F][KEY_LENGTH_T], data_t value_matrix[QUERY_LENGTH_F][KEY_LENGTH_T], data_t output[QUERY_LENGTH_F][KEY_LENGTH_T]);
void Inter_Softmax(data_t logit[QUERY_LENGTH_F][KEY_LENGTH_T], data_t softmax[QUERY_LENGTH_F][KEY_LENGTH_T], data_t max[QUERY_LENGTH_F]);

void Pipelined_FLAT(data_t query_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H], data_t key_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H], data_t bias_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T],
                     data_t value_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H], data_t attention_out_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H]);
/**
 * @brief This function is specifically for flat to save partial output
 * 
 */
void Save_Partial_Output();
void Store_Output_to_DRAM(data_t attention_out_buffer[64][64][16][64], data_t attention_out[576][64][16][64], int idx);


void FlatDataflow(data_t query[576][64][16][64], data_t key[576][64][16][64], data_t value[576][64][16][64], data_t bias[64][16][64][64], data_t attention_out[576][64][16][64]);

//void FlatDataflow(MEM_TYPE query[576][64][16], MEM_TYPE key[576][64][16], MEM_TYPE value[576][64][16], MEM_TYPE bias[64][16][64], data_t attention_out[576][64][16][64]);


void Load_Query_from_DRAM_old(data_t query_buffer[BATCH_B][QUERY_LENGTH_F][NUM_HEAD_N][HEAD_DIM_H], data_t query[576][64][16][64], int idx);
void Load_Key_from_DRAM_old(data_t key_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H], data_t key[576][64][16][64], int idx);
void Load_Value_from_DRAM_old(data_t value_buffer[BATCH_B][KEY_LENGTH_T][NUM_HEAD_N][HEAD_DIM_H], data_t value[576][64][16][64], int idx);
void Load_Bias_from_DRAM_old(data_t bias_buffer[BATCH_B][NUM_HEAD_N][QUERY_LENGTH_F][KEY_LENGTH_T], data_t bias[64][16][64][64], int idx);
#endif // MACRO
