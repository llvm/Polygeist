#include <iostream>
#include "hls_stream.h"

#define STRIDE 1
#define PADDING 0
#define IF_LENGTH 5
#define IF_WIDTH 5
#define WT_LENGTH 3
#define WT_WIDTH 3
#define OP_LENGTH 3
#define OP_WIDTH 3

#define PE_LENGTH 3
#define PE_BREADTH 3
#define SUM_BUFFER_SIZE 3
#define BUFFER_SIZE 3

#define active_cycles PE_LENGTH + PE_BREADTH + (OP_WIDTH * 3)

using namespace std;
void reset();
void convolution(int DRAM_ip_data[BUFFER_SIZE - 1 + WT_LENGTH][IF_WIDTH], int DRAM_Wt_data[WT_LENGTH][WT_WIDTH], int DRAM_op_data[BUFFER_SIZE][OP_WIDTH]);