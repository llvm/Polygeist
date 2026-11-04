#include <stdlib.h>
#include <stdio.h>

//#include "dcl.h"
#include <vector> 
#include <algorithm>
#include <string.h>
#include <iostream>

#include <cstddef>
//#include "D:\vivado2018setup\Vivado\2018.3\include\gmp.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#include <algorithm>

typedef ap_fixed<32, 10> FM_TYPE;
typedef ap_fixed<32, 10> WT_TYPE;

#define WEIGHT_LAYER1_INPUTDIM 67
#define WEIGHT_LAYER1_OUTPUTDIM 122
#define WEIGHT_LAYER2_INPUTDIM 122
#define WEIGHT_LAYER2_OUTPUTDIM 122
#define TIME_STEP 10

#define NUM_EDGE 51732
#define NUM_NODE 3783
#define LAYER1_INPUTDIM 67
#define TOTAL_TIME 192
#define TIME_STEP 10

#define MAX_DEGREE 160

#define SUB_GRAPH_NODE_NUM 600
#define SUB_GRAPH_EDGE_NUM 1800



#define INPUTDIM 67
#define OUTPUTDIM 67

using namespace std;


float gcn_node_weight_layer1[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];    // [67,122]
WT_TYPE gcn_node_weight_layer1_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM];

WT_TYPE gcn_node_weight_layer1_buffer[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];


float gru_update_layer1_W[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
float gru_update_layer1_U[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
float gru_update_layer1_bias[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];  //[67,122]
WT_TYPE gru_update_layer1_bias_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM];

WT_TYPE gru_update_layer1_bias_buffer[WEIGHT_LAYER1_OUTPUTDIM];

float gru_reset_layer1_W[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
float gru_reset_layer1_U[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
float gru_reset_layer1_bias[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];

float gru_htilda_layer1_W[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
float gru_htilda_layer1_U[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
float gru_htilda_layer1_bias[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];

float topk_layer1_W[WEIGHT_LAYER1_INPUTDIM][1];



//create weights for convLSTM

float rnn_update_W[INPUTDIM][OUTPUTDIM];
float rnn_update_U[INPUTDIM][OUTPUTDIM];
float rnn_update_bias[OUTPUTDIM];
float rnn_reset_W[INPUTDIM][OUTPUTDIM];
float rnn_reset_U[INPUTDIM][OUTPUTDIM];
float rnn_reset_bias[OUTPUTDIM];
float rnn_htilda_W[INPUTDIM][OUTPUTDIM];
float rnn_htilda_U[INPUTDIM][OUTPUTDIM];
float rnn_htilda_bias[OUTPUTDIM];
float rnn_output_W[INPUTDIM][OUTPUTDIM];
float rnn_output_U[INPUTDIM][OUTPUTDIM];
float rnn_output_bias[OUTPUTDIM];
float node_embedding[NUM_NODE][INPUTDIM];
float edge_embedding[NUM_EDGE][INPUTDIM];
float cell_state[NUM_NODE][INPUTDIM];
float edge_info_temp[103464];
int edge_info[103464];
float edge_num_temp[TOTAL_TIME];
int edge_num[TOTAL_TIME];
float node_num_temp[TOTAL_TIME];
int node_num[TOTAL_TIME];
int edge_list[2 * SUB_GRAPH_EDGE_NUM];

FM_TYPE node_embedding_fixed[NUM_NODE * INPUTDIM];
FM_TYPE cell_table_fixed[NUM_NODE * INPUTDIM];
FM_TYPE edge_embedding_fixed[SUB_GRAPH_EDGE_NUM * INPUTDIM];

float neighborhood_renumbering_table_temp[TOTAL_TIME][NUM_NODE];
int neighborhood_renumbering_table[TOTAL_TIME][NUM_NODE];
float neighborhood_reverse_renumbering_table_temp[TOTAL_TIME][578];
int neighborhood_reverse_renumbering_table[TOTAL_TIME][578];


// GCN and GRU weights for layer2
float gcn_node_weight_layer2[WEIGHT_LAYER2_INPUTDIM][WEIGHT_LAYER2_OUTPUTDIM]; //[122,122]
WT_TYPE gcn_node_weight_layer2_fixed[WEIGHT_LAYER2_INPUTDIM * WEIGHT_LAYER2_OUTPUTDIM];

WT_TYPE gcn_node_weight_layer2_buffer[WEIGHT_LAYER2_INPUTDIM][WEIGHT_LAYER2_OUTPUTDIM];


float gru_update_layer2_W[WEIGHT_LAYER2_INPUTDIM][WEIGHT_LAYER2_INPUTDIM];   //[122,122]
//WT_TYPE gru_update_layer2_W_fixed[WEIGHT_LAYER2_INPUTDIM * WEIGHT_LAYER2_OUTPUTDIM];  //for the third MLP part


float gru_update_layer2_U[WEIGHT_LAYER2_INPUTDIM][WEIGHT_LAYER2_INPUTDIM];
float gru_update_layer2_bias[WEIGHT_LAYER2_INPUTDIM][WEIGHT_LAYER2_OUTPUTDIM];
WT_TYPE gru_update_layer2_bias_fixed[WEIGHT_LAYER2_INPUTDIM * WEIGHT_LAYER2_OUTPUTDIM];

WT_TYPE gru_update_layer2_bias_buffer[WEIGHT_LAYER2_OUTPUTDIM];



float gru_reset_layer2_W[WEIGHT_LAYER2_INPUTDIM][WEIGHT_LAYER2_INPUTDIM];
float gru_reset_layer2_U[WEIGHT_LAYER2_INPUTDIM][WEIGHT_LAYER2_INPUTDIM];
float gru_reset_layer2_bias[WEIGHT_LAYER2_INPUTDIM][WEIGHT_LAYER2_OUTPUTDIM];

float gru_htilda_layer2_W[WEIGHT_LAYER2_INPUTDIM][WEIGHT_LAYER2_INPUTDIM];
float gru_htilda_layer2_U[WEIGHT_LAYER2_INPUTDIM][WEIGHT_LAYER2_INPUTDIM];
float gru_htilda_layer2_bias[WEIGHT_LAYER2_INPUTDIM][WEIGHT_LAYER2_OUTPUTDIM];

float topk_layer2_W[WEIGHT_LAYER2_INPUTDIM][1];

// final classfication layer weights

float mlp_layer1_weight[77][244];
float mlp_layer1_bias[77];
float mlp_layer2_weight[2][77];
float mlp_layer2_bias[2];


float f_edge_list[NUM_EDGE * 2];
float f_edge_list_time[TIME_STEP][NUM_EDGE * 2];

float edge_val[NUM_EDGE];
float edge_val_time[TIME_STEP][NUM_EDGE];

char info_file_edge_list[1000];
char info_file_edge_val[1000];

float node_embedding_list[NUM_NODE][2];
float node_embedding_list_time[TIME_STEP][NUM_NODE][2];

float node_embedding_val[NUM_NODE];
float node_embedding_val_time[TIME_STEP][NUM_NODE];



char info_file_node_embeddings_list[1000];
char info_file_node_embedding_val[1000];

//float node_embedding[NUM_NODE][WEIGHT_LAYER1_INPUTDIM];
float node_embedding_time[TIME_STEP][NUM_NODE][WEIGHT_LAYER1_INPUTDIM];


FM_TYPE node_embedding_time_fixed[NUM_NODE * INPUTDIM];



float num_edges[TOTAL_TIME + 1];
int num_edges_time[TIME_STEP];
int num_nodes_time[TIME_STEP];

int node_out_degree[NUM_NODE];
int node_out_degree_time[TIME_STEP][NUM_NODE];


float node_embedding_subgraph[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_INPUTDIM];
float edge_embedding_subgraph[SUB_GRAPH_EDGE_NUM][WEIGHT_LAYER1_INPUTDIM];

int degree_table[SUB_GRAPH_NODE_NUM * 3];
int neighbor_table[SUB_GRAPH_NODE_NUM * MAX_DEGREE * 2];
int neighborhood_ref_table[NUM_NODE];

FM_TYPE node_embedding_fixed_output[SUB_GRAPH_NODE_NUM * INPUTDIM];


std::vector <int> node_record_vector;
std::vector <int> node_record_vector_previous;
std::vector <int> node_record_vector_new;


WT_TYPE node_embedding_in[SUB_GRAPH_NODE_NUM * INPUTDIM];
WT_TYPE edge_embedding_in[SUB_GRAPH_EDGE_NUM * INPUTDIM];




FM_TYPE node_embedding_table[SUB_GRAPH_NODE_NUM][INPUTDIM];
FM_TYPE edge_embedding_table[SUB_GRAPH_EDGE_NUM][INPUTDIM];
FM_TYPE cell_table[SUB_GRAPH_NODE_NUM][OUTPUTDIM];


WT_TYPE output_table[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER2_OUTPUTDIM];
//int edge_list[SUB_GRAPH_EDGE_NUM * 2];


WT_TYPE gnn_update_weight_W[INPUTDIM][OUTPUTDIM];
WT_TYPE gnn_update_weight_U[INPUTDIM][OUTPUTDIM];
WT_TYPE gnn_update_weight_bias[OUTPUTDIM];

WT_TYPE gnn_reset_weight_W[INPUTDIM][OUTPUTDIM];
WT_TYPE gnn_reset_weight_U[INPUTDIM][OUTPUTDIM];
WT_TYPE gnn_reset_weight_bias[OUTPUTDIM];


WT_TYPE gnn_htilda_weight_W[INPUTDIM][OUTPUTDIM];
WT_TYPE gnn_htilda_weight_U[INPUTDIM][OUTPUTDIM];
WT_TYPE gnn_htilda_weight_bias[OUTPUTDIM];

WT_TYPE gnn_output_weight_W[INPUTDIM][OUTPUTDIM];
WT_TYPE gnn_output_weight_U[INPUTDIM][OUTPUTDIM];
WT_TYPE gnn_output_weight_bias[OUTPUTDIM];



WT_TYPE gnn_update_weight_W_fixed[INPUTDIM * OUTPUTDIM];
WT_TYPE gnn_update_weight_U_fixed[INPUTDIM * OUTPUTDIM];
WT_TYPE gnn_update_weight_bias_fixed[OUTPUTDIM];

WT_TYPE gnn_reset_weight_W_fixed[INPUTDIM * OUTPUTDIM];
WT_TYPE gnn_reset_weight_U_fixed[INPUTDIM * OUTPUTDIM];
WT_TYPE gnn_reset_weight_bias_fixed[OUTPUTDIM];

WT_TYPE gnn_htilda_weight_W_fixed[INPUTDIM * OUTPUTDIM];
WT_TYPE gnn_htilda_weight_U_fixed[INPUTDIM * OUTPUTDIM];
WT_TYPE gnn_htilda_weight_bias_fixed[OUTPUTDIM];


WT_TYPE gnn_output_weight_W_fixed[INPUTDIM * OUTPUTDIM];
WT_TYPE gnn_output_weight_U_fixed[INPUTDIM * OUTPUTDIM];
WT_TYPE gnn_output_weight_bias_fixed[OUTPUTDIM];


float edge_embedding_subgraph_time[TIME_STEP][SUB_GRAPH_EDGE_NUM][INPUTDIM];
float node_embedding_subgraph_time[TIME_STEP][SUB_GRAPH_NODE_NUM][INPUTDIM];
int neighborhood_ref_table_time[TIME_STEP][NUM_NODE];

int neighborhood_inverse_ref_table_time[TIME_STEP][NUM_NODE];
int edge_list_time[TIME_STEP][SUB_GRAPH_EDGE_NUM * 2];


FM_TYPE node_embedding_table_1[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_INPUTDIM];
FM_TYPE edge_embedding_table_1[SUB_GRAPH_EDGE_NUM][WEIGHT_LAYER1_INPUTDIM];
FM_TYPE node_embedding_table_2[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_INPUTDIM];
FM_TYPE edge_embedding_table_2[SUB_GRAPH_EDGE_NUM][WEIGHT_LAYER1_INPUTDIM];

int origin_node_index_time[TIME_STEP][SUB_GRAPH_NODE_NUM];
int origin_node_index_time_old[TIME_STEP][SUB_GRAPH_NODE_NUM];
std::vector <int> origin_node_index_time_vector;
std::vector <int> origin_node_index_time_old_vector;

std::vector <int> old_node_on_chip_position_vector;
std::vector <int> replace_node_on_chip_position_vector;


std::vector <int> valid_node_position;
std::vector <int> invalid_node_position;
int old_node_on_chip_position[TIME_STEP][SUB_GRAPH_NODE_NUM];
int replace_node_on_chip_position[TIME_STEP][SUB_GRAPH_NODE_NUM];

int origin_node_index_time_fixed[TIME_STEP * SUB_GRAPH_NODE_NUM];

int old_node_num[TIME_STEP];
int new_node_num[TIME_STEP];

int mask[TIME_STEP][SUB_GRAPH_NODE_NUM];

int load_list[TIME_STEP][SUB_GRAPH_NODE_NUM];
int store_list[TIME_STEP][SUB_GRAPH_NODE_NUM];
int valid_list[TIME_STEP][SUB_GRAPH_NODE_NUM];

int load_list_size[TIME_STEP];
int store_list_size[TIME_STEP];
int valid_list_size[TIME_STEP];
std::vector <int> load_vector;
std::vector <int> store_vector;
std::vector <int> valid_vector;

std::vector <int> new_node_index_vector;

FM_TYPE node_embedding_table_temp[SUB_GRAPH_NODE_NUM][INPUTDIM];
FM_TYPE cell_state_table_temp[SUB_GRAPH_NODE_NUM][OUTPUTDIM];

//FM_TYPE cell_table_fixed[NUM_NODE * OUTPUTDIM];


void load_weights_convLSTM()
{

	printf("Loading weights for evolveGCN (RNN + GCN)...\n");

	FILE* f;
	f = fopen("convLSTM.bin", "rb");
	int number_count = 0;
	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(rnn_update_W, sizeof(float), INPUTDIM * OUTPUTDIM, f);
	number_count += INPUTDIM * OUTPUTDIM;

	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(rnn_update_U, sizeof(float), INPUTDIM * OUTPUTDIM, f);
	number_count += INPUTDIM * OUTPUTDIM;

	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(rnn_update_bias, sizeof(float), OUTPUTDIM, f);
	number_count += OUTPUTDIM;


	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(rnn_reset_W, sizeof(float), INPUTDIM * OUTPUTDIM, f);
	number_count += INPUTDIM * OUTPUTDIM;

	

	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(rnn_reset_U, sizeof(float), INPUTDIM * OUTPUTDIM, f);
	number_count += INPUTDIM * OUTPUTDIM;

	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(rnn_reset_bias, sizeof(float), OUTPUTDIM, f);
	number_count += OUTPUTDIM;


	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(rnn_htilda_W, sizeof(float), INPUTDIM * OUTPUTDIM, f);
	number_count += INPUTDIM * OUTPUTDIM;

	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(rnn_htilda_U, sizeof(float), INPUTDIM * OUTPUTDIM, f);
	number_count += INPUTDIM * OUTPUTDIM;

	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(rnn_htilda_bias, sizeof(float), OUTPUTDIM, f);
	number_count += OUTPUTDIM;


	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(rnn_output_W, sizeof(float), INPUTDIM * OUTPUTDIM, f);
	number_count += INPUTDIM * OUTPUTDIM;

	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(rnn_output_U, sizeof(float), INPUTDIM * OUTPUTDIM, f);
	number_count += INPUTDIM * OUTPUTDIM;

	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(rnn_output_bias, sizeof(float), OUTPUTDIM, f);
	number_count += OUTPUTDIM;

	

	
}

void load_node_embedding() {
	FILE* f;
	f = fopen("node_embedding.bin", "rb");
	int number_count = 0;
	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(node_embedding, sizeof(float), NUM_NODE * INPUTDIM, f);
	number_count += NUM_NODE * INPUTDIM;
	
	
}


void load_edge_embedding() {
	FILE* f;
	f = fopen("edge_embedding.bin", "rb");
	int number_count = 0;
	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(edge_embedding, sizeof(float), NUM_EDGE * INPUTDIM, f);
	number_count += NUM_EDGE * INPUTDIM;

	
		
}

void load_edge_info() {
	FILE* f;
	f = fopen("edge_info.bin", "rb");
	int number_count = 0;
	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(edge_info_temp, sizeof(float), 103464, f);
	number_count += 103464;
	for (int i = 0; i < 103464; i++) {
		edge_info[i] = (int)edge_info_temp[i];
	}

}

void load_edge_num() {
	FILE* f;
	f = fopen("edge_num.bin", "rb");
	int number_count = 0;
	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(edge_num_temp, sizeof(float), TOTAL_TIME, f);
	number_count += TOTAL_TIME;
	for (int i = 0; i < TOTAL_TIME; i++) {
		edge_num[i] = (int)edge_num_temp[i];
	}
	
}

void load_node_num() {
	FILE* f;
	f = fopen("node_num.bin", "rb");
	int number_count = 0;
	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(node_num_temp, sizeof(float), TOTAL_TIME, f);
	number_count += TOTAL_TIME;
	for (int i = 0; i < TOTAL_TIME; i++) {
		node_num[i] = (int)node_num_temp[i];
	}
	
}

void load_renumbering_table() {
	FILE* f;
	f = fopen("neighborhood_ref_table.bin", "rb");
	int number_count = 0;
	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(neighborhood_renumbering_table_temp, sizeof(float), TOTAL_TIME * NUM_NODE, f);
	number_count += TOTAL_TIME * NUM_NODE;
	for (int i = 0; i < TOTAL_TIME; i++) {
		for (int j = 0; j < NUM_NODE; j++) {
			neighborhood_renumbering_table[i][j] = (int)neighborhood_renumbering_table_temp[i][j];
			
		}
		
	}

	

}


void load_reverse_renumbering_table() {
	FILE* f;
	f = fopen("neighborhood_reverse_ref_table.bin", "rb");
	int number_count = 0;
	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(neighborhood_reverse_renumbering_table_temp, sizeof(float), TOTAL_TIME * 578, f);
	number_count += TOTAL_TIME * 578;
	for (int i = 0; i < TOTAL_TIME; i++) {
		for (int j = 0; j < 578; j++) {
			neighborhood_reverse_renumbering_table[i][j] = (int)neighborhood_reverse_renumbering_table_temp[i][j];

		}

	}

}

void process_info(int time) {
	int total = 0;
	for (int i = 0; i < time ; i++) {
		total = total + 2 * edge_num[i];
	}
	for (int i = 0; i < edge_num[time]; i++) {
		edge_list[ 2 * i] = edge_info[total + 2 * i];
		edge_list[2 * i + 1] = edge_info[total + 2 * i + 1];

	}

	

	

}



void convert_weights_to_fix() { //the weight input layer 1 is 67. the weight output layer 1 is 122 (GNN); the weight input layer 2 is 122, the weight output of layer 2 is 122 (RNN�� 

	

	for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			gnn_update_weight_W_fixed[dim_in * OUTPUTDIM + dim_out] = (WT_TYPE)rnn_update_W[dim_in][dim_out];

		}
	}

	for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			gnn_update_weight_U_fixed[dim_in * OUTPUTDIM + dim_out] = (WT_TYPE)rnn_update_U[dim_in][dim_out];

		}
	}

	for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			gnn_reset_weight_W_fixed[dim_in * OUTPUTDIM + dim_out] = (WT_TYPE)rnn_reset_W[dim_in][dim_out];

		}
	}

	for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			gnn_reset_weight_U_fixed[dim_in * OUTPUTDIM + dim_out] = (WT_TYPE)rnn_reset_U[dim_in][dim_out];

		}
	}


	for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			gnn_htilda_weight_W_fixed[dim_in * OUTPUTDIM + dim_out] = (WT_TYPE)rnn_htilda_W[dim_in][dim_out];

		}
	}

	for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			gnn_htilda_weight_U_fixed[dim_in * OUTPUTDIM + dim_out] = (WT_TYPE)rnn_htilda_U[dim_in][dim_out];

		}
	}

	for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			gnn_output_weight_W_fixed[dim_in * OUTPUTDIM + dim_out] = (WT_TYPE)rnn_output_W[dim_in][dim_out];

		}
	}

	for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			gnn_output_weight_U_fixed[dim_in * OUTPUTDIM + dim_out] = (WT_TYPE)rnn_output_U[dim_in][dim_out];

		}
	}






	for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
		gnn_update_weight_bias_fixed[dim_out] = (WT_TYPE)rnn_update_bias[dim_out];

	}



	for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
		gnn_reset_weight_bias_fixed[dim_out] = (WT_TYPE)rnn_reset_bias[dim_out];

	}



	for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
		gnn_htilda_weight_bias_fixed[dim_out] = (WT_TYPE)rnn_htilda_bias[dim_out];

	}





	for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
		gnn_output_weight_bias_fixed[dim_out] = (WT_TYPE)rnn_output_bias[dim_out];

	}





	printf("Convert weights to fix finished ...\n");
}


void convert_embedding_to_fix() {

	for (int nd = 0; nd < NUM_NODE; nd++) {
		for (int dim = 0; dim < INPUTDIM; dim++) {
			cell_table_fixed[nd * INPUTDIM + dim] = (FM_TYPE)cell_state[nd][dim];
			node_embedding_fixed[nd * INPUTDIM + dim] = (FM_TYPE)node_embedding[nd][dim];
		}
	}
}

void convert_edge_embedding_to_fix(int time) {
	int total = 0;
	for (int i = 0; i < time; i++) {
		total = total +  edge_num[i];
	}
	
	for (int i = 0; i < edge_num[time]; i++) {
		for (int dim = 0; dim < INPUTDIM; dim++) {
			edge_embedding_fixed[i * INPUTDIM + dim] = (FM_TYPE)edge_embedding[total + i][dim];
		}
		

	}

	

}


/// 

//weights are stored in on-chip buffers
void load_weights( // this function check has passed.

	WT_TYPE gnn_update_weight_W_fixed[INPUTDIM * OUTPUTDIM],
	WT_TYPE gnn_update_weight_U_fixed[INPUTDIM * OUTPUTDIM],
	WT_TYPE gnn_reset_weight_W_fixed[INPUTDIM * OUTPUTDIM],
	WT_TYPE gnn_reset_weight_U_fixed[INPUTDIM * OUTPUTDIM],
	WT_TYPE gnn_htilda_weight_W_fixed[INPUTDIM * OUTPUTDIM],
	WT_TYPE gnn_htilda_weight_U_fixed[INPUTDIM * OUTPUTDIM],
	WT_TYPE gnn_output_weight_W_fixed[INPUTDIM * OUTPUTDIM],
	WT_TYPE gnn_output_weight_U_fixed[INPUTDIM * OUTPUTDIM],
	WT_TYPE gnn_update_weight_bias_fixed[OUTPUTDIM],
	WT_TYPE gnn_reset_weight_bias_fixed[OUTPUTDIM],
	WT_TYPE gnn_htilda_weight_bias_fixed[OUTPUTDIM],
	WT_TYPE gnn_output_weight_bias_fixed[OUTPUTDIM]
) {
#pragma HLS inline off


	for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			gnn_update_weight_W[dim_in][dim_out] = gnn_update_weight_W_fixed[dim_in * OUTPUTDIM + dim_out];
		}
	}

	for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			gnn_update_weight_U[dim_in][dim_out] = gnn_update_weight_U_fixed[dim_in * OUTPUTDIM + dim_out];
		}
	}

	for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			gnn_reset_weight_W[dim_in][dim_out] = gnn_reset_weight_W_fixed[dim_in * OUTPUTDIM + dim_out];
		}
	}

	for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			gnn_reset_weight_U[dim_in][dim_out] = gnn_reset_weight_U_fixed[dim_in * OUTPUTDIM + dim_out];
		}
	}

	for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			gnn_htilda_weight_W[dim_in][dim_out] = gnn_htilda_weight_W_fixed[dim_in * OUTPUTDIM + dim_out];
		}
	}

	for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			gnn_htilda_weight_U[dim_in][dim_out] = gnn_htilda_weight_U_fixed[dim_in * OUTPUTDIM + dim_out];
		}
	}


	for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			gnn_output_weight_W[dim_in][dim_out] = gnn_output_weight_W_fixed[dim_in * OUTPUTDIM + dim_out];
		}
	}

	for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			gnn_output_weight_U[dim_in][dim_out] = gnn_output_weight_U_fixed[dim_in * OUTPUTDIM + dim_out];
		}
	}





	for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
		gnn_update_weight_bias[dim_out] = gnn_update_weight_bias_fixed[dim_out];
	}

	for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
		gnn_reset_weight_bias[dim_out] = gnn_reset_weight_bias_fixed[dim_out];
	}

	for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
		gnn_htilda_weight_bias[dim_out] = gnn_htilda_weight_bias_fixed[dim_out];
	}

	for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
		gnn_output_weight_bias[dim_out] = gnn_output_weight_bias_fixed[dim_out];
		
	}



}


void load_graph_new(  //This function has been checked.
	FM_TYPE node_embedding_in[NUM_NODE * INPUTDIM],
	FM_TYPE cell_state_in[NUM_NODE * OUTPUTDIM],
	FM_TYPE edge_attr_in[SUB_GRAPH_EDGE_NUM * INPUTDIM],
	FM_TYPE node_embedding_table[SUB_GRAPH_NODE_NUM][INPUTDIM],
	FM_TYPE cell_table[SUB_GRAPH_NODE_NUM][INPUTDIM],
	FM_TYPE edge_embedding_table[SUB_GRAPH_EDGE_NUM][INPUTDIM],
	int num_of_nodes,
	int num_of_edges,
	int time,
	int neighborhood_inverse_ref_table_time[578])
{

#pragma HLS inline off

	//first adjust the node_embedding_table
	for (int i = 0; i < num_of_nodes; i++) {
		for (int dim = 0; dim < INPUTDIM; dim++) {
			
			int location_off_chip = neighborhood_inverse_ref_table_time[i];
			node_embedding_table[i][dim] = node_embedding_in[location_off_chip * INPUTDIM + dim];
		}
	}

	


	for (int i = 0; i < num_of_nodes; i++) {
		for (int dim = 0; dim < OUTPUTDIM; dim++) {
			int location_off_chip = neighborhood_inverse_ref_table_time[i];
			cell_table[i][dim] = cell_state_in[location_off_chip * OUTPUTDIM + dim];
		}
	}

	


	for (int e = 0; e < num_of_edges; e++) {
		for (int dim = 0; dim < INPUTDIM; dim++) {
			edge_embedding_table[e][dim] = edge_attr_in[ e * INPUTDIM + dim];
		}
	}

	

}


void prepare_degree_neighbor_table_new(int edge_list[SUB_GRAPH_EDGE_NUM * 2], 
	                                   int num_of_nodes, int num_of_edges, int time,
	                                   int neighborhood_ref_table[NUM_NODE])  // this function has been ckecked by C++ (but not python)
{
#pragma HLS inline off


	
	for (int n = 0; n < num_of_nodes * 3; n++) {
		degree_table[n] = 0;
	}

	for (int e = 0; e < num_of_edges; e++) {
		int u = neighborhood_ref_table[edge_list[e * 2]];     // source node id in the on-chip buffer.
		
		int v = neighborhood_ref_table[edge_list[e * 2 + 1]];   // target node id in the on-chip buffer.
		
		degree_table[u * 3] += 1;       //the format of degree table: [ total_neighbrhood_number  | start_idx_for_neigh_table  |  offset_idx_for_neigh_table  ]
	}


	for (int n = 1; n < num_of_nodes; n++) {
		
		degree_table[n * 3 + 1] = degree_table[(n - 1) * 3] * 2 + degree_table[(n - 1) * 3 + 1]; // *2: one is for the neighbor node id; the next one is for the corresponding edge id
	}// this for loop is used to define the start idx = past node start idx + past node degree *2 ( for neighbor_idx and edge_index)

	for (int e = 0; e < num_of_edges; e++) {
#pragma HLS inline off
		int u = neighborhood_ref_table[edge_list[e * 2]];     // source node id in the on-chip buffer.
		int v = neighborhood_ref_table[edge_list[e * 2 + 1]];   // target node id in the on-chip buffer.

		int total_neigh = degree_table[u * 3];
		int start_idx = degree_table[u * 3 + 1];
		int offset_idx = degree_table[u * 3 + 2];

		neighbor_table[start_idx + offset_idx] = v;
		//printf("the number of v is:%d\n", v);
		//printf("the number of e is:%d\n", e);
		//printf("the number of index is:%d\n", start_idx + offset_idx + 1);
		neighbor_table[start_idx + offset_idx + 1] = e;
		degree_table[u * 3 + 2] += 2;
	}

	
}


void message_passing_one_node_vec(hls::stream<FM_TYPE>& emb_val_1,
	hls::stream<FM_TYPE>& emb_val_2,
	hls::stream<FM_TYPE>& emb_val_3,
	hls::stream<FM_TYPE>& emb_val_4,
	hls::stream<FM_TYPE>& emb_val_5,
	hls::stream<FM_TYPE>& emb_val_6,
	hls::stream<FM_TYPE>& emb_val_7,
	hls::stream<FM_TYPE>& emb_val_8,
	hls::stream<FM_TYPE>& cell_state,
	int num_of_nodes, int num_of_edges,
	FM_TYPE message_tb[SUB_GRAPH_NODE_NUM][INPUTDIM],
	FM_TYPE edge_attr[SUB_GRAPH_EDGE_NUM][INPUTDIM],
	FM_TYPE cell_table[SUB_GRAPH_NODE_NUM][INPUTDIM],
	int time)  // This function has been checked.
{

#pragma HLS inline off
#pragma HLS array_partition variable=message_tb dim=2 complete
#pragma HLS array_partition variable=edge_attr dim=2 complete

	for (int nd = 0; nd < num_of_nodes; nd++) {
		int u = nd;  // u is the source node position on chip
		int total_neigh = degree_table[u * 3];  // we need a degree table 1. neighborhood number, start index in the neighbor table.
		int start_idx = degree_table[u * 3 + 1];

		FM_TYPE norm[160];
		FM_TYPE node_embedding_vec[INPUTDIM];
		FM_TYPE msg = 0; //this is used to store message for one edge
		for (int i = 0; i < 160; i++) {
			norm[i] = 0;
		}

		for (int i = 0; i < INPUTDIM; i++) {
			node_embedding_vec[i] = 0;
		}

		//total_neigh = 20;
		for (int i = 0; i < total_neigh; i++) {
			int v = neighbor_table[start_idx + i * 2];  // v is the neighborhood node index
			//printf("the number of v is :%d\n", v);
			int e = neighbor_table[start_idx + i * 2 + 1];// e is the edge index
			norm[i] = (1.0 / sqrt(degree_table[u * 3] + 1)) * (1.0 / sqrt(degree_table[v * 3] + 1));
			
			for (int dim = 0; dim < INPUTDIM; dim++) {
				//msg = norm[i] * (node_embedding_vec[dim] + edge_attr[dim]);  //for each node, we need to load an edge embedding table on chip
#pragma HLS pipeline II = 1
				msg = norm[i] * (edge_attr[e][dim] + message_tb[v][dim]); //?
				node_embedding_vec[dim] += msg;
			}
		}
		

		for (int dim = 0; dim < INPUTDIM; dim++) {
#pragma HLS pipeline II = 1
			FM_TYPE temp = (1.0 / (degree_table[u * 3] + 1.0));
			node_embedding_vec[dim] += message_tb[u][dim] * temp;
			//printf("the value of node_embed is:%.6f\n", (float)node_embedding_vec[dim]);
			
			emb_val_1.write(node_embedding_vec[dim]);
			emb_val_2.write(node_embedding_vec[dim]);
			emb_val_3.write(node_embedding_vec[dim]);
			emb_val_4.write(node_embedding_vec[dim]);
			emb_val_5.write(node_embedding_vec[dim]);
			emb_val_6.write(node_embedding_vec[dim]);
			emb_val_7.write(node_embedding_vec[dim]);
			emb_val_8.write(node_embedding_vec[dim]);
			//i++;
			//printf("the number of emb_val is:%.3f\n", (float)emb_val.read());
			//printf("the total number of value in the stream is:%d\n", i);

		}

		for (int dim = 0; dim < INPUTDIM; dim++) {
			cell_state.write(cell_table[u][dim]);
		}
		
		
		
	}

}

void CONV_GNN_sigmoid(hls::stream<FM_TYPE>& emb_input_W, 
	                 hls::stream<FM_TYPE>& emb_input_U, 
	                 hls::stream<FM_TYPE>& emb_output, 
	                 WT_TYPE convs_weight_W[INPUTDIM][OUTPUTDIM], 
	                 WT_TYPE convs_weight_U[INPUTDIM][OUTPUTDIM], 
	                 WT_TYPE convs_bias[OUTPUTDIM],
	                 int num_of_nodes, int num_of_edges)
{ // This function has been checked.
#pragma HLS array_partition variable=convs_weight_W dim=1 complete
#pragma HLS array_partition variable=convs_weight_U dim=1 complete
	//#pragma HLS array_partition variable=convs_bias dim=1 complete

	
	//linear, x = self.linear(x)
	for (int nd = 0; nd < num_of_nodes; nd++) {
		FM_TYPE node_emb_input_W[INPUTDIM];
		FM_TYPE node_emb_input_U[INPUTDIM];
		FM_TYPE node_emb_output[OUTPUTDIM];
		//#pragma HLS array_partition variable=node_emb_output dim=1 complete
#pragma HLS array_partition variable=node_emb_input_W dim=1 complete
#pragma HLS array_partition variable=node_emb_input_U dim=1 complete

		for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {

			node_emb_input_W[dim_in] = emb_input_W.read();
			
			node_emb_input_U[dim_in] = emb_input_U.read();
		}
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
#pragma HLS PIPELINE II = 1

			
			node_emb_output[dim_out] = convs_bias[dim_out];
			FM_TYPE tmp = 0;
			for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {

				tmp += node_emb_input_W[dim_in] * convs_weight_W[dim_in][dim_out] + node_emb_input_U[dim_in] * convs_weight_U[dim_in][dim_out];
			}
			node_emb_output[dim_out] += tmp;
			node_emb_output[dim_out] = 1.0 / (1.0 + expf(-node_emb_output[dim_out]));
			//printf("the value of output is:%.6f\n", (float)node_emb_output[dim_out]);
			emb_output.write(node_emb_output[dim_out]);
			
		}

	}
}


void CONV_GNN_tanh(hls::stream<FM_TYPE>& emb_input_W, 
	               hls::stream<FM_TYPE>& emb_input_U, 
	               hls::stream<FM_TYPE>& emb_output, 
	               WT_TYPE convs_weight_W[INPUTDIM][OUTPUTDIM], 
	               WT_TYPE convs_weight_U[INPUTDIM][OUTPUTDIM], 
	               WT_TYPE convs_bias[OUTPUTDIM], 
	               int num_of_nodes, int num_of_edges)
{
#pragma HLS array_partition variable=convs_weight_W dim=1 complete
#pragma HLS array_partition variable=convs_weight_U dim=1 complete
	//#pragma HLS array_partition variable=convs_bias dim=1 complete
	
	
	//linear, x = self.linear(x)
	for (int nd = 0; nd < num_of_nodes; nd++) {
		FM_TYPE node_emb_input_W[INPUTDIM];
		FM_TYPE node_emb_input_U[INPUTDIM];
		FM_TYPE node_emb_output[OUTPUTDIM];
		//#pragma HLS array_partition variable=node_emb_output dim=1 complete
#pragma HLS array_partition variable=node_emb_input_W dim=1 complete
#pragma HLS array_partition variable=node_emb_input_U dim=1 complete

		for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {

			node_emb_input_W[dim_in] = emb_input_W.read();
			
			node_emb_input_U[dim_in] = emb_input_U.read();
		}
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
#pragma HLS PIPELINE II = 1

			//node_emb_output[dim_out] = convs_bias[dim_out];
			node_emb_output[dim_out] = convs_bias[dim_out];
			FM_TYPE tmp = 0;
			for (int dim_in = 0; dim_in < INPUTDIM; dim_in++) {

				tmp += node_emb_input_W[dim_in] * convs_weight_W[dim_in][dim_out] + node_emb_input_U[dim_in] * convs_weight_U[dim_in][dim_out];
			}
			node_emb_output[dim_out] += tmp;
			

			node_emb_output[dim_out] = (expf(node_emb_output[dim_out]) - expf(-node_emb_output[dim_out])) / (expf(node_emb_output[dim_out]) + expf(-node_emb_output[dim_out]));
			//printf("the value of output is:%.6f\n", (float)node_emb_output[dim_out]);
			emb_output.write(node_emb_output[dim_out]);
		}

	}
}



void LSTM_multi_1(hls::stream<FM_TYPE>& update_output, 
	              hls::stream<FM_TYPE>& cell_input,
	              hls::stream<FM_TYPE>& reset_output, 
	              hls::stream<FM_TYPE>& htilda_output, 
	              hls::stream<FM_TYPE>& cell_output_temp, 
	              hls::stream<FM_TYPE>& cell_output, 
	              int num_of_nodes, int num_of_edges) {  // This function has been checked.

	for (int nd = 0; nd < num_of_nodes; nd++) {
		FM_TYPE update_output_vector[OUTPUTDIM];
		FM_TYPE cell_input_vector[OUTPUTDIM];
		FM_TYPE reset_output_vector[OUTPUTDIM];
		FM_TYPE htilda_output_vector[OUTPUTDIM];
		FM_TYPE cell_output_vector[OUTPUTDIM];
#pragma HLS array_partition variable=update_output_vector dim=1 complete
#pragma HLS array_partition variable=cell_input_vector dim=1 complete
#pragma HLS array_partition variable=reset_output_vector dim=1 complete
#pragma HLS array_partition variable=htilda_output_vector dim=1 complete
#pragma HLS array_partition variable=cell_output_vector dim=1 complete


		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			update_output_vector[dim_out] = update_output.read();
			cell_input_vector[dim_out] = cell_input.read();
			reset_output_vector[dim_out] = reset_output.read();
			htilda_output_vector[dim_out] = htilda_output.read();
		}


		GRU_multi_1_loop:	for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
	#pragma HLS pipeline II = 1
			cell_output_vector[dim_out] = update_output_vector[dim_out] * cell_input_vector[dim_out] + reset_output_vector[dim_out] * htilda_output_vector[dim_out];
			//htilda_output.write(htilda_output_vector[dim_out]);
		}

		GRU_multi_1_wrtie_loop:	for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			cell_output_temp.write(cell_output_vector[dim_out]);
			cell_output.write(cell_output_vector[dim_out]);
			
		}
	}

}



void LSTM_multi_2(hls::stream<FM_TYPE>& output_output,
	             hls::stream<FM_TYPE>& cell_input, 
	             hls::stream<FM_TYPE>& rnn_output, 
	            int num_of_nodes, int num_of_edges) {  // This function has been checked.

	for (int nd = 0; nd < num_of_nodes; nd++) {
		FM_TYPE output_output_vector[OUTPUTDIM];
		FM_TYPE cell_input_vector[OUTPUTDIM];
		FM_TYPE rnn_output_vector[OUTPUTDIM];

#pragma HLS array_partition variable=output_output_vector dim=1 complete
#pragma HLS array_partition variable=cell_input_vector dim=1 complete
#pragma HLS array_partition variable=rnn_output_vector dim=1 complete




		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			output_output_vector[dim_out] = output_output.read();
			cell_input_vector[dim_out] = cell_input.read();

		}


		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
#pragma HLS pipeline II = 1
			rnn_output_vector[dim_out] = (expf(cell_input_vector[dim_out]) - expf(-cell_input_vector[dim_out])) / (expf(cell_input_vector[dim_out]) + expf(-cell_input_vector[dim_out]));
			rnn_output_vector[dim_out] = rnn_output_vector[dim_out] * output_output_vector[dim_out];

		}

		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			rnn_output.write(rnn_output_vector[dim_out]);
			//printf("the value is:%.6f\n", (float)rnn_output_vector[dim_out]);
		}
	}

}


void write_back(hls::stream<FM_TYPE>& rnn_output,
	           hls::stream<FM_TYPE>& cell_output, 
	           int num_of_nodes, int time, WT_TYPE node_embedding_fixed_output[NUM_NODE * INPUTDIM],
	           WT_TYPE cell_state_fixed_output[NUM_NODE * OUTPUTDIM],
			   int neighborhood_inverse_ref_table_time[578]
	           
	           ) {
#pragma HLS inline off	
	for (int nd = 0; nd < num_of_nodes; nd++) {
		FM_TYPE rnn_output_vector[OUTPUTDIM];
		FM_TYPE cell_output_vector[OUTPUTDIM];
		
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			rnn_output_vector[dim_out] = rnn_output.read();
			cell_output_vector[dim_out] = cell_output.read();
			
			node_embedding_table_temp[nd][dim_out] = rnn_output_vector[dim_out];
			cell_state_table_temp[nd][dim_out] = cell_output_vector[dim_out];
		}

	}
	for (int i = 0; i < num_of_nodes; i++) {
		for (int dim_out = 0; dim_out < OUTPUTDIM; dim_out++) {
			node_embedding_fixed_output[neighborhood_inverse_ref_table_time[i] * OUTPUTDIM + dim_out] = node_embedding_table_temp[i][dim_out];
			cell_state_fixed_output[neighborhood_inverse_ref_table_time[i] * OUTPUTDIM + dim_out] = cell_state_table_temp[i][dim_out];
		}

	}


}






void dataflow_region(
	WT_TYPE node_embedding_output[NUM_NODE * INPUTDIM],
	WT_TYPE cell_state[NUM_NODE * OUTPUTDIM],
	FM_TYPE node_embedding_table[SUB_GRAPH_NODE_NUM][INPUTDIM],
	FM_TYPE cell_table[SUB_GRAPH_NODE_NUM][INPUTDIM],
	FM_TYPE edge_embedding_table[SUB_GRAPH_NODE_NUM][INPUTDIM],
	int neighborhood_inverse_ref_table_time[578],

	WT_TYPE gnn_update_weight_W[INPUTDIM][OUTPUTDIM],
	WT_TYPE gnn_update_weight_U[INPUTDIM][OUTPUTDIM],
	WT_TYPE gnn_reset_weight_W[INPUTDIM][OUTPUTDIM],
	WT_TYPE gnn_reset_weight_U[INPUTDIM][OUTPUTDIM],
	WT_TYPE gnn_htilda_weight_W[INPUTDIM][OUTPUTDIM],
	WT_TYPE gnn_htilda_weight_U[INPUTDIM][OUTPUTDIM],
	WT_TYPE gnn_output_weight_W[INPUTDIM][OUTPUTDIM],
	WT_TYPE gnn_output_weight_U[INPUTDIM][OUTPUTDIM],
	WT_TYPE gnn_update_weight_bias[OUTPUTDIM],
	WT_TYPE gnn_reset_weight_bias[OUTPUTDIM],
	WT_TYPE gnn_htilda_weight_bias[OUTPUTDIM],
	WT_TYPE gnn_output_weight_bias[OUTPUTDIM],



	int node_number_time,
	int edge_number_time,
	int time

	
)
{
	
	hls::stream<FM_TYPE> message_passing_fifo_1;
	hls::stream<FM_TYPE> message_passing_fifo_2;
	hls::stream<FM_TYPE> message_passing_fifo_3;
	hls::stream<FM_TYPE> message_passing_fifo_4;
	hls::stream<FM_TYPE> message_passing_fifo_5;
	hls::stream<FM_TYPE> message_passing_fifo_6;
	hls::stream<FM_TYPE> message_passing_fifo_7;
	hls::stream<FM_TYPE> message_passing_fifo_8;

	hls::stream<FM_TYPE> update_output_fifo;
	//hls::stream<FM_TYPE> update_output_fifo_U;
	hls::stream<FM_TYPE> reset_output_fifo;

	hls::stream<FM_TYPE> output_output_fifo;
	//hls::stream<FM_TYPE> reset_output_fifo_U;
	hls::stream<FM_TYPE> htilda_output_temp_fifo;
	//hls::stream<FM_TYPE> htilda_output_temp_fifo_U;

	hls::stream<FM_TYPE> htilda_output_fifo;

	hls::stream<FM_TYPE> cell_state_fifo;

	hls::stream<FM_TYPE> cell_output_temp_fifo;
	hls::stream<FM_TYPE> cell_output_fifo;
	//hls::stream<FM_TYPE> htilda_output_fifo_U;

	hls::stream<FM_TYPE> rnn_output_fifo;
#pragma HLS STREAM variable=message_passing_fifo_1 depth=400
#pragma HLS STREAM variable=message_passing_fifo_2 depth=400
#pragma HLS STREAM variable=message_passing_fifo_3 depth=400
#pragma HLS STREAM variable=message_passing_fifo_4 depth=400
#pragma HLS STREAM variable=message_passing_fifo_5 depth=400
#pragma HLS STREAM variable=message_passing_fifo_6 depth=400
#pragma HLS STREAM variable=message_passing_fifo_7 depth=400
#pragma HLS STREAM variable=message_passing_fifo_8 depth=400


#pragma HLS STREAM variable=update_output_fifo depth=400
	//#pragma HLS STREAM variable=update_output_fifo_U depth=400
#pragma HLS STREAM variable=reset_output_fifo depth=400

#pragma HLS STREAM variable=output_output_fifo depth=400
//#pragma HLS STREAM variable=reset_output_fifo_U depth=400
//#pragma HLS STREAM variable=htilda_output_temp_fifo depth=400
#pragma HLS STREAM variable=htilda_output_fifo depth=400

#pragma HLS STREAM variable=cell_state_fifo depth=400
#pragma HLS STREAM variable=cell_output_temp_fifo depth=400
#pragma HLS STREAM variable=cell_output_fifo depth=400

#pragma HLS STREAM variable=rnn_output_fifo depth=400
//#pragma HLS STREAM variable=htilda_output_fifo_U depth=400



#pragma HLS dataflow
	

	message_passing_one_node_vec(message_passing_fifo_1, message_passing_fifo_2, message_passing_fifo_3, message_passing_fifo_4, message_passing_fifo_5, message_passing_fifo_6, message_passing_fifo_7, message_passing_fifo_8, cell_state_fifo, node_number_time, edge_number_time, node_embedding_table, edge_embedding_table, cell_table, time);
	
	CONV_GNN_sigmoid(message_passing_fifo_1, message_passing_fifo_2, update_output_fifo, gnn_update_weight_W, gnn_update_weight_U, gnn_update_weight_bias, node_number_time, edge_number_time);
	CONV_GNN_sigmoid(message_passing_fifo_3, message_passing_fifo_4, reset_output_fifo, gnn_reset_weight_W, gnn_reset_weight_U, gnn_reset_weight_bias, node_number_time, edge_number_time);
	CONV_GNN_sigmoid(message_passing_fifo_5, message_passing_fifo_6, output_output_fifo, gnn_output_weight_W, gnn_output_weight_U, gnn_output_weight_bias, node_number_time, edge_number_time);
	CONV_GNN_tanh(message_passing_fifo_7, message_passing_fifo_8, htilda_output_fifo, gnn_htilda_weight_W, gnn_htilda_weight_U, gnn_htilda_weight_bias, node_number_time, edge_number_time);
	LSTM_multi_1(update_output_fifo, cell_state_fifo, reset_output_fifo, htilda_output_fifo, cell_output_temp_fifo, cell_output_fifo, node_number_time, edge_number_time);
	LSTM_multi_2(output_output_fifo, cell_output_temp_fifo, rnn_output_fifo, node_number_time, edge_number_time);
	write_back(rnn_output_fifo, cell_output_fifo, node_number_time, time, node_embedding_output, cell_state, neighborhood_inverse_ref_table_time);
}







void DGNN_Booster_convLSTM(
	int edge_list_in[SUB_GRAPH_EDGE_NUM * 2],
	int node_number_time[192],  // need to change from 137 to 192 for bitcoin dataset
	int edge_number_time[192],


	WT_TYPE gnn_update_weight_W_fixed[INPUTDIM * OUTPUTDIM],
	WT_TYPE gnn_update_weight_U_fixed[INPUTDIM * OUTPUTDIM],
	WT_TYPE gnn_update_weight_bias_fixed[OUTPUTDIM],


	WT_TYPE gnn_reset_weight_W_fixed[INPUTDIM * OUTPUTDIM],
	WT_TYPE gnn_reset_weight_U_fixed[INPUTDIM * OUTPUTDIM],
	WT_TYPE gnn_reset_weight_bias_fixed[OUTPUTDIM],


	WT_TYPE gnn_htilda_weight_W_fixed[INPUTDIM * OUTPUTDIM],
	WT_TYPE gnn_htilda_weight_U_fixed[INPUTDIM * OUTPUTDIM],
	WT_TYPE gnn_htilda_weight_bias_fixed[OUTPUTDIM],

	WT_TYPE gnn_output_weight_W_fixed[INPUTDIM * OUTPUTDIM],
	WT_TYPE gnn_output_weight_U_fixed[INPUTDIM * OUTPUTDIM],
	WT_TYPE gnn_output_weight_bias_fixed[OUTPUTDIM],


	FM_TYPE node_embedding_fixed[NUM_NODE * INPUTDIM],  // need to change to NUM-NODE
	FM_TYPE cell_table_fixed[NUM_NODE * INPUTDIM],
	FM_TYPE edge_attr_in_fixed[SUB_GRAPH_EDGE_NUM * INPUTDIM],


	int neighborhood_ref_table_time[NUM_NODE],     //need to change to NUM_NODE
	int neighborhood_inverse_ref_table_time[578],  // need to change from 501 -> 578
	int time[100]

)
{
#pragma HLS INTERFACE s_axilite port=return
	
#pragma HLS INTERFACE m_axi depth=100000 port=edge_list_in offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=node_number_time offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=edge_number_time offset=slave bundle=INPUT


#pragma HLS INTERFACE m_axi depth=100000 port=gnn_update_weight_W_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_update_weight_U_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_update_weight_bias_fixed offset=slave bundle=INPUT


#pragma HLS INTERFACE m_axi depth=100000 port=gnn_reset_weight_W_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_reset_weight_U_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_reset_weight_bias_fixed offset=slave bundle=INPUT

#pragma HLS INTERFACE m_axi depth=100000 port=gnn_htilda_weight_W_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_htilda_weight_U_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_htilda_weight_bias_fixed offset=slave bundle=INPUT

#pragma HLS INTERFACE m_axi depth=100000 port=gnn_output_weight_W_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_output_weight_U_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_output_weight_bias_fixed offset=slave bundle=INPUT

#pragma HLS INTERFACE m_axi depth=100000 port=node_embedding_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=cell_table_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=edge_attr_in_fixed offset=slave bundle=INPUT


#pragma HLS INTERFACE m_axi depth=100000 port=neighborhood_ref_table_time offset=slave bundle=INPUT2
#pragma HLS INTERFACE m_axi depth=100000 port=neighborhood_inverse_ref_table_time offset=slave bundle=INPUT2
#pragma HLS INTERFACE m_axi depth=100000 port=time offset=slave bundle=INPUT2


#pragma HLS bind_storage variable=gnn_update_weight_W impl = bram
#pragma HLS bind_storage variable=gnn_update_weight_U impl = bram
#pragma HLS bind_storage variable=gnn_update_weight_bias impl = bram
#pragma HLS bind_storage variable=gnn_reset_weight_W impl = bram
#pragma HLS bind_storage variable=gnn_reset_weight_U impl = bram
#pragma HLS bind_storage variable=gnn_reset_weight_bias impl = bram
#pragma HLS bind_storage variable=gnn_htilda_weight_W impl = bram
#pragma HLS bind_storage variable=gnn_htilda_weight_U impl = bram
#pragma HLS bind_storage variable=gnn_htilda_weight_bias impl = bram
#pragma HLS bind_storage variable=gnn_output_weight_W impl = bram
#pragma HLS bind_storage variable=gnn_output_weight_U impl = bram
#pragma HLS bind_storage variable=gnn_output_weight_bias impl = bram

#pragma HLS bind_storage variable=node_embedding_table impl = bram
#pragma HLS bind_storage variable=edge_embedding_table impl = bram
#pragma HLS bind_storage variable=node_embedding_table_temp impl = bram
#pragma HLS bind_storage variable=cell_state_table_temp impl = bram

	if (time[0] == 0) {
		load_weights(gnn_update_weight_W_fixed, gnn_update_weight_U_fixed, gnn_reset_weight_W_fixed, gnn_reset_weight_U_fixed, gnn_htilda_weight_W_fixed, gnn_htilda_weight_U_fixed, gnn_output_weight_W_fixed, gnn_output_weight_U_fixed, gnn_update_weight_bias_fixed, gnn_reset_weight_bias_fixed, gnn_htilda_weight_bias_fixed, gnn_output_weight_bias_fixed);

	}
	
	int node_number = node_number_time[time[0]];
	int edge_number = edge_number_time[time[0]];
	load_graph_new(node_embedding_fixed, cell_table_fixed, edge_attr_in_fixed, node_embedding_table, cell_table, edge_embedding_table, node_number, edge_number, time[0],  neighborhood_inverse_ref_table_time);
	
	prepare_degree_neighbor_table_new(edge_list_in, node_number, edge_number, time[0], neighborhood_ref_table_time);
	//compute_workflow(edge_list_in, edge_list, edge_attr_in_fixed, cell_table_fixed, node_embedding_fixed, gnn_update_weight_W, gnn_update_weight_U, gnn_update_weight_bias, gnn_reset_weight_W, gnn_reset_weight_U, gnn_reset_weight_bias, gnn_htilda_weight_W, gnn_htilda_weight_U, gnn_htilda_weight_bias, gnn_output_weight_W, gnn_output_weight_U, gnn_output_weight_bias, node_number_time, edge_number_time, valid_list, store_list, store_list_size);
	dataflow_region(node_embedding_fixed, cell_table_fixed, node_embedding_table, cell_table, edge_embedding_table, neighborhood_inverse_ref_table_time, gnn_update_weight_W, gnn_update_weight_U, gnn_reset_weight_W, gnn_reset_weight_U, gnn_htilda_weight_W, gnn_htilda_weight_U, gnn_output_weight_W, gnn_output_weight_U, gnn_update_weight_bias, gnn_reset_weight_bias, gnn_htilda_weight_bias, gnn_output_weight_bias, node_number, edge_number, time[0]);
	
}



int main() {

	load_weights_convLSTM();
	load_node_embedding();
	load_edge_embedding();
	load_edge_num();
	load_node_num();
	load_edge_info();
	load_renumbering_table();
	load_reverse_renumbering_table();

	//initialize cell_state to 0
	for (int nd = 0; nd < NUM_NODE; nd++) {
		for (int dim = 0; dim < INPUTDIM; dim++) {
			cell_state[nd][dim] = 0;
		}
	}
	
	convert_weights_to_fix();
	convert_embedding_to_fix();

	//process_info(0);//load edge list
	//convert_edge_embedding_to_fix(0);

	int total_time = 2;
	int time_array[100];
	for (int time = 0; time < total_time; time++) {
		//printf("the time now is:%d\n", time);
		process_info(time);//load edge list
		convert_edge_embedding_to_fix(time);
		time_array[0] = time;
		DGNN_Booster_convLSTM(edge_list, node_num, edge_num, gnn_update_weight_W_fixed, gnn_update_weight_U_fixed, gnn_update_weight_bias_fixed, gnn_reset_weight_W_fixed, gnn_reset_weight_U_fixed, gnn_reset_weight_bias_fixed, gnn_htilda_weight_W_fixed, gnn_htilda_weight_U_fixed, gnn_htilda_weight_bias_fixed, gnn_output_weight_W_fixed, gnn_output_weight_U_fixed, gnn_output_weight_bias_fixed, node_embedding_fixed, cell_table_fixed, edge_embedding_fixed, neighborhood_renumbering_table[time], neighborhood_reverse_renumbering_table[time], time_array);
	}
	
	
	for (int nd = 0; nd < 5; nd++) {
		for (int dim = 0; dim < 67; dim++) {
			printf("the value is:%.6f\n", (float)cell_table_fixed[nd * 67 + dim]);
			printf("the value is:%.6f\n", (float)node_embedding_fixed[nd * 67 + dim]);
		}
	}
	
	
}