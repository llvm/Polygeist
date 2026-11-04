#include <stdlib.h>
#include <stdio.h>

//#include "dcl.h"
#include <vector> 
#include <algorithm>
#include <string.h>
#include<iostream>

#include <ap_fixed.h>
#include "hls_stream.h"

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

//global weights for evolveGCN. It contains GNN weighrs, RNN weights and final layer classifier weights
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

using namespace std;

typedef ap_fixed<32, 10> FM_TYPE;
typedef ap_fixed<32, 10> WT_TYPE;

// GCN and GRU weights for layer1
float gcn_node_weight_layer1[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];    // [67,122]
WT_TYPE gcn_node_weight_layer1_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM];

WT_TYPE gcn_node_weight_layer1_buffer_1[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];
WT_TYPE gcn_node_weight_layer1_buffer_2[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];


float gru_update_layer1_W[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
float gru_update_layer1_U[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
float gru_update_layer1_bias[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];  //[67,122]


WT_TYPE gru_update_layer1_W_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM];
WT_TYPE gru_update_layer1_W_buffer_1[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
WT_TYPE gru_update_layer1_W_buffer_2[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];

WT_TYPE gru_update_layer1_bias_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM];

WT_TYPE gru_update_layer1_bias_buffer_1[WEIGHT_LAYER1_OUTPUTDIM];
WT_TYPE gru_update_layer1_bias_buffer_2[WEIGHT_LAYER1_OUTPUTDIM];

float gru_reset_layer1_W[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
float gru_reset_layer1_U[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
float gru_reset_layer1_bias[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];

float gru_htilda_layer1_W[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
float gru_htilda_layer1_U[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
float gru_htilda_layer1_bias[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];

float topk_layer1_W[WEIGHT_LAYER1_INPUTDIM][1];

// GCN and GRU weights for layer2
float gcn_node_weight_layer2[WEIGHT_LAYER2_INPUTDIM][WEIGHT_LAYER2_OUTPUTDIM]; //[122,122]



float gru_update_layer2_W[WEIGHT_LAYER2_INPUTDIM][WEIGHT_LAYER2_INPUTDIM];   //[122,122]
//WT_TYPE gru_update_layer2_W_fixed[WEIGHT_LAYER2_INPUTDIM * WEIGHT_LAYER2_OUTPUTDIM];  //for the third MLP part


float gru_update_layer2_U[WEIGHT_LAYER2_INPUTDIM][WEIGHT_LAYER2_INPUTDIM];
float gru_update_layer2_bias[WEIGHT_LAYER2_INPUTDIM][WEIGHT_LAYER2_OUTPUTDIM];




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
//float node_embedding_time[TIME_STEP][NUM_NODE][WEIGHT_LAYER1_INPUTDIM];

float num_edges[TOTAL_TIME + 1];
int num_edges_time[TIME_STEP];
int num_nodes_time[TIME_STEP];

int node_out_degree[NUM_NODE];
int node_out_degree_time[TIME_STEP][NUM_NODE];


float node_embedding_subgraph[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_INPUTDIM];
float node_embedding_subgraph_time[TIME_STEP][SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_INPUTDIM];
float edge_embedding_subgraph[SUB_GRAPH_EDGE_NUM][WEIGHT_LAYER1_INPUTDIM];
float edge_embedding_subgraph_time[TIME_STEP][SUB_GRAPH_EDGE_NUM][WEIGHT_LAYER1_INPUTDIM];

int degree_table[SUB_GRAPH_NODE_NUM * 3];
int neighbor_table[SUB_GRAPH_NODE_NUM * MAX_DEGREE * 2];
int neighborhood_ref_table[NUM_NODE];
int neighborhood_ref_table_time[TIME_STEP][NUM_NODE];

std::vector <int> node_record_vector;


FM_TYPE node_embedding_in[TIME_STEP * SUB_GRAPH_NODE_NUM * WEIGHT_LAYER1_INPUTDIM];
FM_TYPE edge_embedding_in[TIME_STEP * SUB_GRAPH_EDGE_NUM * WEIGHT_LAYER1_INPUTDIM];
FM_TYPE output_fixed[SUB_GRAPH_NODE_NUM * WEIGHT_LAYER1_OUTPUTDIM];

FM_TYPE node_embedding_table_1[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_INPUTDIM];
FM_TYPE edge_embedding_table_1[SUB_GRAPH_EDGE_NUM][WEIGHT_LAYER1_INPUTDIM];
FM_TYPE node_embedding_table_2[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_INPUTDIM];
FM_TYPE edge_embedding_table_2[SUB_GRAPH_EDGE_NUM][WEIGHT_LAYER1_INPUTDIM];

FM_TYPE node_embedding_table_time[TIME_STEP][SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_INPUTDIM];
FM_TYPE edge_embedding_table_time[TIME_STEP][SUB_GRAPH_EDGE_NUM][WEIGHT_LAYER1_INPUTDIM];

FM_TYPE output_table[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_OUTPUTDIM];


//int edge_list[SUB_GRAPH_EDGE_NUM * 2];

int edge_list_time[TIME_STEP][SUB_GRAPH_EDGE_NUM * 2];



FM_TYPE weight_buffer_1[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];
FM_TYPE weight_buffer_2[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];

FM_TYPE emb_buffer_1[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_INPUTDIM];
FM_TYPE emb_buffer_2[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_INPUTDIM];


FM_TYPE rnn_update_weight_W_buffer[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
FM_TYPE rnn_update_weight_U_buffer[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
FM_TYPE rnn_update_bias_buffer[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];
FM_TYPE rnn_reset_weight_W_buffer[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
FM_TYPE rnn_reset_weight_U_buffer[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
FM_TYPE rnn_reset_bias_buffer[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];
FM_TYPE rnn_htilda_weight_W_buffer[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
FM_TYPE rnn_htilda_weight_U_buffer[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM];
FM_TYPE rnn_htilda_bias_buffer[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];


FM_TYPE rnn_update_weight_W_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM];
FM_TYPE rnn_update_weight_U_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM];
FM_TYPE rnn_update_bias_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM];
FM_TYPE rnn_reset_weight_W_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM];
FM_TYPE rnn_reset_weight_U_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM];
FM_TYPE rnn_reset_bias_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM];
FM_TYPE rnn_htilda_weight_W_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM];
FM_TYPE rnn_htilda_weight_U_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM];
FM_TYPE rnn_htilda_bias_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM];



float node_embedding[NUM_NODE][WEIGHT_LAYER1_INPUTDIM];
float node_embedding_time[TIME_STEP][SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_INPUTDIM];
float edge_embedding[NUM_EDGE][WEIGHT_LAYER1_INPUTDIM];
float edge_embedding_time[TIME_STEP][SUB_GRAPH_EDGE_NUM][WEIGHT_LAYER1_INPUTDIM];
float cell_state[NUM_NODE][WEIGHT_LAYER1_INPUTDIM];
float edge_info_temp[103464];
int edge_info[103464];
float edge_num_temp[TOTAL_TIME];
int edge_num[TOTAL_TIME];
int edge_num_time[TIME_STEP];
float node_num_temp[TOTAL_TIME];
int node_num[TOTAL_TIME];
int node_num_time[TIME_STEP];
int edge_list[2 * SUB_GRAPH_EDGE_NUM];

float neighborhood_renumbering_table_temp[TOTAL_TIME][NUM_NODE];
int neighborhood_renumbering_table[TOTAL_TIME][NUM_NODE];
int neighborhood_renumbering_table_time[TIME_STEP][NUM_NODE];
float neighborhood_reverse_renumbering_table_temp[TOTAL_TIME][501];
int neighborhood_reverse_renumbering_table[TOTAL_TIME][501];
int neighborhood_reverse_renumbering_table_time[TIME_STEP][SUB_GRAPH_NODE_NUM];



void load_weights_float() { // This function has been checked.
	printf("Loading weights for evolveGCN (RNN + GCN)...\n");
	// printf("execute here now!\n");
	// FILE *f_2 = fopen("../DGNN_parameters/EvolveGCN_weights/EvolveGCN.bin", "r"); // Open in read mode as an example
	// if (f_2 == NULL) {
	// 	perror("Error opening file");
	// 	// Handle the error, e.g., exit the program or attempt to open another file
	// }

	FILE* f;
	//printf("execute here!");
	f = fopen("EvolveGCN.bin", "rb");
	//printf("execute here!");
	int number_count = 0;
	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(gcn_node_weight_layer1, sizeof(float), WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM, f);
	number_count += WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM;

	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(gru_update_layer1_W, sizeof(float), WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM, f);
	number_count += WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM;

	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(gru_update_layer1_U, sizeof(float), WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM, f);
	number_count += WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM;

	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(gru_update_layer1_bias, sizeof(float), WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM, f);
	number_count += WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM;

	


	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(gru_reset_layer1_W, sizeof(float), WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM, f);
	number_count += WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM;

	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(gru_reset_layer1_U, sizeof(float), WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM, f);
	number_count += WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM;

	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(gru_reset_layer1_bias, sizeof(float), WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM, f);
	number_count += WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM;


	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(gru_htilda_layer1_W, sizeof(float), WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM, f);
	number_count += WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM;

	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(gru_htilda_layer1_U, sizeof(float), WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM, f);
	number_count += WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM;

	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(gru_htilda_layer1_bias, sizeof(float), WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM, f);
	number_count += WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM;

	
}

void load_node_embedding() {
	FILE* f;
	f = fopen("node_embedding.bin", "rb");
	int number_count = 0;
	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(node_embedding, sizeof(float), NUM_NODE * WEIGHT_LAYER1_INPUTDIM, f);
	number_count += NUM_NODE * WEIGHT_LAYER1_INPUTDIM;


}


void load_edge_embedding() {
	FILE* f;
	f = fopen("edge_embedding.bin", "rb");
	int number_count = 0;
	fseek(f, number_count * sizeof(float), SEEK_SET);
	fread(edge_embedding, sizeof(float), NUM_EDGE * WEIGHT_LAYER1_INPUTDIM, f);
	number_count += NUM_EDGE * WEIGHT_LAYER1_INPUTDIM;



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
	fread(neighborhood_reverse_renumbering_table_temp, sizeof(float), TOTAL_TIME * 501, f);
	number_count += TOTAL_TIME * 501;
	for (int i = 0; i < TOTAL_TIME; i++) {
		for (int j = 0; j < 501; j++) {
			neighborhood_reverse_renumbering_table[i][j] = (int)neighborhood_reverse_renumbering_table_temp[i][j];

		}

	}

}


void process_edge_list_info(int time, int time_step) {
	int total = 0;
	int total_internel = 0;
	for (int i = 0; i < time; i++) {
		total = total + 2 * edge_num[i];
	}
	for (int t = 0; t < time_step; t++) {
		for (int i = 0; i < edge_num[time + t]; i++) {
			edge_list_time[t][2 * i] = edge_info[total + total_internel + 2 * i];
			edge_list_time[t][2 * i + 1] = edge_info[total + total_internel + 2 * i + 1];

		}
		total_internel = total_internel + 2 * edge_num[time + t];
	}
	

}

void process_node_embedding(int time, int time_step, int neighborhood_reverse_renumbering_table[TOTAL_TIME][501]) { // This function has been checked.
	for (int t = 0; t < time_step; t++) {
		for (int nd = 0; nd < node_num[time + t]; nd++) {
			int position = neighborhood_reverse_renumbering_table[time + t][nd];
			//printf("the value of pos is:%d\n", position);
			for (int dim = 0; dim < WEIGHT_LAYER1_INPUTDIM; dim++) {
				

				node_embedding_time[t][nd][dim] = node_embedding[position][dim];
			}
		}
	}

	
}

void process_edge_embedding(int time, int time_step) {// This function has been checked.
	int total = 0;
	int total_internel = 0;
	for (int i = 0; i < time; i++) {
		total = total + edge_num[i];
	}
	for (int t = 0; t < time_step; t++) {
		
		for (int e = 0; e < edge_num[time + t]; e++) {
			for (int dim = 0; dim < WEIGHT_LAYER1_INPUTDIM; dim++) {
				edge_embedding_time[t][e][dim] = edge_embedding[total + total_internel + e][dim];
			}
		}
		total_internel = total_internel + edge_num[time + t];
	}
}

void process_num(int time, int time_step) { // This function has been checked.
	int total_node = 0;
	int total_edge = 0;
	int total_internel = 0;
	for (int i = 0; i < time; i++) {
		total_node = total_node + 1;
		total_edge = total_edge + 1;
	}
	for (int t = 0; t < time_step; t++) {
		node_num_time[t] = node_num[total_node + total_internel + t];
		edge_num_time[t] = edge_num[total_edge + total_internel + t];
	}
}

void process_renumbering_table(int time, int time_step) {// This function has been checked.
	int total_1 = 0;
	int total_2 = 0;
	int total_internel_1 = 0;
	int total_internel_2 = 0;
	for (int i = 0; i < time; i++) {
		total_1 = total_1 + 1;
		total_2 = total_2 + 1;
	}

	for (int t = 0; t < time_step; t++) {
		for (int nd = 0; nd < NUM_NODE; nd++) {
			neighborhood_renumbering_table_time[t][nd] = neighborhood_renumbering_table[total_1 + t][nd];
		}
	}

	for (int t = 0; t < time_step; t++) {
		for (int nd = 0; nd < SUB_GRAPH_NODE_NUM; nd++) {
			neighborhood_reverse_renumbering_table_time[t][nd] = neighborhood_reverse_renumbering_table[total_2 + t][nd];
		}
	}
	
}


void convert_weights_to_fix() { //the weight input layer 1 is 67. the weight output layer 1 is 122 (GNN); the weight input layer 2 is 122, the weight output of layer 2 is 122 (RNN�� 
	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_OUTPUTDIM; dim_out++) {
			gcn_node_weight_layer1_fixed[dim_in * WEIGHT_LAYER1_OUTPUTDIM + dim_out] = (WT_TYPE)gcn_node_weight_layer1[dim_in][dim_out];
		}
	}


	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_INPUTDIM; dim_out++) {
			rnn_update_weight_W_fixed[dim_in * WEIGHT_LAYER1_INPUTDIM + dim_out] = (WT_TYPE)gru_update_layer1_W[dim_in][dim_out];

		}
	}


	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_INPUTDIM; dim_out++) {
			rnn_update_weight_U_fixed[dim_in * WEIGHT_LAYER1_INPUTDIM + dim_out] = (WT_TYPE)gru_update_layer1_U[dim_in][dim_out];

		}
	}

	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_OUTPUTDIM; dim_out++) {
			rnn_update_bias_fixed[dim_in * WEIGHT_LAYER1_OUTPUTDIM + dim_out] = (WT_TYPE)gru_update_layer1_bias[dim_in][dim_out];

		}
	}

	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_INPUTDIM; dim_out++) {
			rnn_reset_weight_W_fixed[dim_in * WEIGHT_LAYER1_INPUTDIM + dim_out] = (WT_TYPE)gru_reset_layer1_W[dim_in][dim_out];

		}
	}

	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_INPUTDIM; dim_out++) {
			rnn_reset_weight_U_fixed[dim_in * WEIGHT_LAYER1_INPUTDIM + dim_out] = (WT_TYPE)gru_reset_layer1_U[dim_in][dim_out];

		}
	}

	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_OUTPUTDIM; dim_out++) {
			rnn_reset_bias_fixed[dim_in * WEIGHT_LAYER1_OUTPUTDIM + dim_out] = (WT_TYPE)gru_reset_layer1_bias[dim_in][dim_out];

		}
	}


	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_INPUTDIM; dim_out++) {
			rnn_htilda_weight_W_fixed[dim_in * WEIGHT_LAYER1_INPUTDIM + dim_out] = (WT_TYPE)gru_htilda_layer1_W[dim_in][dim_out];

		}
	}

	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_INPUTDIM; dim_out++) {
			rnn_htilda_weight_U_fixed[dim_in * WEIGHT_LAYER1_INPUTDIM + dim_out] = (WT_TYPE)gru_htilda_layer1_U[dim_in][dim_out];

		}
	}

	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_OUTPUTDIM; dim_out++) {
			rnn_htilda_bias_fixed[dim_in * WEIGHT_LAYER1_OUTPUTDIM + dim_out] = (WT_TYPE)gru_htilda_layer1_bias[dim_in][dim_out];

		}
	}

	printf("Convert weights to fix finished ...\n");
}


void convert_embedding_to_fix() {
	for (int time = 0; time < TIME_STEP; time++) {
		for (int i = 0; i < SUB_GRAPH_NODE_NUM; i++) {
			for (int j = 0; j < WEIGHT_LAYER1_INPUTDIM; j++) {
				node_embedding_in[time * SUB_GRAPH_NODE_NUM * WEIGHT_LAYER1_INPUTDIM + i * WEIGHT_LAYER1_INPUTDIM + j] = (FM_TYPE)node_embedding_time[time][i][j];
			}
		}
	}

	for (int time = 0; time < TIME_STEP; time++) {
		for (int i = 0; i < SUB_GRAPH_EDGE_NUM; i++) {
			for (int j = 0; j < WEIGHT_LAYER1_INPUTDIM; j++) {
				edge_embedding_in[time * SUB_GRAPH_EDGE_NUM * WEIGHT_LAYER1_INPUTDIM + i * WEIGHT_LAYER1_INPUTDIM + j] = (FM_TYPE)edge_embedding_time[time][i][j];
			}
		}
	}

}



void load_weights(WT_TYPE gnn_layer_1_weight_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM],
	WT_TYPE rnn_update_weight_W_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_update_weight_U_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_update_bias_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM],
	WT_TYPE rnn_reset_weight_W_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_reset_weight_U_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_reset_bias_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM],
	WT_TYPE rnn_htilda_weight_W_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_htilda_weight_U_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_htilda_bias_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM]
) { // This function has been checked.

#pragma HLS inline off
	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_OUTPUTDIM; dim_out++) {
			weight_buffer_2[dim_in][dim_out] = gnn_layer_1_weight_fixed[dim_in * WEIGHT_LAYER1_OUTPUTDIM + dim_out];
			//printf("the value is:%.6f\n", (float)weight_buffer_2[dim_in][dim_out]);
		}
	}


	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_INPUTDIM; dim_out++) {
			rnn_update_weight_W_buffer[dim_in][dim_out] = rnn_update_weight_W_fixed[dim_in * WEIGHT_LAYER1_INPUTDIM + dim_out];

		}
	}

	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_INPUTDIM; dim_out++) {
			rnn_update_weight_U_buffer[dim_in][dim_out] = rnn_update_weight_U_fixed[dim_in * WEIGHT_LAYER1_INPUTDIM + dim_out];

		}
	}

	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_OUTPUTDIM; dim_out++) {
			rnn_update_bias_buffer[dim_in][dim_out] = rnn_update_bias_fixed[dim_in * WEIGHT_LAYER1_OUTPUTDIM + dim_out];

		}
	}


	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_INPUTDIM; dim_out++) {
			rnn_reset_weight_W_buffer[dim_in][dim_out] = rnn_reset_weight_W_fixed[dim_in * WEIGHT_LAYER1_INPUTDIM + dim_out];

		}
	}

	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_INPUTDIM; dim_out++) {
			rnn_reset_weight_U_buffer[dim_in][dim_out] = rnn_reset_weight_U_fixed[dim_in * WEIGHT_LAYER1_INPUTDIM + dim_out];

		}
	}

	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_OUTPUTDIM; dim_out++) {
			rnn_reset_bias_buffer[dim_in][dim_out] = rnn_reset_bias_fixed[dim_in * WEIGHT_LAYER1_OUTPUTDIM + dim_out];

		}
	}

	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_INPUTDIM; dim_out++) {
			rnn_htilda_weight_W_buffer[dim_in][dim_out] = rnn_htilda_weight_W_fixed[dim_in * WEIGHT_LAYER1_INPUTDIM + dim_out];

		}
	}

	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_INPUTDIM; dim_out++) {
			rnn_htilda_weight_U_buffer[dim_in][dim_out] = rnn_htilda_weight_U_fixed[dim_in * WEIGHT_LAYER1_INPUTDIM + dim_out];

		}
	}

	for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_OUTPUTDIM; dim_out++) {
			rnn_htilda_bias_buffer[dim_in][dim_out] = rnn_htilda_bias_fixed[dim_in * WEIGHT_LAYER1_OUTPUTDIM + dim_out];
			//printf("the value is:%.6f\n", (float)rnn_htilda_bias_buffer[dim_in][dim_out]);

		}
	}



}


void load_graph(int edge_list_in[TIME_STEP][SUB_GRAPH_EDGE_NUM * 2], 
	            int edge_list[SUB_GRAPH_EDGE_NUM * 2], 
	            int neighborhood_ref_table_time[TIME_STEP][NUM_NODE], 
	            WT_TYPE node_embedding_in[TIME_STEP * SUB_GRAPH_NODE_NUM * WEIGHT_LAYER1_INPUTDIM], 
	            WT_TYPE edge_attr_in[TIME_STEP * SUB_GRAPH_EDGE_NUM * WEIGHT_LAYER1_INPUTDIM], 
	            WT_TYPE node_embedding_table[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_INPUTDIM], 
	            WT_TYPE edge_embedding_table[SUB_GRAPH_EDGE_NUM][WEIGHT_LAYER1_INPUTDIM], 
	            int num_of_nodes, int num_of_edges, int time)
{  // This function has been checked.
#pragma HLS inline off
	for (int i = 0; i < num_of_nodes; i++) {
		for (int dim = 0; dim < WEIGHT_LAYER1_INPUTDIM; dim++) {
			node_embedding_table[i][dim] = node_embedding_in[time * SUB_GRAPH_NODE_NUM * WEIGHT_LAYER1_INPUTDIM + i * WEIGHT_LAYER1_INPUTDIM + dim];
		}

	}

	for (int e = 0; e < num_of_edges; e++) {
		for (int dim = 0; dim < WEIGHT_LAYER1_INPUTDIM; dim++) {
			edge_embedding_table[e][dim] = edge_attr_in[time * SUB_GRAPH_EDGE_NUM * WEIGHT_LAYER1_INPUTDIM + e * WEIGHT_LAYER1_INPUTDIM + dim];
		}
	}

	for (int i = 0; i < num_of_edges * 2; i++) {
		edge_list[i] = edge_list_in[time][i];
	}


	for (int n = 0; n < num_of_nodes * 3; n++) {
		degree_table[n] = 0;
	}

	for (int e = 0; e < num_of_edges; e++) {
		int u = neighborhood_ref_table_time[time][edge_list[e * 2]];     // source node id
		//printf("the original source node id is:%d\n", edge_list[e * 2]);
		//printf("the source node id is:%d\n", u);
		int v = neighborhood_ref_table_time[time][edge_list[e * 2 + 1]];   // target node id
		//printf("the original target node id is:%d\n", edge_list[e * 2 + 1]);
		//printf("the target node id is:%d\n", v);

		degree_table[u * 3] += 1;       //the format of degree table: [ total_neighbrhood_number  | start_idx_for_neigh_table  |  offset_idx_for_neigh_table  ]
	}                                   //the format of neighbor table: [neighborhood_node_index | edge number]

	for (int n = 1; n < num_of_nodes; n++) {
		degree_table[n * 3 + 1] = degree_table[(n - 1) * 3] * 2 + degree_table[(n - 1) * 3 + 1]; // *2: one is for the neighbor node id; the next one is for the corresponding edge id
	}// this for loop is used to define the start idx = past node start idx + past node degree *2 ( for neighbor_idx and edge_index)

	for (int e = 0; e < num_of_edges; e++) {
		int u = neighborhood_ref_table_time[time][edge_list[e * 2]];     // source node id
		int v = neighborhood_ref_table_time[time][edge_list[e * 2 + 1]];   // target node id

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



void message_passing(FM_TYPE emb_val[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_INPUTDIM], 
	                 int num_of_nodes, int num_of_edges, 
	                 FM_TYPE message_tb[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_INPUTDIM], 
	                 FM_TYPE edge_attr[SUB_GRAPH_EDGE_NUM][WEIGHT_LAYER1_INPUTDIM])
{  //This function has been checked.
#pragma HLS inline off
#pragma HLS array_partition variable=message_tb dim=2 complete
#pragma HLS array_partition variable=edge_attr dim=2 complete
	
	for (int nd = 0; nd < num_of_nodes; nd++) {
		int u = nd;  // u is the source node
		int total_neigh = degree_table[u * 3];  // we need a degree table 1. neighborhood number, start index in the neighbor table.
		int start_idx = degree_table[u * 3 + 1];

		FM_TYPE norm[160];
		FM_TYPE node_embedding_vec[WEIGHT_LAYER1_INPUTDIM];
		FM_TYPE msg = 0; //this is used to store message for one edge

		for (int i = 0; i < 160; i++) {
			norm[i] = 0;
		}

		for (int i = 0; i < WEIGHT_LAYER1_INPUTDIM; i++) {
			node_embedding_vec[i] = 0;
		}

		for (int i = 0; i < total_neigh; i++) {
			int v = neighbor_table[start_idx + i * 2];  // v is the neighborhood node index
			int e = neighbor_table[start_idx + i * 2 + 1];// e is the edge index
			norm[i] = (1.0 / sqrt(degree_table[u * 3] + 1.0)) * (1.0 / sqrt(degree_table[v * 3] + 1.0));
			//printf("the value is:%.6f",)
			for (int dim = 0; dim < WEIGHT_LAYER1_INPUTDIM; dim++) {
#pragma HLS pipeline II = 1
				//msg = norm[i] * (node_embedding_vec[dim] + edge_attr[dim]);  //for each node, we need to load an edge embedding table on chip
				msg = norm[i] * (edge_attr[e][dim] + message_tb[v][dim]); //?
				node_embedding_vec[dim] += msg;
			}
		}

		for (int dim = 0; dim < WEIGHT_LAYER1_INPUTDIM; dim++) {
#pragma HLS pipeline II = 1
			FM_TYPE temp = (1.0 / (degree_table[u * 3] + 1.0));
			node_embedding_vec[dim] +=  temp * message_tb[u][dim];
			emb_val[nd][dim] = node_embedding_vec[dim];
			

		}
	}

}

void write_to_fifo(FM_TYPE rnn_input[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM],
	hls::stream<FM_TYPE>& rnn_input_fifo_1,
	hls::stream<FM_TYPE>& rnn_input_fifo_2,
	hls::stream<FM_TYPE>& rnn_input_fifo_3,
	hls::stream<FM_TYPE>& rnn_input_fifo_4,
	hls::stream<FM_TYPE>& rnn_input_fifo_5,
	hls::stream<FM_TYPE>& rnn_input_fifo_6,
	hls::stream<FM_TYPE>& rnn_input_fifo_7) {  // This function has been checked.
	/*
	
	for (int i = 0; i < WEIGHT_LAYER1_INPUTDIM; i++) {
		for (int j = 0; j < WEIGHT_LAYER1_OUTPUTDIM; j++) {
			printf("the value is:%.6f\n", (float)rnn_input[i][j]);
		}
	}
	
	*/
	

	for (int dim_out = 0; dim_out < WEIGHT_LAYER1_OUTPUTDIM; dim_out++) {
		for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
			rnn_input_fifo_1.write(rnn_input[dim_in][dim_out]);
			rnn_input_fifo_2.write(rnn_input[dim_in][dim_out]);
			rnn_input_fifo_3.write(rnn_input[dim_in][dim_out]);
			rnn_input_fifo_4.write(rnn_input[dim_in][dim_out]);
			rnn_input_fifo_5.write(rnn_input[dim_in][dim_out]);
			rnn_input_fifo_6.write(rnn_input[dim_in][dim_out]);
			rnn_input_fifo_7.write(rnn_input[dim_in][dim_out]);
		}
	}
}

void GRU_cell_sigmoid(hls::stream<FM_TYPE>& rnn_input_1, 
	hls::stream<FM_TYPE>& rnn_input_2, 
	hls::stream<FM_TYPE>& rnn_output, 
	WT_TYPE rnn_weight_W[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM], 
	WT_TYPE rnn_weight_U[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM], 
	WT_TYPE rnn_bias[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM]) 


{// This function has been checked.
#pragma HLS array_partition variable=rnn_weight_W dim=2 complete
#pragma HLS array_partition variable=rnn_weight_U dim=2 complete

	for (int dim_out = 0; dim_out < WEIGHT_LAYER1_OUTPUTDIM; dim_out++) {
		FM_TYPE rnn_input_1_vector[WEIGHT_LAYER1_INPUTDIM];
		FM_TYPE rnn_input_2_vector[WEIGHT_LAYER1_INPUTDIM];
		FM_TYPE rnn_output_vector[WEIGHT_LAYER1_INPUTDIM];
#pragma HLS array_partition variable=rnn_input_1_vector dim=1 complete
#pragma HLS array_partition variable=rnn_input_2_vector dim=1 complete
		for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
			rnn_input_1_vector[dim_in] = rnn_input_1.read();
			
			rnn_input_2_vector[dim_in] = rnn_input_2.read();
			//printf("the value is:%.6f\n", (float)rnn_input_2_vector[dim_in]);
		}
		for (int nd = 0; nd < WEIGHT_LAYER1_INPUTDIM; nd++) {

#pragma HLS pipeline II = 1
			FM_TYPE tmp = rnn_bias[nd][dim_out];
			//printf("the value is:%.6f\n", (float)tmp);
			for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
				//tmp += rnn_input[nd][dim_in] * rnn_weight[dim_in][dim_out];
				tmp += rnn_weight_W[nd][dim_in] * rnn_input_1_vector[dim_in] + rnn_weight_U[nd][dim_in] * rnn_input_2_vector[dim_in];
			}
			rnn_output_vector[nd] = tmp;
			
			rnn_output_vector[nd] = 1.0 / (1.0 + expf(-rnn_output_vector[nd]));
			rnn_output.write(rnn_output_vector[nd]);
			//printf("the value is:%.6f\n", (float)rnn_output_vector[nd]);
		}
	}
}


void GRU_multi_1(hls::stream<FM_TYPE>& rnn_input, 
	             hls::stream<FM_TYPE>& reset_output_fifo, 
	             hls::stream<FM_TYPE>& htilda_output_fifo) {
	// This function has been checked.
	for (int dim_out = 0; dim_out < WEIGHT_LAYER1_OUTPUTDIM; dim_out++) {
		FM_TYPE rnn_input_vector[WEIGHT_LAYER1_INPUTDIM];
		FM_TYPE reset_output_vector[WEIGHT_LAYER1_INPUTDIM];
		FM_TYPE htilda_output_vector[WEIGHT_LAYER1_INPUTDIM];
#pragma HLS array_partition variable=rnn_input_vector dim=1 complete
#pragma HLS array_partition variable=reset_output_vector dim=1 complete
#pragma HLS array_partition variable=htilda_output_vector dim=1 complete
		for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
			rnn_input_vector[dim_in] = rnn_input.read();
			reset_output_vector[dim_in] = reset_output_fifo.read();
		}
		for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
#pragma HLS pipeline II = 1
			htilda_output_vector[dim_in] = rnn_input_vector[dim_in] * reset_output_vector[dim_in];
		}
		for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
			htilda_output_fifo.write(htilda_output_vector[dim_in]);
			//printf("the value is:%.6f\n", (float)htilda_output_vector[dim_in]);
		}
	}


}


void GRU_cell_tanh(hls::stream<FM_TYPE>& rnn_input_1, 
	               hls::stream<FM_TYPE>& rnn_input_2, 
	               hls::stream<FM_TYPE>& rnn_output, 
	               WT_TYPE rnn_weight_W[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM], 
	               WT_TYPE rnn_weight_U[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM], 
	               WT_TYPE rnn_bias[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM]) {
#pragma HLS array_partition variable=rnn_weight_W dim=2 complete
#pragma HLS array_partition variable=rnn_weight_U dim=2 complete

	for (int dim_out = 0; dim_out < WEIGHT_LAYER1_OUTPUTDIM; dim_out++) {
		FM_TYPE rnn_input_1_vector[WEIGHT_LAYER1_INPUTDIM];
		FM_TYPE rnn_input_2_vector[WEIGHT_LAYER1_INPUTDIM];
		FM_TYPE rnn_output_vector[WEIGHT_LAYER1_INPUTDIM];
#pragma HLS array_partition variable=rnn_input_1_vector dim=1 complete
#pragma HLS array_partition variable=rnn_input_2_vector dim=1 complete
		for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
			rnn_input_1_vector[dim_in] = rnn_input_1.read();
			rnn_input_2_vector[dim_in] = rnn_input_2.read();
		}
		for (int nd = 0; nd < WEIGHT_LAYER1_INPUTDIM; nd++) {

#pragma HLS pipeline II = 1
			FM_TYPE tmp = rnn_bias[nd][dim_out];
			for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
				//tmp += rnn_input[nd][dim_in] * rnn_weight[dim_in][dim_out];
				tmp += rnn_weight_W[nd][dim_in] * rnn_input_1_vector[dim_in] + rnn_weight_U[nd][dim_in] * rnn_input_2_vector[dim_in];
			}
			rnn_output_vector[nd] = tmp;
			rnn_output_vector[nd] = (expf(rnn_output_vector[nd]) - expf(-rnn_output_vector[nd])) / (expf(rnn_output_vector[nd]) + expf(-rnn_output_vector[nd]));
			rnn_output.write(rnn_output_vector[nd]);
			//printf("the value is:%.6f\n", (float)rnn_output_vector[nd]);
		}
	}
}


void GRU_multi_2(hls::stream<FM_TYPE>& update_output, 
	             hls::stream<FM_TYPE>& rnn_input, 
	             hls::stream<FM_TYPE>& htilda_output, 
	             WT_TYPE rnn_output[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM]) 

{  //This function has been checked.
#pragma HLS array_partition variable=rnn_output dim=1 complete	

	for (int dim_out = 0; dim_out < WEIGHT_LAYER1_OUTPUTDIM; dim_out++) {
		FM_TYPE update_output_vector[WEIGHT_LAYER1_INPUTDIM];
		FM_TYPE rnn_input_vector[WEIGHT_LAYER1_INPUTDIM];
		FM_TYPE htilda_output_vector[WEIGHT_LAYER1_INPUTDIM];
		FM_TYPE rnn_output_vector[WEIGHT_LAYER1_INPUTDIM];
#pragma HLS array_partition variable=update_output_vector dim=1 complete
#pragma HLS array_partition variable=rnn_input_vector dim=1 complete
#pragma HLS array_partition variable=htilda_output_vector dim=1 complete	

		for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
			update_output_vector[dim_in] = update_output.read();
			//printf("the value is:%.6f\n", (float)update_output_vector[dim_in]);
			rnn_input_vector[dim_in] = rnn_input.read();
			htilda_output_vector[dim_in] = htilda_output.read();

		}

		for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {
#pragma HLS pipeline II = 1
			
			rnn_output[dim_in][dim_out] = ((WT_TYPE)1.0 - update_output_vector[dim_in]) * rnn_input_vector[dim_in] + update_output_vector[dim_in] * htilda_output_vector[dim_in];
			//printf("the value is:%.6f\n", (float)rnn_output[dim_in][dim_out]);
		}
	}


}


void CONV_GNN(FM_TYPE gnn_input[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_INPUTDIM], 
	         FM_TYPE gnn_output[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_OUTPUTDIM], 
	         FM_TYPE conv_weights[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_OUTPUTDIM],
	         int num_of_nodes, int num_of_edges)
{  //This function has been checked.
#pragma HLS inline off
#pragma HLS array_partition variable=conv_weights dim=1 complete
#pragma HLS array_partition variable=gnn_input dim=2 complete

	
	for (int nd = 0; nd < num_of_nodes; nd++) {

		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_OUTPUTDIM; dim_out++) {
#pragma HLS PIPELINE II = 1


			FM_TYPE tmp = 0;
			gnn_output[nd][dim_out] = 0;
			for (int dim_in = 0; dim_in < WEIGHT_LAYER1_INPUTDIM; dim_in++) {

				tmp += gnn_input[nd][dim_in] * conv_weights[dim_in][dim_out];
			}
			gnn_output[nd][dim_out] += tmp;

		}

	}
}




void GRU_dataflow(FM_TYPE rnn_input[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM], 
	FM_TYPE rnn_output[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM], 
	WT_TYPE rnn_update_weight_W[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM], 
	WT_TYPE rnn_update_weight_U[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM], 
	WT_TYPE rnn_update_bias[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM], 
	WT_TYPE rnn_reset_weight_W[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM], 
	WT_TYPE rnn_reset_weight_U[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM], 
	WT_TYPE rnn_reset_bias[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM], 
	WT_TYPE rnn_htilda_weight_W[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM], 
	WT_TYPE rnn_htilda_weight_U[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM], 
	WT_TYPE rnn_htilda_bias[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM])


{
	FM_TYPE update_output[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];
	FM_TYPE reset_output[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];
	FM_TYPE htilda_output_temp[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];
	FM_TYPE htilda_output[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM];


	hls::stream<FM_TYPE> rnn_input_fifo_1;
	hls::stream<FM_TYPE> rnn_input_fifo_2;
	hls::stream<FM_TYPE> rnn_input_fifo_3;
	hls::stream<FM_TYPE> rnn_input_fifo_4;
	hls::stream<FM_TYPE> rnn_input_fifo_5;
	hls::stream<FM_TYPE> rnn_input_fifo_6;
	hls::stream<FM_TYPE> rnn_input_fifo_7;



	hls::stream<FM_TYPE> update_output_fifo;
	hls::stream<FM_TYPE> reset_output_fifo;
	hls::stream<FM_TYPE> htilda_output_temp_fifo;
	hls::stream<FM_TYPE> htilda_output_fifo;


#pragma HLS STREAM variable=rnn_input_fifo_1 depth=400
#pragma HLS STREAM variable=rnn_input_fifo_2 depth=400
#pragma HLS STREAM variable=rnn_input_fifo_3 depth=400
#pragma HLS STREAM variable=rnn_input_fifo_4 depth=400
#pragma HLS STREAM variable=rnn_input_fifo_5 depth=400
#pragma HLS STREAM variable=rnn_input_fifo_6 depth=400
#pragma HLS STREAM variable=rnn_input_fifo_7 depth=400



#pragma HLS STREAM variable=update_output_fifo depth=400
#pragma HLS STREAM variable=reset_output_fifo depth=400
#pragma HLS STREAM variable=htilda_output_temp_fifo depth=400
#pragma HLS STREAM variable=htilda_output_fifo depth=400
#pragma HLS dataflow
	write_to_fifo(rnn_input, rnn_input_fifo_1, rnn_input_fifo_2, rnn_input_fifo_3, rnn_input_fifo_4, rnn_input_fifo_5, rnn_input_fifo_6, rnn_input_fifo_7);
	GRU_cell_sigmoid(rnn_input_fifo_1, rnn_input_fifo_2, update_output_fifo, rnn_update_weight_W, rnn_update_weight_U, rnn_update_bias);
	GRU_cell_sigmoid(rnn_input_fifo_3, rnn_input_fifo_4, reset_output_fifo, rnn_reset_weight_W, rnn_reset_weight_U, rnn_reset_bias);
	GRU_multi_1(rnn_input_fifo_5, reset_output_fifo, htilda_output_temp_fifo);
	GRU_cell_tanh(rnn_input_fifo_6, htilda_output_temp_fifo, htilda_output_fifo, rnn_htilda_weight_W, rnn_htilda_weight_U, rnn_htilda_bias);
	GRU_multi_2(update_output_fifo, rnn_input_fifo_7, htilda_output_fifo, rnn_output);


}







void compute_dataflow(int edge_list_in[TIME_STEP][SUB_GRAPH_EDGE_NUM * 2],
	int edge_list[SUB_GRAPH_EDGE_NUM * 2],
	int neighborhood_ref_table_time[TIME_STEP][NUM_NODE],
	FM_TYPE node_embedding_in[TIME_STEP * SUB_GRAPH_NODE_NUM * WEIGHT_LAYER1_INPUTDIM],
	FM_TYPE edge_attr_in[TIME_STEP * SUB_GRAPH_EDGE_NUM * WEIGHT_LAYER1_INPUTDIM],
	//WT_TYPE gnn_weights[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM],

	FM_TYPE output[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_OUTPUTDIM],

	WT_TYPE rnn_update_weight_W[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_update_weight_U[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_update_bias[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM],
	WT_TYPE rnn_reset_weight_W[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_reset_weight_U[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_reset_bias[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM],
	WT_TYPE rnn_htilda_weight_W[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_htilda_weight_U[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_htilda_bias[WEIGHT_LAYER1_INPUTDIM][WEIGHT_LAYER1_OUTPUTDIM],
	int node_number_time[TIME_STEP], int edge_number_time[TIME_STEP], int step
	)

{
#pragma HLS inline off
	for (int time = 0; time < step; time++) {
		if (time % 2 == 0) {
			load_graph(edge_list_in, edge_list, neighborhood_ref_table_time, node_embedding_in, edge_attr_in, node_embedding_table_1, edge_embedding_table_1, node_number_time[time], edge_number_time[time], time);
			message_passing(emb_buffer_1, node_number_time[time], edge_number_time[time], node_embedding_table_1, edge_embedding_table_1);
			GRU_dataflow(weight_buffer_2, weight_buffer_1, rnn_update_weight_W, rnn_update_weight_U, rnn_update_bias, rnn_reset_weight_W, rnn_reset_weight_U, rnn_reset_bias, rnn_htilda_weight_W, rnn_htilda_weight_U, rnn_htilda_bias);
			CONV_GNN(emb_buffer_1, output, weight_buffer_1, node_number_time[time], edge_number_time[time]);

		}
		else if (time % 2 == 1) {


			load_graph(edge_list_in, edge_list, neighborhood_ref_table_time, node_embedding_in, edge_attr_in, node_embedding_table_1, edge_embedding_table_1, node_number_time[time], edge_number_time[time], time);
			message_passing(emb_buffer_2, node_number_time[time], edge_number_time[time], node_embedding_table_1, edge_embedding_table_1);
			GRU_dataflow(weight_buffer_1, weight_buffer_2, rnn_update_weight_W, rnn_update_weight_U, rnn_update_bias, rnn_reset_weight_W, rnn_reset_weight_U, rnn_reset_bias, rnn_htilda_weight_W, rnn_htilda_weight_U, rnn_htilda_bias);
			CONV_GNN(emb_buffer_2, output, weight_buffer_2, node_number_time[time], edge_number_time[time]);
			
		}
			
	}

	
	
}


void write_back(FM_TYPE output[SUB_GRAPH_NODE_NUM * WEIGHT_LAYER1_OUTPUTDIM], 
	            FM_TYPE output_table[SUB_GRAPH_NODE_NUM][WEIGHT_LAYER1_OUTPUTDIM], 
	            int num_of_nodes) {
	for (int nd = 0; nd < num_of_nodes; nd++) {
		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_OUTPUTDIM; dim_out++) {
			output[nd * WEIGHT_LAYER1_OUTPUTDIM + dim_out] = output_table[nd][dim_out];
			//printf("the value is:%.6f\n", (float)output_table[nd][dim_out]);
		}
	}
}


void DGNN_booster_evolveGCN(
	int edge_list_in[TIME_STEP][SUB_GRAPH_EDGE_NUM * 2],
	int node_number_time[TIME_STEP],
	int edge_number_time[TIME_STEP],
	int neighborhood_ref_table_time[TIME_STEP][NUM_NODE],
	int neighborhood_reverse_ref_table_time[TIME_STEP][SUB_GRAPH_NODE_NUM],

	WT_TYPE gnn_layer_1_weight_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM],
	

	WT_TYPE rnn_update_weight_W_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_update_weight_U_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_update_bias_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM],
	WT_TYPE rnn_reset_weight_W_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_reset_weight_U_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_reset_bias_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM],
	WT_TYPE rnn_htilda_weight_W_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_htilda_weight_U_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_INPUTDIM],
	WT_TYPE rnn_htilda_bias_fixed[WEIGHT_LAYER1_INPUTDIM * WEIGHT_LAYER1_OUTPUTDIM],


	FM_TYPE node_embedding_fixed[TIME_STEP * SUB_GRAPH_NODE_NUM * WEIGHT_LAYER1_INPUTDIM],
	FM_TYPE edge_attr_in_fixed[TIME_STEP * SUB_GRAPH_EDGE_NUM * WEIGHT_LAYER1_INPUTDIM],

	FM_TYPE output_fixed[SUB_GRAPH_NODE_NUM * WEIGHT_LAYER1_OUTPUTDIM],
	int step[100]
)

{

#pragma HLS INTERFACE s_axilite port=return

#pragma HLS INTERFACE m_axi depth=100000 port=edge_list_in offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=node_number_time offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=edge_number_time offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=neighborhood_ref_table_time offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=neighborhood_reverse_ref_table_time offset=slave bundle=INPUT

#pragma HLS INTERFACE m_axi depth=100000 port=gnn_layer_1_weight_fixed offset=slave bundle=INPUT

#pragma HLS INTERFACE m_axi depth=100000 port=rnn_update_weight_W_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=rnn_update_weight_U_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=rnn_update_bias_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=rnn_reset_weight_W_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=rnn_reset_weight_U_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=rnn_reset_bias_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=rnn_htilda_weight_W_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=rnn_htilda_weight_U_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=rnn_htilda_bias_fixed offset=slave bundle=INPUT

#pragma HLS INTERFACE m_axi depth=100000 port=node_embedding_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=edge_attr_in_fixed offset=slave bundle=INPUT
#pragma HLS INTERFACE m_axi depth=100000 port=output_fixed offset=slave bundle=OUTPUT
#pragma HLS INTERFACE m_axi depth=100000 port=step offset=slave bundle=INPUT

#pragma HLS bind_storage variable=weight_buffer_1 impl = bram
#pragma HLS bind_storage variable=weight_buffer_2 impl = bram

#pragma HLS bind_storage variable=emb_buffer_1 impl = bram
#pragma HLS bind_storage variable=emb_buffer_2 impl = bram

#pragma HLS bind_storage variable=rnn_update_weight_W_buffer impl = bram
#pragma HLS bind_storage variable=rnn_update_weight_U_buffer impl = bram
#pragma HLS bind_storage variable=rnn_update_bias_buffer impl = bram

#pragma HLS bind_storage variable=rnn_reset_weight_W_buffer impl = bram
#pragma HLS bind_storage variable=rnn_reset_weight_U_buffer impl = bram
#pragma HLS bind_storage variable=rnn_reset_bias_buffer impl = bram

#pragma HLS bind_storage variable=rnn_htilda_weight_W_buffer impl = bram
#pragma HLS bind_storage variable=rnn_htilda_weight_U_buffer impl = bram
#pragma HLS bind_storage variable=rnn_htilda_bias_buffer impl = bram

#pragma HLS bind_storage variable=output_table impl = bram


	load_weights(gnn_layer_1_weight_fixed, rnn_update_weight_W_fixed, rnn_update_weight_U_fixed, rnn_update_bias_fixed, rnn_reset_weight_W_fixed, rnn_reset_weight_U_fixed, rnn_reset_bias_fixed, rnn_htilda_weight_W_fixed, rnn_htilda_weight_U_fixed, rnn_htilda_bias_fixed);
	
	compute_dataflow(edge_list_in, edge_list, neighborhood_ref_table_time, node_embedding_fixed, edge_attr_in_fixed,  output_table, rnn_update_weight_W_buffer, rnn_update_weight_U_buffer, rnn_update_bias_buffer, rnn_reset_weight_W_buffer, rnn_reset_weight_U_buffer, rnn_reset_bias_buffer, rnn_htilda_weight_W_buffer, rnn_htilda_weight_U_buffer, rnn_htilda_bias_buffer, node_number_time, edge_number_time, step[0]);
	write_back(output_fixed, output_table, node_number_time[step[0]-1]);


}




int main() {

	load_weights_float();
	//printf("execute at chkpt 1\n");
	load_node_embedding();
	//printf("execute at chkpt 2\n");
	load_edge_embedding();
	load_edge_num();
	load_node_num();
	load_edge_info();
	load_renumbering_table();
	load_reverse_renumbering_table();
	
	process_edge_list_info(0, 3);
	process_node_embedding(0, 3, neighborhood_reverse_renumbering_table);
	process_edge_embedding(0, 3);
	process_num(0, 3);
	process_renumbering_table(0, 3);

	convert_weights_to_fix();
	convert_embedding_to_fix();

	int step[100];
	step[0] = 10;
	DGNN_booster_evolveGCN(edge_list_time, node_num_time, edge_num_time, neighborhood_renumbering_table_time, neighborhood_reverse_renumbering_table_time, gcn_node_weight_layer1_fixed, rnn_update_weight_W_fixed, rnn_update_weight_U_fixed, rnn_update_bias_fixed, rnn_reset_weight_W_fixed, rnn_reset_weight_U_fixed, rnn_reset_bias_fixed, rnn_htilda_weight_W_fixed, rnn_htilda_weight_U_fixed, rnn_htilda_bias_fixed, node_embedding_in, edge_embedding_in, output_fixed, step);
	
	printf("the simulation ends!\n");
	//   for (int nd = 0; nd < SUB_GRAPH_NODE_NUM; nd++) {
	//  		for (int dim_out = 0; dim_out < WEIGHT_LAYER1_OUTPUTDIM; dim_out++) {
	//  			printf("the value is:%.6f\n", (float)output_fixed[nd * WEIGHT_LAYER1_OUTPUTDIM + dim_out]);
	//  		}
	// }

		

}
