#include <stdio.h>
#include <stdlib.h>

#include <vector> 
#include <algorithm>
#include <string.h>
#include <iostream>
#include <cstddef>

#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include "hls_math.h"

#include <fstream>
#include <cmath>

#include <ap_int.h>



// #define N 128 //The number of clients
// #define C 10 //The number of clients to be selected
// #define SORT_NUM 32

// #define N_CLIENT 72

// #define BUFFER_SIZE 64
// #define UNROLL_FACTOR 16

#define N 104 //The number of clients
#define C 10 //The number of clients to be selected
#define SORT_NUM 32

#define N_CLIENT 100

#define BUFFER_SIZE 13
#define UNROLL_FACTOR 13

#define SCALE 5



//typedef float DATA_TYPE;
//typedef float Key_Type;
typedef ap_fixed<32,16> DATA_TYPE;
typedef ap_fixed<32,16> Key_Type;
typedef struct {
    Key_Type number;
    int index;
} Data_Sort;

// define some on-chip buffers

DATA_TYPE mu[N];
DATA_TYPE sigma[BUFFER_SIZE][BUFFER_SIZE];
DATA_TYPE a[N];
DATA_TYPE beta[N];
DATA_TYPE sigma_1d[N];



DATA_TYPE mu_bar_accumu[N];
bool client_selection_flag[N_CLIENT];
DATA_TYPE sigma_12_22_multiply[N_CLIENT][C]; // sigma_12 * sigma_22_inverse [N - m * m]
DATA_TYPE sigma_12_21_multiply[BUFFER_SIZE][N_CLIENT]; // [N - m * N - M]

DATA_TYPE delta_l[N];

//DATA_TYPE sigma_11[N][N - 1][N - 1];
//DATA_TYPE sigma_12[N][N]; // The actual size is N * (N - 1)
//DATA_TYPE sigma_21[N][N - 1];
DATA_TYPE sigma_12[BUFFER_SIZE][N];
DATA_TYPE sigma_22[N];
//DATA_TYPE mu_1[N][N]; // The actual size is N * (N - 1)
DATA_TYPE mu_1[BUFFER_SIZE][N];
DATA_TYPE mu_2[N];
DATA_TYPE mu_bar[BUFFER_SIZE][N];

Data_Sort sort_client_list[N_CLIENT];
//Data_Sort input_sort[SORT_NUM];
Data_Sort output_sort[SORT_NUM];
DATA_TYPE mu_bar_sorted_data[N];
int mu_bar_sorted_index[N];

int selected_client_index_table[C];
int non_selected_client_index_table[N_CLIENT];

DATA_TYPE sigma_11_select[BUFFER_SIZE][N_CLIENT];
DATA_TYPE sigma_12_select[N_CLIENT][C];
DATA_TYPE sigma_21_select[C][N_CLIENT];
DATA_TYPE sigma_22_select[C][C];
DATA_TYPE sigma_22_select_inverse[C][C];
DATA_TYPE sigma_updated[BUFFER_SIZE][N_CLIENT];

DATA_TYPE alpha_mu_2_selected[C];
DATA_TYPE sigma_mu_multiply[N_CLIENT];
DATA_TYPE mu_1_select[N];
DATA_TYPE mu_updated[N_CLIENT];

//bool client_selection_flag[N];

void load_parameters(DATA_TYPE mu_dram[N], DATA_TYPE a_dram[N], DATA_TYPE beta_dram[N] )
{
    for (int i = 0; i < N; i++)
    {
#pragma HLS pipeline II=1
        mu[i] = mu_dram[i];
        
        a[i] = a_dram[i];

        beta[i] = beta_dram[i];
    }

}

void load_sigma_1d(DATA_TYPE sigma_dram[N][N])
{
    for (int i = 0; i < N ;i++)
    {
#pragma HLS pipeline II=1
        sigma_1d[i] = sigma_dram[i][i];
    }
}


void on_chip_buffer_initialization(DATA_TYPE mu_bar_accumu[N], int selected_client_index_table[C], int non_selected_client_index_table[N_CLIENT])
{
     for (int client_index = 0; client_index < N; client_index++)
    {
#pragma HLS pipeline II=1
        mu_bar_accumu[client_index] = 0;
        //client_selection_flag[client_index] = 0;  //initialization, 0 means non client is selected.
    }

    for (int i = 0; i < C;i++)
    {
#pragma HLS pipeline II=1
        selected_client_index_table[i] = 0;
    }

    for (int i = 0; i < N_CLIENT; i++)
    {
#pragma HLS pipeline II=1
        non_selected_client_index_table[i] = 0;
    }

}

void compute_delta_l_kernel(DATA_TYPE mu[N], DATA_TYPE sigma_1d[N], DATA_TYPE a[N], DATA_TYPE beta[N], DATA_TYPE delta_l[N], int index)
{
     
DATA_TYPE mu_buffer[UNROLL_FACTOR];
DATA_TYPE sigma_buffer[UNROLL_FACTOR];
DATA_TYPE a_buffer[UNROLL_FACTOR];
DATA_TYPE beta_buffer[UNROLL_FACTOR];
DATA_TYPE delta_l_buffer[UNROLL_FACTOR];

#pragma HLS ARRAY_PARTITION variable=a_buffer complete dim=1 
#pragma HLS ARRAY_PARTITION variable=beta_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=mu_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=sigma_buffer complete dim=1
#pragma HLS inline off

for (int client_index = 0; client_index < UNROLL_FACTOR; client_index++)
{
#pragma HLS pipeline II = 1
    mu_buffer[client_index] = mu[UNROLL_FACTOR * index + client_index];
    a_buffer[client_index] = a[UNROLL_FACTOR * index + client_index];
    beta_buffer[client_index] = beta[UNROLL_FACTOR * index + client_index];
    sigma_buffer[client_index] = sigma_1d[UNROLL_FACTOR * index + client_index];
}

//#pragma HLS ARRAY_PARTITION variable=temp complete dim=1

     for (int client_index = 0; client_index < UNROLL_FACTOR; client_index++)
    {
        //partition to get mu_1
#pragma HLS unroll
        delta_l_buffer[client_index] = a_buffer[client_index] * beta_buffer[client_index];
        delta_l_buffer[client_index] = mu_buffer[client_index] - delta_l_buffer[client_index] * hls::sqrt(hls::abs(sigma_buffer[client_index]));

    }

    for (int client_index = 0; client_index < UNROLL_FACTOR; client_index++)
    {
#pragma HLS pipeline II = 1
        delta_l[UNROLL_FACTOR * index + client_index] = delta_l_buffer[client_index];
    }
}


void compute_delta_l(DATA_TYPE mu[N], DATA_TYPE sigma_1d[N], DATA_TYPE a[N], DATA_TYPE beta[N], DATA_TYPE delta_l[N])
{

#pragma HLS inline off
     
    for (int index = 0; index < N/UNROLL_FACTOR; index++)
    {
        compute_delta_l_kernel(mu, sigma_1d, a, beta, delta_l, index);
    }
}

void mu_sigma_selection(DATA_TYPE mu[N], DATA_TYPE sigma_dram[N][N], DATA_TYPE mu_1[BUFFER_SIZE][N], DATA_TYPE mu_2[N], DATA_TYPE sigma_12[BUFFER_SIZE][N], DATA_TYPE sigma_1d[N], DATA_TYPE sigma_22[N], int selection_index)

{
// The actual size of mu_1 is N * (N - 1)
// The actual size of sigma_12 is N * (N -1)
 
#pragma HLS inline off
    
    for (int client_index = 0; client_index < BUFFER_SIZE; client_index++)
    {
        for (int i = 0; i < BUFFER_SIZE * selection_index + client_index ; i++)
            {
       #pragma HLS pipeline II = 1
                mu_1[client_index][i] = mu[i];
            }

            for (int i = BUFFER_SIZE * selection_index + client_index + 1; i < N; i++)
            {
       #pragma HLS pipeline II = 1
                mu_1[ client_index][i - 1] = mu[i];
            }

            mu_1[client_index][N - 1] = (DATA_TYPE) 0.0;
            //partition to get mu_2
            mu_2[BUFFER_SIZE * selection_index + client_index] = mu[BUFFER_SIZE * selection_index + client_index];
            
            //partition to get sigma_12
            for (int i = 0; i < BUFFER_SIZE * selection_index + client_index; i++)
            {
       #pragma HLS pipeline II = 1
                sigma_12[ client_index][i] = sigma_dram[i][0];
            }

            for (int i = BUFFER_SIZE * selection_index + client_index + 1; i < N; i++)
            {
        #pragma HLS pipeline II = 1
                sigma_12[ client_index][i - 1] = sigma_dram[i][0];
            }
            sigma_12[client_index][N -1] = (DATA_TYPE) 0.0;

            //partiton to get sigma_22

            //sigma_22[client_index] = (DATA_TYPE) 1.0/sigma[client_index][client_index];
    }

    for (int client_index = 0; client_index < N_CLIENT; client_index++)
    {
    #pragma HLS pipeline II = 1
        sigma_22[client_index] = (DATA_TYPE) 1.0/sigma_1d[client_index];
    }
    for (int client_index = N_CLIENT; client_index < N; client_index++)
    {
    #pragma HLS pipeline II = 1
        sigma_22[client_index] = (DATA_TYPE) 0.0;

    }

    // for (int client_index = 0; client_index < N_CLIENT; client_index++)
    // {
    // #pragma HLS pipeline II = 1
    //     //sigma_22[BUFFER_SIZE * selection_index + client_index] = (DATA_TYPE) 1.0/sigma_1d[BUFFER_SIZE * selection_index + client_index];
    //     sigma_22[client_index]

    // }
    
}

void mu_bar_accumulator(DATA_TYPE mu_bar[BUFFER_SIZE][N], DATA_TYPE mu_bar_accumu[N], int index, int index_outside)
{
    
    DATA_TYPE mu_bar_accumu_buffer[UNROLL_FACTOR];
    DATA_TYPE mu_bar_buffer[UNROLL_FACTOR][N - 1];
    #pragma HLS ARRAY_PARTITION variable=mu_bar_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=mu_bar_accumu_buffer complete dim=1

    for (int client_index = 0; client_index < UNROLL_FACTOR; client_index++)
    {   
        mu_bar_accumu_buffer[client_index] = 0;
        for (int i = 0; i < N - 1; i++)
        {
    #pragma HLS pipeline II = 1
        mu_bar_buffer[client_index][i] = mu_bar[UNROLL_FACTOR * index + client_index][i];
        }
    }
    
    for (int client_index = 0; client_index < UNROLL_FACTOR; client_index++)
    {
    #pragma HLS unroll
    
        for (int i = 0; i < N - 1; i++)
        {
    #pragma HLS pipeline II = 1
        //#pragma HLS unroll
             mu_bar_accumu_buffer[client_index] = mu_bar_accumu_buffer[client_index] +  mu_bar_buffer[client_index][i];
        }
    }

    for (int client_index = 0; client_index < UNROLL_FACTOR; client_index++)
    {
    #pragma HLS pipeline II = 1
        mu_bar_accumu[BUFFER_SIZE * index_outside + UNROLL_FACTOR * index + client_index] = mu_bar_accumu_buffer[client_index];
    }


}


void compute_mu_bar_accumu_kernel(DATA_TYPE mu_1[BUFFER_SIZE][N], DATA_TYPE mu_2[N], DATA_TYPE sigma_12[BUFFER_SIZE][N], DATA_TYPE sigma_22[N], DATA_TYPE delta_l[N], DATA_TYPE mu_bar[BUFFER_SIZE][N],  int index_x, int index_y, int index)
{
    //update the mu_bar
    DATA_TYPE mu_1_buffer[UNROLL_FACTOR][UNROLL_FACTOR];
    DATA_TYPE mu_2_buffer[UNROLL_FACTOR];
    DATA_TYPE sigma_12_buffer[UNROLL_FACTOR][UNROLL_FACTOR];
    DATA_TYPE sigma_22_buffer[UNROLL_FACTOR];
    DATA_TYPE delta_l_buffer[UNROLL_FACTOR];
    DATA_TYPE mu_bar_buffer[UNROLL_FACTOR][UNROLL_FACTOR];
    //DATA_TYPE mu_bar_accumu_buffer[UNROLL_FACTOR];

    #pragma HLS ARRAY_PARTITION variable=mu_1_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=mu_1_buffer complete dim=2
    #pragma HLS ARRAY_PARTITION variable=mu_2_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sigma_12_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sigma_12_buffer complete dim=2
    #pragma HLS ARRAY_PARTITION variable=sigma_22_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=delta_l_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=mu_bar_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=mu_bar_buffer complete dim=2
    //#pragma HLS ARRAY_PARTITION variable=mu_bar_accumu_buffer complete dim=1

    for (int client_index = 0; client_index < UNROLL_FACTOR; client_index++)
    {   
        sigma_22_buffer[client_index] = sigma_22[BUFFER_SIZE * index + UNROLL_FACTOR * index_x + client_index];
        delta_l_buffer[client_index] = delta_l[BUFFER_SIZE * index + UNROLL_FACTOR * index_x + client_index];
        mu_2_buffer[client_index] = mu_2[BUFFER_SIZE * index + UNROLL_FACTOR * index_x + client_index];
        for (int i = 0; i < UNROLL_FACTOR; i++)
        {   
            mu_1_buffer[client_index][i] = mu_1[ UNROLL_FACTOR * index_x + client_index][ UNROLL_FACTOR * index_y + i];
            sigma_12_buffer[client_index][i] = sigma_12[ UNROLL_FACTOR * index_x + client_index][UNROLL_FACTOR * index_y + i];

        }
    }

    for (int client_index = 0; client_index < UNROLL_FACTOR; client_index++)
    {
    #pragma HLS unroll
        for (int i = 0; i < UNROLL_FACTOR; i++)
        {
    #pragma HLS unroll
            mu_bar_buffer[client_index][i] = mu_1_buffer[client_index][i] + sigma_12_buffer[client_index][i] * sigma_22_buffer[client_index] * (delta_l_buffer[client_index] - mu_2_buffer[client_index]);
           
        }
    }

    for (int client_index = 0; client_index < UNROLL_FACTOR; client_index++)
    {
        for (int i = 0; i < UNROLL_FACTOR; i++)
        {
    #pragma HLS pipeline II = 1
            mu_bar[UNROLL_FACTOR * index_x + client_index][UNROLL_FACTOR * index_y + i] = mu_bar_buffer[client_index][i];
        }

    }
    
}




void compute_mu_bar_accumu(DATA_TYPE mu_1[BUFFER_SIZE][N], DATA_TYPE mu_2[N], DATA_TYPE sigma_12[BUFFER_SIZE][N], DATA_TYPE sigma_22[N], DATA_TYPE delta_l[N], DATA_TYPE mu_bar[BUFFER_SIZE][N], DATA_TYPE mu_bar_accumu[N], int index_outside)
{
    //update the mu_bar

    // The actual size of the mu_bar is N * (N -1)

    for (int index_x  = 0; index_x < BUFFER_SIZE/UNROLL_FACTOR; index_x++)
    {
        for (int index_y = 0; index_y < N/UNROLL_FACTOR; index_y++)
        {
            compute_mu_bar_accumu_kernel(mu_1, mu_2, sigma_12, sigma_22, delta_l, mu_bar, index_x, index_y, index_outside);
        }
        mu_bar_accumulator(mu_bar, mu_bar_accumu, index_x, index_outside);
    }
}

void compute_mu_ba_scale(DATA_TYPE mu[N], DATA_TYPE sigma_dram[N][N], DATA_TYPE mu_1[BUFFER_SIZE][N], DATA_TYPE mu_2[N], DATA_TYPE sigma_12[BUFFER_SIZE][N], DATA_TYPE sigma_1d[N], DATA_TYPE sigma_22[N])
{
    for (int index_outside = 0; index_outside < N/BUFFER_SIZE; index_outside++)
    {
        mu_sigma_selection(mu, sigma_dram, mu_1, mu_2, sigma_12, sigma_1d, sigma_22, index_outside);
        compute_mu_bar_accumu(mu_1, mu_2, sigma_12, sigma_22, delta_l, mu_bar, mu_bar_accumu, index_outside);
    }
}

Data_Sort sorting_kernel_2 (Data_Sort input_1, Data_Sort input_2)
{
//#pragma HLS inline off
    if (input_1.number < input_2.number)
    {
        return input_1;   
    }
    else
    {
        return input_2;
    }

}

Data_Sort sorting_kernel_4(Data_Sort input [4])
{
    #pragma HLS ARRAY_PARTITION variable=input complete dim=1
    Data_Sort out_temp[2];
    #pragma HLS ARRAY_PARTITION variable=out_temp complete dim=1
    for (int i = 0; i  < 2; i++)
    {
        #pragma HLS unroll

        out_temp[i] = sorting_kernel_2(input[2 * i], input[2 * i + 1]);
    }

    return sorting_kernel_2(out_temp[0], out_temp[1]);

}

Data_Sort sorting_kernel_8(Data_Sort input[8])
{
    #pragma HLS ARRAY_PARTITION variable=input complete dim=1
    Data_Sort out_temp[4];
    #pragma HLS ARRAY_PARTITION variable=out_temp complete dim=1
    for (int i = 0; i  < 4; i++)
    {
        #pragma HLS unroll

        out_temp[i] = sorting_kernel_2(input[2 * i], input[2 * i + 1]);
    }

    return sorting_kernel_4(out_temp);
}



Data_Sort sorting_kernnel_16 (Data_Sort input[16])
{
    #pragma HLS ARRAY_PARTITION variable=input complete dim=1
    Data_Sort out_temp_1[8];
    #pragma HLS ARRAY_PARTITION variable=out_temp_1 complete dim=1
    Data_Sort out_temp_2[4];
    #pragma HLS ARRAY_PARTITION variable=out_temp_2 complete dim=1
    
    for (int i = 0; i  < 8; i++)
    {
        #pragma HLS unroll

        out_temp_1[i] = sorting_kernel_2(input[2 * i], input[2 * i + 1]);
    }

    for (int i = 0; i  < 4; i++)
    {
        #pragma HLS unroll

        out_temp_2[i] = sorting_kernel_2(out_temp_1[2 * i], out_temp_1[2 * i + 1]);
    }


    return sorting_kernel_4(out_temp_2);
}

Data_Sort sorting_kernel_32 (Data_Sort input[32])
{
    #pragma HLS ARRAY_PARTITION variable=input complete dim=1
    Data_Sort out_temp_1[16];
    #pragma HLS ARRAY_PARTITION variable=out_temp_1 complete dim=1
    Data_Sort out_temp_2[8];
    #pragma HLS ARRAY_PARTITION variable=out_temp_2 complete dim=1
    Data_Sort out_temp_3[4];
    #pragma HLS ARRAY_PARTITION variable=out_temp_3 complete dim=1

    for (int i = 0; i  < 16; i++)
    {
        #pragma HLS unroll

        out_temp_1[i] = sorting_kernel_2(input[2 * i], input[2 * i + 1]);
    }

    for (int i = 0; i  < 8; i++)
    {
        #pragma HLS unroll

        out_temp_2[i] = sorting_kernel_2(out_temp_1[2 * i], out_temp_1[2 * i + 1]);
    }

    for (int i = 0; i  < 4; i++)
    {
        #pragma HLS unroll

        out_temp_3[i] = sorting_kernel_2(out_temp_2[2 * i], out_temp_2[2 * i + 1]);
    }

    return sorting_kernel_4(out_temp_3);
}


Data_Sort sorting_kernel_64 (Data_Sort input[64])

{
    #pragma HLS ARRAY_PARTITION variable=input complete dim=1
    Data_Sort out_temp_1[32];
    #pragma HLS ARRAY_PARTITION variable=out_temp_1 complete dim=1
    Data_Sort out_temp_2[16];
    #pragma HLS ARRAY_PARTITION variable=out_temp_2 complete dim=1
    Data_Sort out_temp_3[8];
    #pragma HLS ARRAY_PARTITION variable=out_temp_3 complete dim=1
    Data_Sort out_temp_4[4];
    #pragma HLS ARRAY_PARTITION variable=out_temp_4 complete dim=1

    for (int i = 0; i  < 32; i++)
    {
        #pragma HLS unroll

        out_temp_1[i] = sorting_kernel_2(input[2 * i], input[2 * i + 1]);
    }

    for (int i = 0; i  < 16; i++)
    {
        #pragma HLS unroll

        out_temp_2[i] = sorting_kernel_2(out_temp_1[2 * i], out_temp_1[2 * i + 1]);
    }

    for (int i = 0; i  < 8; i++)
    {
        #pragma HLS unroll

        out_temp_3[i] = sorting_kernel_2(out_temp_2[2 * i], out_temp_2[2 * i + 1]);
    }

    for (int i = 0; i  < 4; i++)
    {
        #pragma HLS unroll

        out_temp_4[i] = sorting_kernel_2(out_temp_3[2 * i], out_temp_3[2 * i + 1]);
    }

    return sorting_kernel_4(out_temp_4);
   
}

Data_Sort sorting_kernel(Data_Sort sort_input[SORT_NUM])

{
    Data_Sort sort_output;
    DATA_TYPE sort_output_number;
    DATA_TYPE sort_output_index;
    #pragma HLS ARRAY_PARTITION variable=sort_input complete dim=1
#pragma HLS inline off

    if (SORT_NUM == 64)
    {
        sort_output = sorting_kernel_64(sort_input);
    }
   
    if (SORT_NUM == 32)
    {
        sort_output = sorting_kernel_32(sort_input);
    }

    if (SORT_NUM == 16)
    {
        sort_output = sorting_kernnel_16(sort_input);
    }

    if (SORT_NUM == 8)
    {
        sort_output = sorting_kernel_8(sort_input);
    }

    if (SORT_NUM == 4)
    {
        sort_output = sorting_kernel_4(sort_input);
    }

    if (SORT_NUM == 2)
    {
        sort_output = sorting_kernel_2(sort_input[0], sort_input[1]);
    }

    
    return sort_output;
    
}

int sorting(DATA_TYPE mu_bar_accumu[N],  bool client_selection_flag[N_CLIENT], int selected_client_index_table[C], int non_selected_client_index_table[N_CLIENT])
{
    Data_Sort input_sort[SORT_NUM];
    #pragma HLS ARRAY_PARTITION variable=input_sort complete dim=1
    int selected_sorting_client_num = 0;
    Data_Sort sort_output_result;
    Data_Sort output_sort_buffer[SORT_NUM];
    #pragma HLS ARRAY_PARTITION variable=output_sort_buffer complete dim=1
    for (int i = 0; i < N_CLIENT; i++)
    {
    #pragma HLS pipeline II = 1
       if (client_selection_flag[i] == 0)
       {
           sort_client_list[selected_sorting_client_num].number = mu_bar_accumu[i];
           sort_client_list[selected_sorting_client_num].index = i;
           selected_sorting_client_num++;
       }
        
    }

    if (selected_sorting_client_num > SORT_NUM)
    {
        
        int iteration_num = selected_sorting_client_num/SORT_NUM;
        int iter_counter = 0;
        for (int iter = 0; iter < iteration_num; iter++)
        {
                for (int i  = 0; i < SORT_NUM; i++)
                {
                #pragma HLS pipeline II = 1
                    input_sort[i] = sort_client_list[iter * SORT_NUM + i];
                }
                output_sort_buffer[iter] = sorting_kernel(input_sort);
                iter_counter = iter_counter + 1;
        }
        int bound = selected_sorting_client_num - (iter_counter * SORT_NUM);

        for (int i = 0; i < bound; i++)
        {
        #pragma HLS pipeline II = 1
            input_sort[i] = sort_client_list[iter_counter * SORT_NUM + i];
        }

        for (int i = bound; i < SORT_NUM; i++)
        {
        #pragma HLS pipeline II = 1
            input_sort[i].number = 9999;
            input_sort[i].index = 9999;
        }

        output_sort_buffer[iter_counter] = sorting_kernel(input_sort);
        iter_counter = iter_counter + 1;

        for (int i = iter_counter; i < SORT_NUM; i++)
        {
            output_sort_buffer[i].number = 9999;
            output_sort_buffer[i].index = 9999;
        }

        
        sort_output_result = sorting_kernel(output_sort_buffer);
        
    }

    else
    {
        for (int i  = 0; i < selected_sorting_client_num; i++)
        {
            input_sort[i] = sort_client_list[i];
        }

        for (int i = selected_sorting_client_num; i < SORT_NUM; i++)
        {
            input_sort[i].number = 9999;
            input_sort[i].index = 9999;
        }

        sort_output_result = sorting_kernel(input_sort);

    }

    int select_client_index = sort_output_result.index;

    client_selection_flag[select_client_index] = 1;


    int client_selected_num = 0;
    int client_not_selected_num = 0;
    // calculate the total number of clients selected.
    for (int i = 0; i < N_CLIENT; i++)
    {
    #pragma HLS pipeline II = 1
        if (client_selection_flag[i] == 1)
        {
            selected_client_index_table[client_selected_num] = i;
            client_selected_num = client_selected_num + 1;
            
        }

        else
        {
            non_selected_client_index_table[client_not_selected_num] = i;
            client_not_selected_num = client_not_selected_num + 1;
        }

        
    }

    return client_selected_num;

}


// void sorting_kernel(DATA_TYPE mu_bar_accumu[SORT_NUM], DATA_TYPE mu_bar_sorted_data[SORT_NUM], int mu_bar_sorted_index[N])

// {
//     // get the accumulation for each client of the mu_bar
//     bool order = 1;
//     //generate the sorting information

//     #pragma HLS aggregate variable=input_sort
//     #pragma HLS aggregate variable=output_sort
//     // #pragma HLS ARRAY_PARTITION variable=input_sort complete dim=1
//     // #pragma HLS ARRAY_PARTITION variable=output_sort complete dim=1
//     // #pragma HLS ARRAY_PARTITION variable=mu_bar_accumu complete dim=1
//     // #pragma HLS ARRAY_PARTITION variable=mu_bar_sorted_data complete dim=1
//     // #pragma HLS ARRAY_PARTITION variable=mu_bar_sorted_index complete dim=1

//     #pragma HLS inline off

//     for (int i = 0; i < SORT_NUM; i++)
//     {
//     #pragma HLS pipeline II = 1
//         input_sort[i].number = mu_bar_accumu[SORT_NUM * index + i];
//         input_sort[i].index = SORT_NUM * index + i;
//     }
//     // for (int i = 0; i < N; i++)
//     // {
//     // #pragma HLS pipeline II = 1
//     //     input_sort[i].number = mu_bar_accumu[i];
//     //     input_sort[i].index = i;
//     // }

//     // for (int i = N ; i < SORT_NUM; i++)
//     // {
//     // #pragma HLS pipeline II = 1
//     //     input_sort[i].number = 9999;
//     //     input_sort[i].index = i;

//     // }
//     hls::stream <Data_Sort, SORT_NUM> kin_strm;
//     hls::stream <bool, SORT_NUM + 1> kin_strm_end;
//     hls::stream <Data_Sort, SORT_NUM> kout_strm;
//     hls::stream <bool, SORT_NUM + 1> kout_strm_end;
//     #pragma HLS STREAM variable=kin_strm depth=SORT_NUM
//     #pragma HLS STREAM variable=kin_strm_end depth=SORT_NUM + 1
//     #pragma HLS STREAM variable=kout_strm depth=SORT_NUM
//     #pragma HLS STREAM variable=kout_strm_end depth=SORT_NUM + 1

//     load_data(input_sort, kin_strm, kin_strm_end, SORT_NUM);
//     bitonic_sort_top<Data_Sort, SORT_NUM>(kin_strm, kin_strm_end, kout_strm, kout_strm_end, order);
//     store_data(output_sort, kout_strm, kout_strm_end, SORT_NUM);

//     for (int i = 0; i < SORT_NUM; i++)
//     {
//     #pragma HLS pipeline II = 1
//         mu_bar_sorted_data[SORT_NUM * index + i] = output_sort[i].number;
//         mu_bar_sorted_index[SORT_NUM * index + i] = output_sort[i].index;
//     }
//     // for (int i = 0; i < N; i++)
//     // {
//     // //#pragma HLS unroll
//     //     mu_bar_sorted_data[i] = output_sort[i].number;
//     //     mu_bar_sorted_index[i] = output_sort[i].index;
//     // }

// }

// int select_client(bool client_selection_flag[N], int selected_client_index_table[C], int non_selected_client_index_table[N])
// {
//     // if we want to select the client with the minimum value, then select mu_bar_sorted_index[0]
// #pragma HLS inline off
//     for (int i = 0; i < N; i++)
//     {
//     #pragma HLS pipeline II = 1
//         if (client_selection_flag[mu_bar_sorted_index[i]] == 0)
//         {
//              client_selection_flag[mu_bar_sorted_index[i]] = 1;
//              break;
//         }
//     }
    

//     int client_selected_num = 0;
//     int client_not_selected_num = 0;
//     // calculate the total number of clients selected.
//     for (int i = 0; i < N; i++)
//     {
//     #pragma HLS pipeline II = 1
//         if (client_selection_flag[i] == 1)
//         {
//             selected_client_index_table[client_selected_num] = i;
//             client_selected_num = client_selected_num + 1;
            
//         }

//         else
//         {
//             non_selected_client_index_table[client_not_selected_num] = i;
//             client_not_selected_num = client_not_selected_num + 1;
//         }

        
//     }

//     return client_selected_num;
// }

void sigma_selection_new(int client_selected_num, DATA_TYPE sigma_dram[N][N], int selected_client_index_table[C], int non_selected_client_index_table[N_CLIENT] )
{
    
// #pragma HLS ARRAY_PARTITION variable=sigma complete dim=1
// #pragma HLS ARRAY_PARTITION variable=sigma complete dim=2
// #pragma HLS ARRAY_PARTITION variable=sigma_22_select complete dim=1
// #pragma HLS ARRAY_PARTITION variable=sigma_22_select complete dim=2
// #pragma HLS ARRAY_PARTITION variable=sigma_21_select complete dim=1
// #pragma HLS ARRAY_PARTITION variable=sigma_21_select complete dim=2
// #pragma HLS ARRAY_PARTITION variable=sigma_12_select complete dim=1
// #pragma HLS ARRAY_PARTITION variable=sigma_12_select complete dim=2
// #pragma HLS ARRAY_PARTITION variable=sigma_11_select complete dim=1
// #pragma HLS ARRAY_PARTITION variable=sigma_11_select complete dim=2
// #pragma HLS ARRAY_PARTITION variable=selected_client_index_table complete dim=1
#pragma HLS inline off    
    for (int i = 0; i < client_selected_num; i++)
    {
        for (int j = 0; j < client_selected_num; j++)
        {
        #pragma HLS pipeline II = 1
            sigma_22_select[i][j] = sigma_dram[selected_client_index_table[i]][selected_client_index_table[j]];
        }
    }

    for (int i = 0; i < client_selected_num; i++)
    {
        for (int j = 0; j < N_CLIENT - client_selected_num; j++)
        {
        #pragma HLS pipeline II = 1
            sigma_21_select[i][j] = sigma_dram[selected_client_index_table[i]][non_selected_client_index_table[j]];
        }
    }

    for (int i = 0; i < N_CLIENT - client_selected_num; i++)
    {
    
        for (int j = 0; j < client_selected_num; j++)
        {
        #pragma HLS pipeline II = 1
            sigma_12_select[i][j] = sigma_dram[non_selected_client_index_table[i]][selected_client_index_table[j]];
        }
    }

    // for (int i = 0; i < N - client_selected_num; i++)
    // {
    //     for (int j = 0; j < N - client_selected_num; j++)
    //     {
    //     #pragma HLS pipeline II = 1
    //         sigma_11_select[i][j] = sigma[non_selected_client_index_table[i]][non_selected_client_index_table[j]];
    //     }
    // }
}


void sigma_11_selection_new(int client_selected_num, DATA_TYPE sigma_dram[N][N], int index_outside, int total_iter_num, int remaining_iteration)
{
    if (index_outside != total_iter_num - 1)
    {
        for (int i = 0; i < BUFFER_SIZE; i++)
        {
            for (int j = 0; j < N_CLIENT - client_selected_num; j++)
            {
        #pragma HLS pipeline II = 1
                sigma_11_select[i][j] = sigma_dram[non_selected_client_index_table[BUFFER_SIZE * index_outside + i]][non_selected_client_index_table[j]];
            }
        }

    }

    else 

    {
        for (int i = 0; i < remaining_iteration; i++)
        {
            for (int j = 0; j < N_CLIENT - client_selected_num; j++)
            {
        #pragma HLS pipeline II = 1
                sigma_11_select[i][j] = sigma_dram[non_selected_client_index_table[BUFFER_SIZE * index_outside + i]][non_selected_client_index_table[j]];
            }
        }
    }
    


}




// void sigma_selection(int client_selected_num, DATA_TYPE sigma[N][N], bool client_selection_flag[N])
// {
//     int sigma_22_select_dim_i = client_selected_num;
//     int sigma_22_select_dim_j = client_selected_num; // the dimension of sigma_22 is [m * m]
//     int sigma_21_select_dim_i = client_selected_num;
//     int sigma_21_select_dim_j = N - client_selected_num; // the dimension of sigma_21 is [m * N -m]
//     int sigma_12_select_dim_i = N - client_selected_num;
//     int sigma_12_select_dim_j = client_selected_num; // the dimension of sigma_12 is [N - m * m]
//     int sigma_11_select_dim_i = N - client_selected_num;
//     int sigma_11_select_dim_j = N - client_selected_num; // the dimension of sigma_11 is [N - m * N - m]

//     int sigma_22_select_dim_i_counter = 0;
//     int sigma_22_select_dim_j_counter = 0;
//     int sigma_21_select_dim_i_counter = 0;
//     int sigma_21_select_dim_j_counter = 0;
//     int sigma_12_select_dim_i_counter = 0;
//     int sigma_12_select_dim_j_counter = 0;
//     int sigma_11_select_dim_i_counter = 0;
//     int sigma_11_select_dim_j_counter = 0;

//     // generate the sigma matrixs.
//     for (int i = 0; i < N; i++)
//     {
//     #pragma HLS unroll
//         for (int j = 0; j < N ;j++)
//         {
//         #pragma HLS unroll
//             if (client_selection_flag[i] == 1 & client_selection_flag[j] == 1)
//             {
//                 sigma_22_select[sigma_22_select_dim_i_counter][sigma_22_select_dim_j_counter] = sigma[i][j];
//                 sigma_22_select_dim_j_counter++;
//                 if (sigma_22_select_dim_j_counter == sigma_22_select_dim_j)
//                 {
//                     sigma_22_select_dim_j_counter = 0;
//                     sigma_22_select_dim_i_counter++;
//                 }
//             }
//             else if (client_selection_flag[i] == 1 & client_selection_flag[j] == 0)
//             {
//                 sigma_21_select[sigma_21_select_dim_i_counter][sigma_21_select_dim_j_counter] = sigma[i][j];
//                 sigma_21_select_dim_j_counter++;
//                 if (sigma_21_select_dim_j_counter == sigma_21_select_dim_j)
//                 {
//                     sigma_21_select_dim_j_counter = 0;
//                     sigma_21_select_dim_i_counter++;
//                 }
//             }
//             else if (client_selection_flag[i] == 0 & client_selection_flag[j] == 1)
//             {
//                 sigma_12_select[sigma_12_select_dim_i_counter][sigma_12_select_dim_j_counter] = sigma[i][j];
//                 sigma_12_select_dim_j_counter++;
//                 if (sigma_12_select_dim_j_counter == sigma_12_select_dim_j)
//                 {
//                     sigma_12_select_dim_j_counter = 0;
//                     sigma_12_select_dim_i_counter++;
//                 }
//             }
//             else if (client_selection_flag[i] == 0 & client_selection_flag[j] == 0)
//             {
//                 sigma_11_select[sigma_11_select_dim_i_counter][sigma_11_select_dim_j_counter] = sigma[i][j];
//                 sigma_11_select_dim_j_counter++;
//                 if (sigma_11_select_dim_j_counter == sigma_11_select_dim_j)
//                 {
//                     sigma_11_select_dim_j_counter = 0;
//                     sigma_11_select_dim_i_counter++;
//                 }
//             }
//         }
//     }
// }

// int inverseMatrix(DATA_TYPE mat[C][C], DATA_TYPE inv[C][C], int matrix_dim) {

// #pragma HLS inline off
// #pragma HLS ARRAY_PARTITION variable=mat complete dim=1
// #pragma HLS ARRAY_PARTITION variable=mat complete dim=2
// #pragma HLS ARRAY_PARTITION variable=inv complete dim=1
// #pragma HLS ARRAY_PARTITION variable=inv complete dim=2

//     DATA_TYPE temp;

//     //Initialize the inverse matrix as the identity matrix
//     for (int i = 0; i < matrix_dim; i++) {
//     #pragma HLS unroll
//         for (int j = 0; j < matrix_dim; j++) {
//     #pragma HLS unroll
//             if (i == j)
//                 inv[i][j] = 1.0;
//             else
//                 inv[i][j] = 0.0;
//         }
//     }
//     // Perform Gauss-Jordan elimination
//     for (int i = 0; i < matrix_dim; i++) {
//     #pragma HLS unroll
//         if (mat[i][i] == 0.0)
//             return 0; // Matrix is not invertible

//         // Scale the current row
//         temp = mat[i][i];
//         for (int j = 0; j < matrix_dim; j++) {
//         //#pragma HLS pipeline II = 1
//         #pragma HLS unroll
//             mat[i][j] /= temp;
//             inv[i][j] /= temp;
//         }

//         // Eliminate other rows
//         for (int j = 0; j < matrix_dim; j++) {
//             if (i != j) {
//                 temp = mat[j][i];
//                 for (int k1 = 0; k1 < i+ 1; k1++)  // k1 < i+1
//                 {
//                 #pragma HLS unroll
//                     inv[j][k1] -= temp * inv[i][k1];
//                 }
//                 for (int k2 = i; k2 < matrix_dim; k2++) {
//                 #pragma HLS unroll
//                     mat[j][k2] -= temp * mat[i][k2];
//                     //inv[j][k] -= temp * inv[i][k];
//                 }
//             }
//         }
//     }

//     return 1; // Matrix is invertible
// }


void inverse_matrix_new(DATA_TYPE mat[C][C], DATA_TYPE inv[C][C], int matrix_dim) 
{
    inv[0][0] = (DATA_TYPE) 1.0/mat[0][0];
    DATA_TYPE c_matrix[C];
    DATA_TYPE b_matrix[C];
    for (int iter = 1; iter < matrix_dim; iter++)
    {
        DATA_TYPE d = mat[iter][iter];

        for (int i = 0; i < iter; i++)
        {
        #pragma HLS pipeline II = 1
            c_matrix[i] = mat[iter][i];
            b_matrix[i] = mat[i][iter];
        }

        DATA_TYPE c_a_inverse[C];

        for (int i = 0; i < iter; i++)
        {
        #pragma HLS pipeline II = 1
            c_a_inverse[i] = 0;
        }

        for (int i = 0; i < iter; i++)
        {
            for (int j = 0; j < iter; j++)
            {
         #pragma HLS pipeline II = 1
                c_a_inverse[i] += c_matrix[j] * inv[j][i];
            }
            
        }

        DATA_TYPE b = 0;
        for (int i = 0; i < iter; i++)
        {
        #pragma HLS pipeline II = 1
            b = b + c_a_inverse[i] * b_matrix[i];
        }

        DATA_TYPE t;

        t = (DATA_TYPE)1.0/ (d - b);

        //printf("the value of t is:%f\n", t);

        DATA_TYPE a_inverse_b[C];

        for (int i = 0; i < iter; i++)
        {
        #pragma HLS pipeline II = 1
            a_inverse_b[i] = 0;
        }

        for (int i = 0; i < iter; i++)
        {
            for (int j = 0; j < iter; j++)
            {
        #pragma HLS pipeline II = 1
                a_inverse_b[i] += inv[i][j] * b_matrix[j];
            }
            
        }

        DATA_TYPE Y[C];

        for (int i = 0; i < iter; i++)
        {
        #pragma HLS pipeline II = 1
            Y[i] = -a_inverse_b[i] * t;
        }

        // for (int i = 0; i < 2; i++)
        // {
        //     printf("the value of Y is:%f\n", Y[i]);
        // }

        // DATA_TYPE c_a_inverse[C];
        // for (int i = 0; i < iter; i++)
        // {
        //     for (int j = 0; j < iter; j++)
        //     {
        //         c_a_inverse[i] += c_matrix[j] * inv[j][i];
        //     }
        // }

        DATA_TYPE Z[C];

        for (int i = 0; i < iter; i++)
        {
        #pragma HLS pipeline II = 1
            Z[i] = -c_a_inverse[i] * t;
        }

        //  for (int i = 0; i < 2; i++)
        // {
        //     printf("the value of Z is:%f\n", Z[i]);
        // }


        DATA_TYPE BZ[C][C];

        for (int i = 0; i < iter; i++)
        {
            for (int j = 0; j < iter; j++)
            {
        #pragma HLS pipeline II = 1
                BZ[i][j]  =  b_matrix[i] * Z[j];
            }
        }

        DATA_TYPE I_minus_BZ[C][C];

        for (int i = 0; i < C; i++)
        {
            for (int j = 0; j < C; j++)
            {
        #pragma HLS pipeline II = 1
                if (i == j)
                {
                    I_minus_BZ[i][j] = 1;
                }
                else 
                {
                    I_minus_BZ[i][j] = 0;
                }
            }
        }

        for (int i = 0; i < iter; i++)
        {
            for (int j = 0; j < iter; j++)
            {
        #pragma HLS pipeline II = 1
                I_minus_BZ[i][j] = I_minus_BZ[i][j] - BZ[i][j];
            }
        }

        DATA_TYPE I_minus_BZ_A_inverse[C][C];

        for (int i = 0; i < iter; i++)
        {
            for (int j = 0; j < iter; j++)
            {
        #pragma HLS pipeline II = 1
                I_minus_BZ_A_inverse[i][j] = 0;
            }
        }

        for (int i = 0; i < iter; i++)
        {
            for (int j = 0; j < iter; j++)
            {
                for (int k = 0; k < iter; k++)
                {
            #pragma HLS pipeline II = 1
                    I_minus_BZ_A_inverse[i][j] += inv[i][k] * I_minus_BZ[k][j];
                }
            }
        }

        // Theen we need to update the inverse matrix
        for (int i = 0; i < iter; i++)
        {
            for (int j = 0; j < iter; j++)
            {
        #pragma HLS pipeline II = 1
                inv[i][j] = I_minus_BZ_A_inverse[i][j];
            }
        }

        for (int i = 0; i < iter; i++)
        {
    #pragma HLS pipeline II = 1
            inv[i][iter] = Y[i];
        }

        for (int i = 0; i < iter; i++)
        {
    #pragma HLS pipeline II = 1
            inv[iter][i] = Z[i];
        }

        inv[iter][iter] = t;

    }

}


void compute_sigma_12_21_multiply(DATA_TYPE sigma_12_21_multiply[BUFFER_SIZE][N_CLIENT], DATA_TYPE sigma_12_22_multiply[N_CLIENT][C], DATA_TYPE sigma_21_select[C][N_CLIENT], int client_selected_num, int index_outside, int iter_total_num, int iter_remaining)
{
    
    for (int i = 0; i < BUFFER_SIZE;i++)
        {
            for (int j = 0; j < N_CLIENT; j++)
            {
    #pragma HLS pipeline II=1
                sigma_12_21_multiply[i][j] = 0;
            }
        }
    
    if (index_outside != iter_total_num -1 )
    {
        for (int i = 0; i < BUFFER_SIZE; i++)
        {
                for (int j = 0; j < N_CLIENT - client_selected_num; j++)
                {
            //#pragma HLS pipeline II = 1
                    for (int k = 0; k < client_selected_num; k++)
                    {
                    #pragma HLS pipeline II = 1
                        sigma_12_21_multiply[i][j] += sigma_12_22_multiply[BUFFER_SIZE * index_outside + i][k] * sigma_21_select[k][j];
                    }
                }
        }
    }

    else
    {
         for (int i = 0; i < iter_remaining; i++)
        {
                for (int j = 0; j < N_CLIENT - client_selected_num; j++)
                {
            //#pragma HLS pipeline II = 1
                    for (int k = 0; k < client_selected_num; k++)
                    {
                    #pragma HLS pipeline II = 1
                        sigma_12_21_multiply[i][j] += sigma_12_22_multiply[BUFFER_SIZE * index_outside + i][k] * sigma_21_select[k][j];
                    }
                }
        }
    }
    
}

void compute_sigma_updated(DATA_TYPE sigma_11_select[BUFFER_SIZE][N_CLIENT], DATA_TYPE sigma_12_21_multiply[BUFFER_SIZE][N_CLIENT], DATA_TYPE sigma_updaated[BUFFER_SIZE][N_CLIENT], int client_selected_num, int index_outside, int iter_total_num, int iter_remaining)
{
    if (index_outside != iter_total_num - 1)
    {
        for (int i = 0; i < BUFFER_SIZE; i++)
        {
        
            for (int j = 0; j < N_CLIENT - client_selected_num; j++)
            {
            #pragma HLS pipeline II = 1
                sigma_updated[i][j] = sigma_11_select[i][j] - sigma_12_21_multiply[i][j];
            }
        }
    }

    else

    {
        for (int i = 0; i < iter_remaining; i++)
        {
        
            for (int j = 0; j < N_CLIENT - client_selected_num; j++)
            {
            #pragma HLS pipeline II = 1
                sigma_updated[i][j] = sigma_11_select[i][j] - sigma_12_21_multiply[i][j];
            }
        }
    }
    

}

void compute_sigma_12_22_multiply(DATA_TYPE sigma_12_select[N_CLIENT][C], DATA_TYPE sigma_22_select_inverse[C][C], DATA_TYPE sigma_12_22_multiply[N_CLIENT][C], int client_selected_num)
{
     for (int i = 0; i < N_CLIENT;i++)
    {
        for (int j = 0; j < client_selected_num; j++)
        {
#pragma HLS pipeline II=1
            sigma_12_22_multiply[i][j] = 0;
        }
    }
     
     for (int i = 0; i < N_CLIENT - client_selected_num; i++)
    {
        for (int j = 0; j < client_selected_num; j++)
        {
    //#pragma HLS pipeline II = 1
            for (int k = 0; k < client_selected_num; k++)
            {
        //#pragma HLS unroll
        #pragma HLS pipeline II = 1
                sigma_12_22_multiply[i][j] += sigma_12_select[i][k] * sigma_22_select_inverse[k][j];
            }
        }
    }
}

// void compute_sigma_updated(DATA_TYPE sigma_11_select[N][N], DATA_TYPE sigma_12_select[N][C], DATA_TYPE sigma_22_select_inverse[C][C], DATA_TYPE sigma_12_22_multiply[N][C], DATA_TYPE sigma_12_21_multiply[N][N], DATA_TYPE sigma_updated[N][N], int client_selected_num)
// {
    
// // #pragma HLS ARRAY_PARTITION variable=sigma_11_select complete dim=1
// // #pragma HLS ARRAY_PARTITION variable=sigma_11_select complete dim=2
// // #pragma HLS ARRAY_PARTITION variable=sigma_12_select complete dim=1
// // #pragma HLS ARRAY_PARTITION variable=sigma_12_select complete dim=2
// // #pragma HLS ARRAY_PARTITION variable=sigma_21_select complete dim=1
// // #pragma HLS ARRAY_PARTITION variable=sigma_21_select complete dim=2
// // #pragma HLS ARRAY_PARTITION variable=sigma_22_select_inverse complete dim=1
// // #pragma HLS ARRAY_PARTITION variable=sigma_22_select_inverse complete dim=2
// // #pragma HLS ARRAY_PARTITION variable=sigma_12_22_multiply complete dim=1
// // #pragma HLS ARRAY_PARTITION variable=sigma_12_22_multiply complete dim=2
// // #pragma HLS ARRAY_PARTITION variable=sigma_12_21_multiply complete dim=1
// // #pragma HLS ARRAY_PARTITION variable=sigma_12_21_multiply complete dim=2
// // #pragma HLS ARRAY_PARTITION variable=sigma_updated complete dim=1
// // #pragma HLS ARRAY_PARTITION variable=sigma_updated complete dim=2
// #pragma HLS inline off

//     for (int i = 0; i < N - client_selected_num; i++)
//     {
//         for (int j = 0; j < client_selected_num; j++)
//         {
//     //#pragma HLS pipeline II = 1
//             for (int k = 0; k < client_selected_num; k++)
//             {
//         //#pragma HLS unroll
//         #pragma HLS pipeline II = 1
//                 sigma_12_22_multiply[i][j] += sigma_12_select[i][k] * sigma_22_select_inverse[k][j];
//             }
//         }
//     }

//     for (int i = 0; i < N - client_selected_num; i++)
//     {
//         for (int j = 0; j < N - client_selected_num; j++)
//         {
//     //#pragma HLS pipeline II = 1
//             for (int k = 0; k < client_selected_num; k++)
//             {
//             #pragma HLS pipeline II = 1
//                 sigma_12_21_multiply[i][j] += sigma_12_22_multiply[i][k] * sigma_21_select[k][j];
//             }
//         }
//     }

//     for (int i = 0; i < N - client_selected_num; i++)
//     {
    
//         for (int j = 0; j < N - client_selected_num; j++)
//         {
//          #pragma HLS pipeline II = 1
//             sigma_updated[i][j] = sigma_11_select[i][j] - sigma_12_21_multiply[i][j];
//         }
//     }
// }




void compute_mu_updated (DATA_TYPE mu[N], DATA_TYPE mu_1_select[N_CLIENT], DATA_TYPE mu_2[N],  DATA_TYPE delta_l[N], DATA_TYPE mu_updated[N_CLIENT], DATA_TYPE sigma_12_22_multiply[N_CLIENT][C], DATA_TYPE alpha_mu_2_selected[C], int client_selected_num, int selected_client_index_table[C], int non_selected_client_index_table[N_CLIENT])
{
    // #pragma HLS ARRAY_PARTITION variable=mu complete dim=1
    // #pragma HLS ARRAY_PARTITION variable=mu_1_select complete dim=1
    // #pragma HLS ARRAY_PARTITION variable=mu_2 complete dim=1
    // #pragma HLS ARRAY_PARTITION variable=delta_l complete dim=1
    // #pragma HLS ARRAY_PARTITION variable=mu_updated complete dim=1
    // #pragma HLS ARRAY_PARTITION variable=sigma_12_22_multiply complete dim=1
    // #pragma HLS ARRAY_PARTITION variable=sigma_12_22_multiply complete dim=2
    // #pragma HLS ARRAY_PARTITION variable=alpha_mu_2_selected complete dim=1
    // #pragma HLS ARRAY_PARTITION variable=selected_client_index_table complete dim=1
    // #pragma HLS ARRAY_PARTITION variable=non_selected_client_index_table complete dim=1
    #pragma HLS inline off
    

    for (int i = 0; i < client_selected_num;i++)
    {
    #pragma HLS pipeline II = 1
        alpha_mu_2_selected[i] = delta_l[selected_client_index_table[i]] - mu_2[selected_client_index_table[i]];
    }

    // for (int i =  0; i < client_selected_num; i++)
    // {
    //     printf("alpha_mu_2_selected[%d] = %f\n", i, alpha_mu_2_selected[i]);
    // }

    for (int i = 0; i < N_CLIENT - client_selected_num; i++)
    {
    #pragma HLS pipeline II = 1
        mu_1_select[i] = mu[non_selected_client_index_table[i]];
    }

   

    for (int i = 0; i < N_CLIENT - client_selected_num; i++)
    {
    //#pragma HLS pipeline II = 1
        for (int j = 0; j < client_selected_num; j++)
        {
    #pragma HLS pipeline II = 1
            sigma_mu_multiply[i] = sigma_12_22_multiply[i][j] * alpha_mu_2_selected[j];
        }
    }


    for (int i = 0; i < N_CLIENT - client_selected_num; i++)
    {
     #pragma HLS pipeline II = 1
        mu_updated[i] = mu_1_select[i] + sigma_mu_multiply[i];
    }

    
}

void update_sigma(DATA_TYPE sigma_dram[N][N], DATA_TYPE sigma_updated[BUFFER_SIZE][N_CLIENT], int non_selected_client_index_table[N_CLIENT], int client_selected_num, int index_outside, int iter_total_num, int iter_remaining)
{

    if (index_outside != iter_total_num - 1)
    {
         for (int i = 0; i < BUFFER_SIZE; i++)
            {
                for (int j = 0; j < N_CLIENT - client_selected_num; j++)
                {

                #pragma HLS pipeline II = 1
                    sigma_dram[non_selected_client_index_table[BUFFER_SIZE * index_outside + i]][non_selected_client_index_table[j]] = sigma_updated[i][j]/SCALE;
                }

            }
    }

    else 

    {

        for (int i = 0; i < iter_remaining; i++)
            {
                for (int j = 0; j < N_CLIENT - client_selected_num; j++)
                {

                #pragma HLS pipeline II = 1
                    sigma_dram[non_selected_client_index_table[BUFFER_SIZE * index_outside + i]][non_selected_client_index_table[j]] = sigma_updated[i][j]/SCALE;
                }

            }
    }

}



void update_sigma_scale(int client_selected_num, DATA_TYPE sigma_dram[N][N], DATA_TYPE sigma_updated[N_CLIENT][N_CLIENT], int selected_client_index_table[C], int non_selected_client_index_table[N_CLIENT])
{
    
    int total_client_count = N_CLIENT - client_selected_num;
    int iter_total_num = total_client_count / BUFFER_SIZE + 1;
    int iter_remaining = total_client_count - (iter_total_num - 1) * BUFFER_SIZE;

    //void compute_sigma_12_21_multiply(DATA_TYPE sigma_12_21_multiply[BUFFER_SIZE][N], DATA_TYPE sigma_12_22_multiply[N][C], DATA_TYPE sigma_21_select[C][C], int client_selected_num, int index_outside, int iter_total_num, int iter_remaining)
    for (int index_outside = 0; index_outside < iter_total_num; index_outside++)
    {
    
    sigma_11_selection_new(client_selected_num, sigma_dram, index_outside, iter_total_num, iter_remaining);
    compute_sigma_12_21_multiply(sigma_12_21_multiply, sigma_12_22_multiply, sigma_21_select, client_selected_num, index_outside, iter_total_num, iter_remaining);
    compute_sigma_updated(sigma_11_select, sigma_12_21_multiply, sigma_updated, client_selected_num, index_outside, iter_total_num, iter_remaining);
    update_sigma(sigma_dram, sigma_updated, non_selected_client_index_table, client_selected_num, index_outside, iter_total_num, iter_remaining);
    }
    
}

// void update_mu_sigma(DATA_TYPE mu[N], DATA_TYPE sigma[N][N], DATA_TYPE mu_updated[N], DATA_TYPE sigma_updated[N][N], int client_selected_num, int non_selected_client_index_table[N])
// {
   
//     // #pragma HLS ARRAY_PARTITION variable=mu complete dim=1
//     // #pragma HLS ARRAY_PARTITION variable=sigma complete dim=1
//     // #pragma HLS ARRAY_PARTITION variable=sigma complete dim=2
//     // #pragma HLS ARRAY_PARTITION variable=mu_updated complete dim=1
//     // #pragma HLS ARRAY_PARTITION variable=sigma_updated complete dim=1
//     // #pragma HLS ARRAY_PARTITION variable=sigma_updated complete dim=2
//     // #pragma HLS ARRAY_PARTITION variable=non_selected_client_index_table complete dim=1
//     #pragma HLS inline off

//     for (int i = 0; i < N - client_selected_num; i++)
//     {
//     #pragma HLS pipeline II = 1
//         mu[non_selected_client_index_table[i]] = mu_updated[i];
//     }

//     for (int i = 0; i < N - client_selected_num; i++)
//     {
    
//         for (int j = 0; j < N - client_selected_num; j++)
//         {
//         #pragma HLS pipeline II = 1
//             sigma[non_selected_client_index_table[i]][non_selected_client_index_table[j]] = sigma_updated[i][j];
//         }
//     }

// }

void update_mu(DATA_TYPE mu[N], DATA_TYPE mu_updated[N_CLIENT],int non_selected_client_index_table[N_CLIENT], int client_selected_num )
{
#pragma HLS inline off

    for (int i = 0; i < N_CLIENT - client_selected_num; i++)
    {
    #pragma HLS pipeline II = 1
        mu[non_selected_client_index_table[i]] = mu_updated[i]/SCALE;
    }

    
}

void write_back(DATA_TYPE result[N][N], DATA_TYPE result_dram[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
    #pragma HLS pipeline ii = 1
             result_dram[i][j] = result[i][j];
        } 
        
    }
}

void write_back_mu(DATA_TYPE result[N], DATA_TYPE result_dram[N])
{
    for (int i = 0; i < N; i++)
    {
    #pragma HLS pipeline II = 1
        
        result_dram[i] = result[i];
        
        
    }
}

	void write_back_index_table(int client_index_list[C], int selected_client_index_table[C])
{
    for (int i = 0; i < C; i++)
    {
    #pragma HLS pipeline II = 1
        client_index_list[i] = selected_client_index_table[i];
    }

    // for (int i = C; i < N; i++)
    // {
    //     client_index_list[i] = 100;
    // }
}


void client_selection(DATA_TYPE sigma_dram[N][N], DATA_TYPE mu[N],  DATA_TYPE a[N], DATA_TYPE beta[N], int client_number)
{
   
    for (int i = 0; i < N_CLIENT; i++)
    {
    #pragma HLS pipeline II = 1
        client_selection_flag[i] = 0;
    }
    
    for (int iter = 0; iter < client_number; iter++)
    {
        //on_chip_buffer_initialization(mu_bar_accumu,  sigma_12_22_multiply, sigma_12_21_multiply, selected_client_index_table, non_selected_client_index_table);
        on_chip_buffer_initialization(mu_bar_accumu, selected_client_index_table, non_selected_client_index_table);
        load_sigma_1d(sigma_dram);
        compute_delta_l(mu, sigma_1d, a, beta, delta_l);

    //   for (int i = 0; i < N; i++)
    //     {
    //         printf("the current iter is:%d, the value of delta_l is: %f\n", iter, delta_l[i]);
    //     }
        //mu_sigma_selection(mu, sigma, mu_1, mu_2, sigma_12, sigma_22);
        //compute_mu_bar_accumu(mu_1, mu_2, sigma_12, sigma_22, delta_l, mu_bar, mu_bar_accumu);

        //void compute_mu_ba_scale(DATA_TYPE mu[N], DATA_TYPE sigma_dram[N][N], DATA_TYPE mu_1[BUFFER_SIZE][N], DATA_TYPE mu_2[N], DATA_TYPE sigma_12[BUFFER_SIZE][N], DATA_TYPE sigma_1d[N], DATA_TYPE sigma_22[N])
        compute_mu_ba_scale(mu, sigma_dram, mu_1, mu_2, sigma_12, sigma_1d, sigma_22);
        //compute_mu_ba_scale(mu, sigma, delta_l, mu_bar_accumu);
        //sorting(mu_bar_accumu, mu_bar_sorted_data, mu_bar_sorted_index);
        int client_selected_num = 0;
        //client_selected_num = select_client(client_selection_flag, selected_client_index_table, non_selected_client_index_table);
        //DATA_TYPE mu_bar_accumu[N],  bool client_selection_flag[N_CLIENT], int selected_client_index_table[C], int non_selected_client_index_table[N_CLIENT]
        client_selected_num = sorting(mu_bar_accumu, client_selection_flag, selected_client_index_table, non_selected_client_index_table);
        // for (int i = 0; i < N; i++)
        // {
        
        //     printf("the current iter is: %d, the value of accumu is: %f\n", iter, mu_bar_accumu[i]);
        // }
        //sigma_selection(client_selected_num, sigma, client_selection_flag);
        sigma_selection_new(client_selected_num, sigma_dram, selected_client_index_table, non_selected_client_index_table);
        int inverse_flag = 0;
        inverse_matrix_new(sigma_22_select, sigma_22_select_inverse, client_selected_num);

        compute_sigma_12_22_multiply(sigma_12_select, sigma_22_select_inverse, sigma_12_22_multiply, client_selected_num);
        compute_mu_updated(mu, mu_1_select, mu_2, delta_l, mu_updated, sigma_12_22_multiply, alpha_mu_2_selected, client_selected_num, selected_client_index_table, non_selected_client_index_table);
        update_mu(mu, mu_updated, non_selected_client_index_table, client_selected_num);


        
        update_sigma_scale(client_selected_num, sigma_dram, sigma_updated, selected_client_index_table, non_selected_client_index_table);


        // compute_sigma_updated(sigma_11_select, sigma_12_select, sigma_22_select_inverse, sigma_12_22_multiply, sigma_12_21_multiply, sigma_updated, client_selected_num);
        // compute_mu_updated(mu, mu_1_select, mu_2, delta_l, mu_updated, sigma_12_22_multiply, alpha_mu_2_selected, client_selected_num, selected_client_index_table, non_selected_client_index_table);
        // update_mu_sigma(mu, sigma, mu_updated, sigma_updated, client_selected_num, non_selected_client_index_table);

    }
    
}



void GP_kernel_fixed(DATA_TYPE *mu_dram,  DATA_TYPE sigma_dram[N][N], DATA_TYPE *a_dram, DATA_TYPE *beta_dram, int client_number_array[10], int client_index_list[C], DATA_TYPE result_dram[N][N], DATA_TYPE result_dram_2[N])

{
    #pragma HLS interface m_axi port=mu_dram depth=200 bundle = gmem1
    #pragma HLS interface m_axi port=sigma_dram depth=200 bundle = gmem1
    #pragma HLS interface m_axi port=a_dram depth=200 bundle = gmem1
    #pragma HLS interface m_axi port=beta_dram depth=200 bundle = gmem1
    #pragma HLS interface m_axi port=client_index_list depth=200 bundle = gmem3
    //#pragma HLS interface s_axilite port=client_number
    #pragma HLS interface s_axilite port=return
    #pragma HLS interface m_axi port=client_number_array depth=200 bundle = gmem3

    #pragma HLS interface m_axi port=result_dram depth=200 bundle = gmem2
    #pragma HLS interface m_axi port=result_dram_2 depth=200 bundle = gmem2


    int client_number = client_number_array[0];
    load_parameters(mu_dram, a_dram, beta_dram);
    client_selection(sigma_dram, mu, a, beta, client_number);
    //write_back(sigma, result_dram);
    write_back_mu(mu, result_dram_2);
    write_back_index_table(client_index_list, selected_client_index_table);



}



//int main()
//
//{
//
//    // Seed the random number generator with the current time
//    srand(333);
//
//    // Define the size of the array
//    int arraySize = N_CLIENT; // Change this to the desired size
//    int client_number = C; //The maximum number of clients to be selected.
//
//    // Create an array to store the random numbers
//    DATA_TYPE mu_dram[N];
//    DATA_TYPE a_dram[N];
//    DATA_TYPE sigma_dram[N][N];
//    DATA_TYPE beta_dram[N];
//
//    int client_index_list[C];
//
//    DATA_TYPE result_dram[N][N];
//    DATA_TYPE result_dram_2[N];
//
//
//    float mu_float[N_CLIENT];
//    float a_float[N_CLIENT];
//    float sigma_float[N_CLIENT][N_CLIENT];
//    float beta_float[N_CLIENT];
//    float result_float[N_CLIENT][N_CLIENT];
//    float result_float_2[N_CLIENT];
//    int client_number_array[10];
//    client_number_array[0] = client_number;
//
//
//
//    // Generate random numbers in the range [0, 1)
//    for (int i = 0; i < arraySize; i++) {
//        mu_float[i] = (float)rand() / RAND_MAX;
//        printf("the value of mu_float is: %f\n", mu_float[i]);
//
//    }
//
//    for (int i = 0; i < arraySize; i++) {
//        a_float[i] = (float)rand() / RAND_MAX;
//
//    }
//
//    for (int i = 0; i < arraySize; i++) {
//        beta_float[i] = (float)rand() / RAND_MAX;
//
//    }
//
//    for (int i = 0; i < arraySize; i++) {
//        for (int j = 0; j < arraySize; j++)
//        {
//            sigma_float[i][j] = (float)rand() / RAND_MAX;
//        }
//
//    }
//
//    for (int i = 0; i < N; i++)
//    {
//        mu_dram[i] = 0;
//        a_dram[i] = 0;
//        beta_dram[i] = 0;
//        for (int j = 0; j < N; j++)
//        {
//            sigma_dram[i][j] = 0;
//        }
//    }
//
//
//    for (int i = 0; i < arraySize; i++)
//    {
//        mu_dram[i] = (DATA_TYPE) mu_float[i];
//        a_dram[i] = (DATA_TYPE) a_float[i];
//        beta_dram[i] = (DATA_TYPE) beta_float[i];
//        for (int j = 0; j < arraySize; j++)
//        {
//            sigma_dram[i][j] = (DATA_TYPE) sigma_float[i][j];
//        }
//    }
//
//
//
//
//    GP_kernel(mu_dram, sigma_dram, a_dram, beta_dram, client_number_array, client_index_list, result_dram, result_dram_2);
//
//    // Print the generated array
//    // printf("Random numbers in the range [0, 1):\n");
//    // printf("mu_dram:\n");
//    // for (int i = 0; i < arraySize; i++) {
//    //     printf("%.6f\n", mu[i]);
//    // }
//    // printf("a_dram:\n");
//    // for (int i = 0; i < arraySize; i++) {
//    //     printf("%.6f\n", a[i]);
//    // }
//    // printf("beta_dram:\n");
//    // for (int i = 0; i < arraySize; i++) {
//    //     printf("%.6f\n", beta[i]);
//    // }
//    // printf("sigma_dram:\n");
//    // for (int i = 0; i < arraySize; i++) {
//    //     for (int j = 0; j < arraySize - 1; j++)
//    //     {
//    //         printf("%.6f\n", sigma_12[i][j]);
//    //     }
//
//    // }
//
//    // printf("delta_l:\n");
//    // for (int i = 0; i < arraySize; i++) {
//    //     printf("%d\n", non_selected_client_index_table[i]);
//    // }
//
//    for (int i = 0; i < arraySize; i++)
//    {
//        result_float_2[i] = (float) result_dram_2[i];
//        for (int j = 0; j < arraySize; j++)
//        {
//            result_float[i][j] = (float) result_dram[i][j];
//        }
//
//    }
//
//   printf("client_index_list:\n");
//     for (int i = 0; i < C; i++) {
//         printf("%d\n", client_index_list[i]);
//     }
//
//
//    printf("mu updated:\n");
//    for (int i = 0; i < arraySize; i++) {
//        printf("%.6f\n", result_float_2[i]);
//    }
//    // printf("sigma_22:\n");
//    // for (int i = 0; i < arraySize; i++) {
//    //     for (int j = 0; j < arraySize; j++)
//    //     {
//    //         printf("%.6f\n", sigma_dram[i][j]);
//    //     }
//
//    // }
//
//    return 0;
//}
