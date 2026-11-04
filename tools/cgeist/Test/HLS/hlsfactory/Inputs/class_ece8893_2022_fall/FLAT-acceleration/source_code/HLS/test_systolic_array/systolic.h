#ifndef SYSTOLIC_H_
#define SYSTOLIC_H_

#include <assert.h>
#include <stdint.h>
#include <iostream>

#define ROW 64
#define COLUMN 64

// Dimension of the inner "pseudo" SA for each actual PE
#define SYSTOLIC_DIM 16

typedef uint32_t data_t;

void systolic_array(data_t query_matrix[ROW][COLUMN], data_t key_matrix[ROW][COLUMN],
                 data_t value_matrix[ROW][COLUMN], data_t bias_matrix[ROW][COLUMN], data_t output[ROW][COLUMN]);
void computeLogit(data_t query_matrix[ROW][COLUMN], data_t key_matrix[ROW][COLUMN],
					data_t bias_matrix[ROW][COLUMN], data_t logit[ROW][COLUMN]);
void computeAttention(data_t logit[ROW][COLUMN], data_t value_matrix[ROW][COLUMN], data_t output[ROW][COLUMN]);


#endif
