#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "support.h"

// Fixed parameters
#define input_dimension  13
#define possible_outputs  3
#define training_sets   163
#define nodes_per_layer  64
#define layers            2
#define learning_rate  0.01
#define epochs            1
#define test_sets        15
#define norm_param    0.005

#define max 1.0
#define offset 0.5

//Data Bounds
#define TYPE double
#define MAX 1000
#define MIN 1

void backprop(
    TYPE weights1[input_dimension*nodes_per_layer],
    TYPE weights2[nodes_per_layer*nodes_per_layer],
    TYPE weights3[nodes_per_layer*possible_outputs],
    TYPE biases1[nodes_per_layer],
    TYPE biases2[nodes_per_layer],
    TYPE biases3[possible_outputs],
    TYPE training_data[training_sets*input_dimension],
    TYPE training_targets[training_sets*possible_outputs]);
////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
    TYPE weights1[input_dimension*nodes_per_layer];
    TYPE weights2[nodes_per_layer*nodes_per_layer];
    TYPE weights3[nodes_per_layer*possible_outputs];
    TYPE biases1[nodes_per_layer];
    TYPE biases2[nodes_per_layer];
    TYPE biases3[possible_outputs];
    TYPE training_data[training_sets*input_dimension];
    TYPE training_targets[training_sets*possible_outputs];
};
