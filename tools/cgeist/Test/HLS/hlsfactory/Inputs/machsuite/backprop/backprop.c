#include "backprop.h"

void soft_max(TYPE net_outputs[possible_outputs], TYPE activations[possible_outputs]) {
    int i;
    TYPE sum;
    sum = (TYPE) 0.0;

    for(i=0; i < possible_outputs; i++) {
        sum += exp(-activations[i]);
    }
    for(i=0; i < possible_outputs; i++) {
        net_outputs[i] = exp(-activations[i])/sum;
    }
}

void RELU(TYPE activations[nodes_per_layer], TYPE dactivations[nodes_per_layer], int size) {
    int i;
    for( i = 0; i < size; i++) {
        dactivations[i] = activations[i]*(1.0-activations[i]);
        activations[i] = 1.0/(1.0+exp(-activations[i]));
    }
}

void add_bias_to_activations(TYPE biases[nodes_per_layer],  TYPE activations[nodes_per_layer],  int size) {
    int i;
    for( i = 0; i < size; i++){
        activations[i] = activations[i] + biases[i];
    }
}

void matrix_vector_product_with_bias_input_layer(TYPE biases[nodes_per_layer], TYPE weights[input_dimension*nodes_per_layer],  TYPE activations[nodes_per_layer], TYPE input_sample[input_dimension]){
    int i,j;
    mvp_product_input_loop: for(j = 0; j < nodes_per_layer; j++){
        activations[j] = (TYPE)0.0;
        for (i = 0; i < input_dimension; i++){
            activations[j] += weights[j*input_dimension + i] * input_sample[i];
        }
    }
    add_bias_to_activations(biases, activations, nodes_per_layer);
}

void matrix_vector_product_with_bias_second_layer(TYPE biases[nodes_per_layer], TYPE weights[nodes_per_layer*nodes_per_layer],  TYPE activations[nodes_per_layer],  TYPE input_activations[nodes_per_layer]){
    int i,j;
    for (i = 0; i < nodes_per_layer; i++){
        activations[i] = (TYPE)0.0;
        mvp_product_loop2: for(j = 0; j < nodes_per_layer; j++){
            activations[i] += weights[i*nodes_per_layer + j] * input_activations[j];
        }
    }
    add_bias_to_activations(biases, activations, nodes_per_layer);
}

void matrix_vector_product_with_bias_output_layer(TYPE biases[possible_outputs], TYPE weights[nodes_per_layer*possible_outputs], TYPE activations[possible_outputs], TYPE input_activations[nodes_per_layer]){
    int i, j;
    for(j = 0; j < possible_outputs; j++){
        activations[j] = (TYPE)0.0;
        mvp_product_loop1: for (i = 0; i < nodes_per_layer; i++){
            activations[j] += weights[j*nodes_per_layer + i] * input_activations[i];
        }
    }
    add_bias_to_activations(biases, activations, possible_outputs);
}

void take_difference(TYPE net_outputs[possible_outputs], TYPE solutions[possible_outputs], TYPE output_difference[possible_outputs],  TYPE dactivations[possible_outputs]) {
    int i;
    for( i = 0; i < possible_outputs; i++){
        output_difference[i] = (((net_outputs[i]) - solutions[i]) * -1.0) * dactivations[i];
    }
}

void get_delta_matrix_weights3(TYPE delta_weights3[nodes_per_layer*possible_outputs],  TYPE output_difference[possible_outputs],  TYPE last_activations[nodes_per_layer]) {
    int i, j;
    delta_matrix_weights3_loop: for( i = 0; i < nodes_per_layer; i++) {
        for( j = 0; j < possible_outputs; j++) {
            delta_weights3[i*possible_outputs + j] = last_activations[i] * output_difference[j];
        }
    }
}

void get_oracle_activations2(TYPE weights3[nodes_per_layer*possible_outputs],  TYPE output_differences[possible_outputs],  TYPE oracle_activations[nodes_per_layer], TYPE dactivations[nodes_per_layer]) {
    int i, j;
    activations2_loop: for( i = 0; i < nodes_per_layer; i++) {
        oracle_activations[i] = (TYPE)0.0;
        for( j = 0; j < possible_outputs; j++) {
            oracle_activations[i] += output_differences[j] * weights3[i*possible_outputs + j];
        }
        oracle_activations[i] = oracle_activations[i] * dactivations[i];
    }
}

void get_delta_matrix_weights2(TYPE delta_weights2[nodes_per_layer*nodes_per_layer], TYPE output_difference[nodes_per_layer],  TYPE last_activations[nodes_per_layer]) {
    int i, j;
    for( i = 0; i < nodes_per_layer; i++) {
        delta_matrix_weights2_loop: for( j = 0; j < nodes_per_layer; j++) {
            delta_weights2[i*nodes_per_layer + j] = last_activations[i] * output_difference[j];
        }
    }
}

void get_oracle_activations1(TYPE weights2[nodes_per_layer*nodes_per_layer], TYPE output_differences[nodes_per_layer],  TYPE oracle_activations[nodes_per_layer], TYPE dactivations[nodes_per_layer]) {
    int i, j;
    for( i = 0; i < nodes_per_layer; i++) {
        oracle_activations[i] = (TYPE)0.0;
        activations1_loop: for( j = 0; j < nodes_per_layer; j++) {
            oracle_activations[i] += output_differences[j] * weights2[i*nodes_per_layer + j];
        }
        oracle_activations[i] = oracle_activations[i] * dactivations[i];
    }
}

void get_delta_matrix_weights1(TYPE delta_weights1[input_dimension*nodes_per_layer], TYPE output_difference[nodes_per_layer],  TYPE last_activations[input_dimension]) {
    int i, j;
    for( i = 0; i < input_dimension; i++) {
        delta_matrix_weights1_loop: for( j = 0; j < nodes_per_layer; j++) {
            delta_weights1[i*nodes_per_layer + j] = last_activations[i] * output_difference[j];
        }
    }
}

void update_weights(TYPE weights1[input_dimension*nodes_per_layer], TYPE weights2[nodes_per_layer*nodes_per_layer],  TYPE weights3[nodes_per_layer*possible_outputs], TYPE d_weights1[input_dimension*nodes_per_layer], TYPE d_weights2[nodes_per_layer*nodes_per_layer], TYPE d_weights3[nodes_per_layer*possible_outputs], TYPE biases1[nodes_per_layer], TYPE biases2[nodes_per_layer], TYPE biases3[possible_outputs],  TYPE d_biases1[nodes_per_layer],   TYPE d_biases2[nodes_per_layer], TYPE d_biases3[possible_outputs]) {
    int i, j;
    double norm, bias_norm;
    norm = 0.0;
    bias_norm = 0.0;

    for(i=0; i < input_dimension; i++){
        up_weight_loop1: for(j = 0; j < nodes_per_layer; j++){
            weights1[i*nodes_per_layer + j] -= (d_weights1[i*nodes_per_layer + j] * learning_rate);
            norm += weights1[i*nodes_per_layer + j]*weights1[i*nodes_per_layer + j];
        }
    }
    up_weight_loop2: for(i=0; i < nodes_per_layer; i++){
        biases1[i] -= (d_biases1[i]*learning_rate);
        bias_norm += biases1[i]*biases1[i];
    }
    
    norm = sqrt(norm);
    bias_norm = sqrt(bias_norm);

    for(i=0; i < input_dimension; i++){
        up_weight_loop3: for(j = 0; j < nodes_per_layer; j++){
            weights1[i*nodes_per_layer + j] = (weights1[i*nodes_per_layer + j]/norm);
        }
    }
    up_weight_loop4: for(i=0; i < nodes_per_layer; i++){
        biases1[i] = (biases1[i]/bias_norm);
    }

    norm = (double)0.0;
    bias_norm = (double)0.0;

    for(i=0; i < nodes_per_layer; i++){
        up_weight_loop5: for(j = 0; j < nodes_per_layer; j++){
            weights2[i*nodes_per_layer + j] -= (d_weights2[i*nodes_per_layer + j] * learning_rate);
            norm += weights2[i*nodes_per_layer + j]*weights2[i*nodes_per_layer + j];
        }
    }
    up_weight_loop6: for(i=0; i < nodes_per_layer; i++){
        biases2[i] -= (d_biases2[i]*learning_rate);
        bias_norm += biases2[i]*biases2[i];
    }

    norm = sqrt(norm);
    bias_norm = sqrt(bias_norm);

    for(i=0; i < nodes_per_layer; i++){
        up_weight_loop7: for(j = 0; j < nodes_per_layer; j++){
            weights2[i*nodes_per_layer + j] = (weights2[i*nodes_per_layer + j]/norm);
        }
    }
    up_weight_loop8: for(i=0; i < nodes_per_layer; i++){
        biases2[i] = (biases2[i]/bias_norm);
    }

    norm = (double)0.0;
    bias_norm = (double)0.0;

    up_weight_loop9: for(i=0; i < nodes_per_layer; i++){
        for(j = 0; j < possible_outputs; j++){
            weights3[i*possible_outputs + j] -= (d_weights3[i*possible_outputs + j] * learning_rate);
            norm += weights3[i*possible_outputs + j]*weights3[i*possible_outputs + j];
        }
    }
    for(i=0; i<possible_outputs;i++){
        biases3[i] -= d_biases3[i]*learning_rate;
        bias_norm += biases3[i]*biases3[i];
    }

    norm = sqrt(norm);
    bias_norm = sqrt(bias_norm);

    up_weight_loop10: for(i=0; i < nodes_per_layer; i++){
        for(j = 0; j < possible_outputs; j++){
            weights3[i*possible_outputs + j] = (weights3[i*possible_outputs + j]/norm);
        }
    }
    for(i=0; i < possible_outputs; i++){
        biases3[i] = (biases3[i]/bias_norm);
    }
}

void backprop(TYPE weights1[input_dimension*nodes_per_layer], TYPE weights2[nodes_per_layer*nodes_per_layer], TYPE weights3[nodes_per_layer*possible_outputs],  TYPE biases1[nodes_per_layer],   TYPE biases2[nodes_per_layer],   TYPE biases3[possible_outputs],  TYPE training_data[training_sets*input_dimension],  TYPE training_targets[training_sets*possible_outputs]) {
    int i,j;
    //Forward and training structures
    TYPE activations1[nodes_per_layer];
    TYPE activations2[nodes_per_layer];
    TYPE activations3[possible_outputs];
    TYPE dactivations1[nodes_per_layer];
    TYPE dactivations2[nodes_per_layer];
    TYPE dactivations3[possible_outputs];
    TYPE net_outputs[possible_outputs];
    //Training structure
    TYPE output_difference[possible_outputs];
    TYPE delta_weights1[input_dimension*nodes_per_layer]; 
    TYPE delta_weights2[nodes_per_layer*nodes_per_layer];
    TYPE delta_weights3[nodes_per_layer*possible_outputs];
    TYPE oracle_activations1[nodes_per_layer];
    TYPE oracle_activations2[nodes_per_layer];

    for(i=0; i<training_sets; i++){
        init_loop: for(j=0;j<nodes_per_layer;j++){
            #pragma HLS unroll factor=64
            activations1[j] = (TYPE)0.0;
            activations2[j] = (TYPE)0.0;
            if(j<possible_outputs){
                activations3[j] = (TYPE)0.0;
            }
        }
        matrix_vector_product_with_bias_input_layer(biases1, weights1, activations1, &training_data[i*input_dimension]);
        RELU(activations1, dactivations1, nodes_per_layer);
        matrix_vector_product_with_bias_second_layer(biases2, weights2, activations2, activations1);
        RELU(activations2, dactivations2, nodes_per_layer);
        matrix_vector_product_with_bias_output_layer(biases3, weights3, activations3, activations2);
        RELU(activations3, dactivations3, possible_outputs);
        soft_max(net_outputs, activations3);
        take_difference(net_outputs, &training_targets[i*possible_outputs], output_difference, dactivations3);
        get_delta_matrix_weights3(delta_weights3, output_difference, activations2);
        get_oracle_activations2(weights3, output_difference, oracle_activations2, dactivations2);
        get_delta_matrix_weights2(delta_weights2, oracle_activations2, activations1);
        get_oracle_activations1(weights2, oracle_activations2, oracle_activations1, dactivations1);
        get_delta_matrix_weights1(delta_weights1, oracle_activations1, &training_data[i*input_dimension]);
        update_weights(weights1, weights2, weights3, delta_weights1, delta_weights2, delta_weights3, 
                       biases1, biases2, biases3, oracle_activations1, oracle_activations2, output_difference);
    }
}
