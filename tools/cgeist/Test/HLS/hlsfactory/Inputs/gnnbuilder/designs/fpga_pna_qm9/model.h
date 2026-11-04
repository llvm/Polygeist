#pragma once

#define __FLOATING_POINT_MODEL__ 0

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>


#if __FLOATING_POINT_MODEL__
    #pragma message("Floating point model")
    #include <cmath>
    typedef float F_TYPE;
    typedef float W_TYPE;

    #define m_sqrt(x) (std::sqrt(x))
    #define m_rsqrt(x) (F_TYPE(1.0) / std::sqrt(x))
    #define m_recip(x) (F_TYPE(1.0) / x)
    #define m_erf(x) (std::erf(x))
    #define m_tanh(x) (std::tanh(x))
    #define m_pow(x, y) (std::pow(x, y))
    #define m_exp(x) (std::exp(x))
    #define m_log(x) (std::log(x))
    #define m_abs(x) (std::abs(x))
    #define m_sin(x) (std::sin(x))
    #define m_cos(x) (std::cos(x))
    #define m_pi() ((float)3.14159265358979323846)
    #define m_signbit(x) (std::signbit(x))

#else
    #pragma message("Fixed point model")
    #include "ap_fixed.h"
    #include "hls_math.h"

    #define FIXED_TYPE_F ap_fixed<32, 16>
    #define FIXED_TYPE_W ap_fixed<32, 16>
    typedef FIXED_TYPE_F F_TYPE;
    typedef FIXED_TYPE_W W_TYPE;

    #define m_sqrt(x) (hls::sqrt(x))
    #define m_rsqrt(x) (hls::rsqrt(x))
    #define m_recip(x) (hls::recip(x))
    #define m_erf(x) (hls::erf(x))
    #define m_tanh(x) (hls::tanh(x))
    #define m_pow(x, y) (hls::pow(x, y))
    #define m_exp(x) (hls::exp(x))
    #define m_log(x) (hls::log(x))
    #define m_abs(x) (hls::abs(x))
    #define m_sin(x) (hls::sin(x))
    #define m_cos(x) (hls::cos(x))
    #define m_pi() ((W_TYPE)3.14159265358979323846)
    #define m_signbit(x) (hls::signbit(x))

#endif


#include "gnn_builder_lib.h"

extern "C" {
void fpga_pna_qm9_top(
    F_TYPE node_feature_table_input[600][11],
    int edge_list_input[600][2],
    F_TYPE model_output[19],
    int num_of_nodes,
    int num_of_edges,
    int copy_parameters_flag,
    W_TYPE mlp_head_linear_layers_0_weight_fixed_in[64][192],
    W_TYPE mlp_head_linear_layers_0_bias_fixed_in[64],
    W_TYPE mlp_head_linear_layers_1_weight_fixed_in[64][64],
    W_TYPE mlp_head_linear_layers_1_bias_fixed_in[64],
    W_TYPE mlp_head_linear_layers_2_weight_fixed_in[64][64],
    W_TYPE mlp_head_linear_layers_2_bias_fixed_in[64],
    W_TYPE mlp_head_linear_layers_3_weight_fixed_in[64][64],
    W_TYPE mlp_head_linear_layers_3_bias_fixed_in[64],
    W_TYPE mlp_head_linear_layers_4_weight_fixed_in[19][64],
    W_TYPE mlp_head_linear_layers_4_bias_fixed_in[19],
    W_TYPE gnn_convs_0_conv_pre_nns_0_0_weight_fixed_in[11][22],
    W_TYPE gnn_convs_0_conv_pre_nns_0_0_bias_fixed_in[11],
    W_TYPE gnn_convs_0_conv_post_nns_0_0_weight_fixed_in[128][143],
    W_TYPE gnn_convs_0_conv_post_nns_0_0_bias_fixed_in[128],
    W_TYPE gnn_convs_0_conv_lin_weight_fixed_in[128][128],
    W_TYPE gnn_convs_0_conv_lin_bias_fixed_in[128],
    W_TYPE gnn_convs_1_conv_pre_nns_0_0_weight_fixed_in[128][256],
    W_TYPE gnn_convs_1_conv_pre_nns_0_0_bias_fixed_in[128],
    W_TYPE gnn_convs_1_conv_post_nns_0_0_weight_fixed_in[128][1664],
    W_TYPE gnn_convs_1_conv_post_nns_0_0_bias_fixed_in[128],
    W_TYPE gnn_convs_1_conv_lin_weight_fixed_in[128][128],
    W_TYPE gnn_convs_1_conv_lin_bias_fixed_in[128],
    W_TYPE gnn_convs_2_conv_pre_nns_0_0_weight_fixed_in[128][256],
    W_TYPE gnn_convs_2_conv_pre_nns_0_0_bias_fixed_in[128],
    W_TYPE gnn_convs_2_conv_post_nns_0_0_weight_fixed_in[128][1664],
    W_TYPE gnn_convs_2_conv_post_nns_0_0_bias_fixed_in[128],
    W_TYPE gnn_convs_2_conv_lin_weight_fixed_in[128][128],
    W_TYPE gnn_convs_2_conv_lin_bias_fixed_in[128],
    W_TYPE gnn_convs_3_conv_pre_nns_0_0_weight_fixed_in[128][256],
    W_TYPE gnn_convs_3_conv_pre_nns_0_0_bias_fixed_in[128],
    W_TYPE gnn_convs_3_conv_post_nns_0_0_weight_fixed_in[128][1664],
    W_TYPE gnn_convs_3_conv_post_nns_0_0_bias_fixed_in[128],
    W_TYPE gnn_convs_3_conv_lin_weight_fixed_in[128][128],
    W_TYPE gnn_convs_3_conv_lin_bias_fixed_in[128],
    W_TYPE gnn_convs_4_conv_pre_nns_0_0_weight_fixed_in[128][256],
    W_TYPE gnn_convs_4_conv_pre_nns_0_0_bias_fixed_in[128],
    W_TYPE gnn_convs_4_conv_post_nns_0_0_weight_fixed_in[128][1664],
    W_TYPE gnn_convs_4_conv_post_nns_0_0_bias_fixed_in[128],
    W_TYPE gnn_convs_4_conv_lin_weight_fixed_in[128][128],
    W_TYPE gnn_convs_4_conv_lin_bias_fixed_in[128],
    W_TYPE gnn_convs_5_conv_pre_nns_0_0_weight_fixed_in[128][256],
    W_TYPE gnn_convs_5_conv_pre_nns_0_0_bias_fixed_in[128],
    W_TYPE gnn_convs_5_conv_post_nns_0_0_weight_fixed_in[64][1664],
    W_TYPE gnn_convs_5_conv_post_nns_0_0_bias_fixed_in[64],
    W_TYPE gnn_convs_5_conv_lin_weight_fixed_in[64][64],
    W_TYPE gnn_convs_5_conv_lin_bias_fixed_in[64]
);
}