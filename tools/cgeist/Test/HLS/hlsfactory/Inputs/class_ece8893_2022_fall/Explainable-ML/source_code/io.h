#include "gradient.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

template <int inp_channel, int inp_height, int inp_width>
void read_input_feature(float ifmap[inp_channel][inp_height][inp_width]){

    for(int ic=0; ic<inp_channel; ic++){
        for(int ih=0; ih<inp_height; ih++){
            for(int iw=0; iw<inp_width; iw++){
                ifmap[ic][ih][iw] = 1;
            }
        }
    }
}

template <int out_channel, int inp_channel, int k_height, int k_width>
void read_conv_weight(const string& file, float conv_weight[out_channel][inp_channel][k_height][k_width]){
    int size = out_channel*inp_channel*k_height*k_width;
    
    ifstream ifs_conv_input(file, ios::in | ios::binary);
    ifs_conv_input.read((char*)(***conv_weight), size*sizeof(float));
    ifs_conv_input.close();
}

template <int out_channel>
void read_conv_bias(const string& file, float conv_bias[out_channel]){

    ifstream ifs_conv_bias(file, ios::in | ios::binary);
    ifs_conv_bias.read((char*)(conv_bias), out_channel*sizeof(float));
    ifs_conv_bias.close();
}

template <int length, int width>
void read_fc_weight(string file, float fc_weight[length][width]){

    ifstream ifs_fc_weight(file, ios::in | ios::binary);
    ifs_fc_weight.read((char*)(*fc_weight), width*length*sizeof(float));
    ifs_fc_weight.close();
}

template <int length>
void read_fc_bias(string file, float fc_bias[length]){

    ifstream ifs_fc_bias(file, ios::in | ios::binary);
    ifs_fc_bias.read((char*)(fc_bias), length*sizeof(float));
    ifs_fc_bias.close();
}

template <int inp_channel, int inp_height, int inp_width>
void convert_input_3d(
    float conv_feature_map[inp_channel][inp_height][inp_width],
    fm_t fixp_conv_feature_map[inp_channel][inp_height][inp_width]
)
{
    for(int c = 0; c < inp_channel; c++)
        for(int i = 0; i < inp_height; i++)
            for(int j = 0; j < inp_width; j++)
                fixp_conv_feature_map[c][i][j] = (fm_t) conv_feature_map[c][i][j];
}

template <int length>
void convert_input_1d(
    float fc_feature_map[length],
    fm_t fixp_fc_feature_map[length]
)
{
    for(int i=0; i<length; i++){
        fixp_fc_feature_map[i] = (fm_t) fc_feature_map[i];
    }
}

template <int inp_channel, int out_channel>
void convert_conv_layer_params(
    float conv_layer_weights[out_channel][inp_channel][3][3],
    float conv_layer_bias[out_channel],
    wt_t fixp_conv_layer_weights[out_channel][inp_channel][3][3],
    wt_t fixp_conv_layer_bias[out_channel]
)
{
    for(int oc=0; oc<out_channel; oc++){
        fixp_conv_layer_bias[oc] = (wt_t) conv_layer_bias[oc];

        for(int ic=0; ic<inp_channel; ic++){
            for(int kh=0; kh<3; kh++){
                for(int kw=0; kw<3; kw++){
                    fixp_conv_layer_weights[oc][ic][kh][kw] = (wt_t) conv_layer_weights[oc][ic][kh][kw];
                }
            }
        }
    }
}

template <int inp_length, int out_length>
void convert_fc_layer_params(
    float fc_layer_weights[out_length][inp_length],
    float fc_layer_bias[out_length],
    wt_t fixp_fc_layer_weights[out_length][inp_length],
    wt_t fixp_fc_layer_bias[out_length]
)
{
    for(int i=0; i<out_length; i++){
        fixp_fc_layer_bias[i] = (wt_t) fc_layer_bias[i];

        for(int j=0; j<inp_length; j++){
            fixp_fc_layer_weights[i][j] = (wt_t) fc_layer_weights[i][j];
        }
    }

}