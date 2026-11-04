#include "../gradient.h"

template <int out_channel, int inp_channel, int inp_height, int inp_width>
void model_conv(
    fm_t input_feature_map[inp_channel][inp_height][inp_width],
    wt_t layer_weights[out_channel][inp_channel][3][3],
    wt_t layer_bias[out_channel],
    fm_t output_feature_map[out_channel][inp_height][inp_width]
)
{
//--------------------------------------------------------------------------
// Performs convolution with padding along with adding bias
// No activation function at the end
//--------------------------------------------------------------------------

    fm_t current_ip;

    for(int oc=0; oc<out_channel; oc++){
        for(int oh=0; oh<inp_height; oh++){
            for(int ow=0; ow<inp_width; ow++){
                for(int ic=0; ic<inp_channel; ic++){
                    for(int fh=0; fh<3; fh++){
                        for(int fw=0; fw<3; fw++){

                            // PADDING
                            if( (oh+fh-1) < 0 || \
                                (ow+fw-1) < 0 || \
                                (oh+fh) > inp_height || \
                                (ow+fw) > inp_width){
                                    current_ip = 0;
                                }

                            else{
                                current_ip = input_feature_map[ic][oh+fh-1][ow+fw-1];
                            }

                            // MAC operation
                            if(ic == 0 && fh == 0 && fw == 0)
                                output_feature_map[oc][oh][ow]  =   current_ip * layer_weights[oc][ic][fh][fw] + layer_bias[oc];
                            else
                                output_feature_map[oc][oh][ow]  +=  current_ip * layer_weights[oc][ic][fh][fw];

                        }
                    }
                }
            }
        }
    }

}

template<int inp_channel, int inp_height, int inp_width, int wt_height, int wt_width>
void model_conv_flatten_fc(
    fm_t input_feature_map[inp_channel][inp_height][inp_width],
    wt_t layer_weights[wt_height][wt_width],
    wt_t layer_bias[wt_height],
    fm_t output_feature_map[wt_height]
)
{
//--------------------------------------------------------------------------
// For convolution layer going into a fully-connected layer
// Flattens the CNN feature map and,
// computes the matrix multiplication with the FC layer weights
// No activation function at the end
//--------------------------------------------------------------------------
    
    for(int i=0; i<wt_height; i++){
        output_feature_map[i] = layer_bias[i];

        for(int ic=0; ic<inp_channel; ic++){
            for(int ih=0; ih<inp_height; ih++){
                for(int iw=0; iw<inp_width; iw++){
                    output_feature_map[i] += input_feature_map[ic][ih][iw] * layer_weights[i][inp_channel*ic + inp_height*ih + iw];
                }
            }
        }
    }

}

template <int inp_channel, int inp_height, int inp_width>
void model_max_pool(
   fm_t input_feature_map[inp_channel][inp_height][inp_width],
   fm_t output_feature_map[inp_channel][inp_height/2][inp_width/2],
   mk_t mask_map[inp_channel][inp_height/2][inp_width/2]
)
{   
//--------------------------------------------------------------------------
// Performs max-pooling
//--------------------------------------------------------------------------

    fm_t max;
    mk_t pos;

    for(int ic = 0; ic < inp_channel; ic++){
        for(int ih = 0; ih < inp_height; ih+=2){
            for(int iw = 0; iw < inp_width; iw+=2){

                max = input_feature_map[ic][ih][iw];
                pos = 0;

                if(max < input_feature_map[ic][ih][iw+1]){
                    max = input_feature_map[ic][ih][iw+1];
                    pos = 1;
                }

                if(max < input_feature_map[ic][ih+1][iw]){
                    max = input_feature_map[ic][ih+1][iw];
                    pos = 2;
                }
                
                if(max < input_feature_map[ic][ih+1][iw+1]){
                    max = input_feature_map[ic][ih+1][iw+1];
                    pos = 3;
                }

                output_feature_map[ic][ih/2][iw/2] = max;
                mask_map[ic][ih/2][iw/2] = pos;

           }
       }
   }
}

template <int inp_length>
void model_relu_fc(
    fm_t input_feature_map[inp_length],
    fm_t output_feature_map[inp_length]
)
{
//--------------------------------------------------------------------------
// Performs Relu activation on a flatten channel 
//--------------------------------------------------------------------------

    for(int i = 0; i < inp_length; i++){
        if(input_feature_map[i] > 0) 
            output_feature_map[i] = input_feature_map[i];
        else 
            output_feature_map[i] = 0;
    }
}

template <int inp_length, int out_length>
void model_mat_mul(
    fm_t input_feature_map[inp_length],
    wt_t layer_weights[out_length][inp_length],
    wt_t layer_bias[out_length],
    fm_t output_feature_map[out_length]
)
{

    for(int i=0; i<out_length; i++){
        output_feature_map[i] = layer_bias[i];

        for(int j=0; j<inp_length; j++){
            output_feature_map[i] += input_feature_map[j] * layer_weights[i][j];
        }
    }

}