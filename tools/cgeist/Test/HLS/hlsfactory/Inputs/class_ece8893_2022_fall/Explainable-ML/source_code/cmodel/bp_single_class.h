#include "../gradient.h"

template <int inp_width>
void model_relu_fc_grad_single(
    fm_t input_gradient_map[inp_width],
    fm_t input_relu_map[inp_width],
    fm_t output_gradient_map[inp_width]
)
{
//--------------------------------------------------------------------------
// Performs Relu activation for gradients of an FC layer 
//--------------------------------------------------------------------------

    for(int i=0; i<inp_width; i++){
        if(input_relu_map[i] > 0)
            output_gradient_map[i] = input_gradient_map[i];
        else
            output_gradient_map[i] = 0;
    }
}

template <int inp_width, int out_width>
void model_fc_grad_single(
    fm_t input_gradient_map[inp_width],
    wt_t layer_weights[inp_width][out_width],
    fm_t output_gradient_map[out_width]
)
{
//--------------------------------------------------------------------------
// Performs gradient computation across two FC layers 
//--------------------------------------------------------------------------

    for(int j=0; j<out_width; j++){
        output_gradient_map[j] = 0;

        for(int k=0; k<inp_width; k++){
            output_gradient_map[j] += input_gradient_map[k] * layer_weights[k][j];
        }
    }
}

template <int out_channel, int out_height, int out_width>
void model_reshape_single(
    fm_t input_gradient_map[out_channel*out_height*out_width],
    fm_t output_gradient_map[out_channel][out_height][out_width]
)
{
//--------------------------------------------------------------------------
// Reshapes FC layer gradient into previous Conv layers shape
// Useful at the transition between FC and CONV layers
//--------------------------------------------------------------------------

    for(int oc=0; oc<out_channel; oc++){
        for(int oh=0; oh<out_height; oh++){
            for(int ow=0; ow<out_width; ow++){
                int idx = out_channel*oc + out_height*oh + ow;
                output_gradient_map[oc][oh][ow] = input_gradient_map[idx];
            }
        }
    }
}

template <int inp_channel, int inp_height, int inp_width>
void model_upsample_single(
    fm_t input_gradient_map[inp_channel][inp_height][inp_width],
    mk_t mask_map[inp_channel][inp_height][inp_width],
    fm_t output_gradient_map[inp_channel][inp_height*2][inp_width*2]
)
{
//--------------------------------------------------------------------------
// Upsamples the gradient to propagate back from a max-pool layer
//--------------------------------------------------------------------------

    for(int ic=0; ic<inp_channel; ic++){
        for(int ih=0; ih<inp_height*2; ih+=2){
            for(int iw=0; iw<inp_width*2; iw+=2){
                int pos = mask_map[ic][ih/2][iw/2];

                output_gradient_map[ic][ih][iw] = 0;
                output_gradient_map[ic][ih+1][iw] = 0;
                output_gradient_map[ic][ih][iw+1] = 0;
                output_gradient_map[ic][ih+1][iw+1] = 0;

                if(pos == 0)
                    output_gradient_map[ic][ih][iw]      = input_gradient_map[ic][ih/2][iw/2];
                else if(pos == 1)
                    output_gradient_map[ic][ih][iw+1]    = input_gradient_map[ic][ih/2][iw/2];
                else if(pos == 2)
                    output_gradient_map[ic][ih+1][iw]    = input_gradient_map[ic][ih/2][iw/2];
                else
                    output_gradient_map[ic][ih+1][iw+1]  = input_gradient_map[ic][ih/2][iw/2];

            }
        }
    }
}

template <int out_channel, int inp_channel, int inp_height, int inp_width>
void model_flipped_conv_single(
    fm_t input_gradient_map[inp_channel][inp_height][inp_width],
    wt_t layer_weights[inp_channel][out_channel][3][3],
    fm_t output_gradient_map[out_channel][inp_height][inp_width]
)
{

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
                                current_ip = input_gradient_map[ic][oh+fh-1][ow+fw-1];
                            }

                            // MAC operation
                            if(ic == 0 && fh == 0 && fw == 0)
                                output_gradient_map[oc][oh][ow]  =   current_ip * layer_weights[ic][oc][2-fh][2-fw];
                            else
                                output_gradient_map[oc][oh][ow]  +=  current_ip * layer_weights[ic][oc][2-fh][2-fw];

                        }
                    }
                }
            }
        }
    }
}