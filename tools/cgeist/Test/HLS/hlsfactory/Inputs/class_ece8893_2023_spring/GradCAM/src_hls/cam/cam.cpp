#include "../util.h"
#include "max_idx.h"


namespace cam{
// TODO: I still need to template the dimensions used here

const int IN_BUF_DEPTH = 1000;
const int OUT_BUF_HEIGHT = 7;
const int OUT_BUF_WIDTH = 7;
const int L4_DEPTH = 512;
    

void cam_core(
        fm_t cam_output[OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
        fm_t l4_out1[1][L4_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
        wt_t fc_weight[IN_BUF_DEPTH][L4_DEPTH],
        fm_t output[IN_BUF_DEPTH]
        )
        
{
    

    // Find winning class
    int c = max_idx<1000, fm_t>(output);
    
    /*
    for (int k = 0; k < 512; k++){
        for (int i=0; i<7; i++){
            for (int j=0; j<7; j++){
                l4_out1[0][k][i][j] *= fc_weight[c][k];
            }
        }
    } 
    */
    // Now wk is 512 x 7 x 7
    fm_t maxVal = cam_output[0][0];
    for (int i = 0; i < 7; i++){
        #pragma HLS pipeline off
        for (int j = 0; j < 7; j++){
            #pragma HLS pipeline off
            fm_t temp = 0;

            for (int k = 0; k < 512; k++)
            {
                temp += l4_out1[0][k][i][j] * fc_weight[c][k] / 512;
            }
        
            if (temp<0){
                cam_output[i][j] =0;
            }
            else{
                cam_output[i][j]= temp; 
            }
        
        }
    }

    #ifdef CSIM_DEBUG
    for (int i = 0; i < 49; i++) {
        if(cam_output[0][i] >maxVal) {
            maxVal= cam_output[0][i];
        }
    }
    printf("max value is %2f", maxVal);
    for (int i = 0; i < OUT_BUF_HEIGHT; i++) {
        for (int j = 0; j < OUT_BUF_WIDTH; j++) {
            cam_output[i][j] = cam_output[i][j]/ maxVal;
        }
    }
    #endif
    //WRITE_TO_FILE(cam_output, 7, 7, 1);
}
void cam(
    fm_t cam_output[OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    fm_t l4_out1[],
    wt_t fc_weight[IN_BUF_DEPTH][L4_DEPTH],
    fm_t output[IN_BUF_DEPTH]
)
{
    
    //const int IN_BUF_DEPTH = 1000;
    //const int OUT_BUF_HEIGHT = 7;
    //const int OUT_BUF_WIDTH = 7;
    //const int L4_DEPTH = 512;
    

    


    static fm_t in_buf[IN_BUF_DEPTH];
    static fm_t out_buf[OUT_BUF_HEIGHT][OUT_BUF_WIDTH];
    //Load weights
    static fm_t l4_out_buf[1][L4_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];
    static wt_t wt_buf[IN_BUF_DEPTH][L4_DEPTH];

    static fm_t *l4_out_buf_1d = (fm_t *)l4_out_buf;
    // Load weights
    for (int i = 0; i < IN_BUF_DEPTH; i++)
    {
       for (int j = 0; j < L4_DEPTH; j++)
        {
            #pragma HLS pipeline II=1
            wt_buf[i][j] = fc_weight[i][j];
        }
    }

    // load output to find the winning class
    for (int i=0; i<IN_BUF_DEPTH; i++){
        in_buf[i]= output[i];
    }

    //load l4_out1
    for (int ii=0; ii<L4_DEPTH*OUT_BUF_HEIGHT*OUT_BUF_WIDTH; ii++){
        #pragma HLS pipeline II=1
        l4_out_buf_1d[ii] = l4_out1[ii];
    }
    //Compute
    
    cam::cam_core((fm_t (*)[7])out_buf, (fm_t (*)[512][7][7])l4_out_buf, (wt_t (*)[512])fc_weight, (fm_t*)in_buf);
    //store
    for (int i = 0; i < OUT_BUF_HEIGHT; i++)
        {
            for(int j =0; j<OUT_BUF_WIDTH; j++){
            #pragma HLS pipeline II=1
            cam_output[i][j] = out_buf[i][j];
            }
        }
}
}
