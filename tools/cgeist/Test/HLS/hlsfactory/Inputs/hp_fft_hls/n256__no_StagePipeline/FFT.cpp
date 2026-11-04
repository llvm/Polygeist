#include "FFT.h"

int revIdxTab[FFT_NUM];     
complex<float> twiddles[FFT_NUM/2];

void RADIX2_BFLY_double_buffer(complex<float>* data_ld, complex<float>* data_st, int i0, int i1, bool tw_enable, complex<float> tw) {
#pragma HLS inline
    auto d0 = data_ld[i0];                     
    auto d1 = data_ld[i1];
    if(tw_enable){
        d1 = d1 * tw;
    }
    data_st[i0] = d0 + d1;
    data_st[i1] = d0 - d1;
}

void FFT_stage_temporal(complex<float> data_ld[FFT_NUM], complex<float> data_st[FFT_NUM], int stage){
    int bflySize = 1 << stage; //from 2 to FFT_NUM
    int bflyStep = bflySize >> 1; //from 1 to FFT_NUM/2
    Group_loop: 
    for (int k = 0, tw_addr=0; k < bflyStep; ++k, tw_addr+=(FFT_NUM >> stage)) {
    #pragma HLS loop_tripcount min=1 max=1<<(EXP2_FFT-1-EXP2_FFT/2) // fake loop_count, used to estimate average time
        // Butterfly computations
        auto tw = twiddles[tw_addr]; 
        Pair_loop: 
        for (int m = 0; m < FFT_NUM; m += bflySize) {
        #pragma HLS loop_flatten // didn't work due to the dyamic loop bound and complex access pattern of tw
        #pragma HLS pipeline
        #pragma HLS dependence variable=data_ld inter false
        #pragma HLS dependence variable=data_st inter false
        #pragma HLS loop_tripcount min=1 max=1<<(EXP2_FFT/2) // fake loop_count, used to estimate average time
            int i0 = k + m;
            int i1 = k + m + bflyStep;
            // special processing for k=0, wherein twIdx=0 yields tw=1
            RADIX2_BFLY_double_buffer(data_ld, data_st, i0, i1, k > 0, tw);  
        }
    }
}

void FFT_TOP(complex<float> dataIn[FFT_NUM], complex<float> dataOut[FFT_NUM]) 
{
    for (int i = 0; i < FFT_NUM/2; ++i) {
        double angle = -2 * PI * i / FFT_NUM;
        auto tw = complex<float>(cos(angle), sin(angle));
        twiddles[i] = (complex<float>)tw;
    }
    // pre-compute reverse table, note that its inverse function is itself.
    for (int i = 0; i < FFT_NUM; ++i) {
        int reversed = 0;
        int number = i;
        for (int j = 0; j < EXP2_FFT; ++j) {
            reversed = reversed * 2 + (number & 1);
            number >>= 1;
        }
        revIdxTab[i] = reversed;
    }

    static complex<float> data[2][FFT_NUM];
    #pragma HLS array_partition variable=data type=complete dim=1

    PreP_Fwd_loop: for (int i = 0; i < FFT_NUM; i++)      
        #pragma HLS unroll factor=2                               
        data[0][i] = dataIn[i];    
    // reverse ordering the input data, noting it is done by wiring in VLSI
    Rev_Bit_loop: for (int i = 0; i < FFT_NUM; i++) {
    #pragma HLS pipeline
    #pragma HLS unroll factor=2
        data[1][i] = data[0][revIdxTab[i]];
    }
    Stage_loop: for (int bflyExp = 1; bflyExp <= EXP2_FFT; bflyExp++) {
        FFT_stage_temporal(data[bflyExp%2], data[1-bflyExp%2], bflyExp);
    }
    PostP_Fwd_loop: for (int i = 0; i < FFT_NUM; i++){
        #pragma HLS unroll factor=2 
        dataOut[i] = data[1][i];        
    }
}

