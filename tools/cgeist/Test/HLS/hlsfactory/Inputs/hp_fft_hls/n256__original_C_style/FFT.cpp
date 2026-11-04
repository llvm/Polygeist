#include "FFT.h"
// Radix-2 butterfly computation
inline void RADIX2_BFLY(complex<float> *data, int i0, int i1) {       
    auto d0 = data[i0];                     
    auto d1 = data[i1];                     
    data[i0] = d0 + d1;                     
    data[i1] = d0 - d1;                     
}

void FFT_TOP(complex<float> dataIn[FFT_NUM], complex<float> dataOut[FFT_NUM]) {
    int revIdxTab[FFT_NUM];     
    complex<float> twiddles[FFT_NUM<2? 1 : FFT_NUM/2];
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
    static complex<float> data[FFT_NUM];
                                                          
    PreP_Fwd_loop: for (int i = 0; i < FFT_NUM; i++)      
        data[i] = dataIn[i];                                        

    //use buffer to reverse bit
    static complex<float> dataBuf[FFT_NUM];
    Rev_Bit_loop_ld: for (int i = 0; i < FFT_NUM; i++)
        dataBuf[i] = data[revIdxTab[i]];
    Rev_Bit_loop_st: for (int i = 0; i < FFT_NUM; i++)
        data[i] = dataBuf[i];

    int bflySize, bflyExp, bflyStep, i0, i1;
    auto index_init = [&](int k) {
        i0 = k;
        i1 = k + bflyStep;
    };
    auto index_step = [&](int bflySize) {
        i0 += bflySize;     
        i1 += bflySize;
    };
    //Stage 1: bflyExp = 1 contains only the special case tw=1
    bflySize = 2; bflyStep = 1;
    index_init(0);
    Stage_1_0: for (int m = 0; m < FFT_NUM; m += 2, index_step(2))
        RADIX2_BFLY(data, i0, i1);

    //Stage 2: bflyExp = 2 contains only the special case tw=1 and tw = -j
    bflySize = 4; bflyStep = 2;
    index_init(0);
    Stage_2_0: for (int m = 0; m < FFT_NUM; m += 4, index_step(4))
        RADIX2_BFLY(data, i0, i1);

    index_init(1);
    Stage_2_1: for (int m = 0; m < FFT_NUM; m += 4, index_step(4)) {
        data[i1] = complex<float>(data[i1].imag(), -data[i1].real());
        RADIX2_BFLY(data, i0, i1);
    }

    //Stage 3 and beyond
    Stage_3more: for (bflyExp = 3; bflyExp <= EXP2_FFT; bflyExp++) {
        bflySize = 1<<bflyExp; //from 8 to FFT_NUM
        bflyStep = bflySize >> 1; //from 4 to FFT_NUM/2
        // special processing for k=0, wherein twIdx=0 yields tw=1
        index_init(0);
        Stage_3more_0: for (int m = 0; m < FFT_NUM; m += bflySize, index_step(bflySize))
            #pragma HLS loop_tripcount min=1 max=1<<(EXP2_FFT/2)
            RADIX2_BFLY(data, i0, i1);

        Stage_3more_Group_loop: for (int k = 1; k < bflyStep; ++k) {
            #pragma HLS loop_tripcount min=3 max=(1<<(EXP2_FFT-1-EXP2_FFT/2))-1 // fake loop_count, used to estimate average time
            // Butterfly computations              
            auto tw = twiddles[(FFT_NUM >> bflyExp) * k];
            index_init(k);
            Stage_3more_Pair_loop: for (int m = 0; m < FFT_NUM; m += bflySize, index_step(bflySize)) {
                #pragma HLS loop_tripcount min=1 max=1<<(EXP2_FFT/2) // fake loop_count, used to estimate average time
                data[i1] *= tw;
                RADIX2_BFLY(data, i0, i1);
            }
        }
    }

    PostP_Fwd_loop: for (int i = 0; i < FFT_NUM; i++)
        dataOut[i] = data[i];        
}
