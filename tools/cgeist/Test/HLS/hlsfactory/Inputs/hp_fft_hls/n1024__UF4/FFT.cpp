#include "FFT.h"


int revIdxTab[FFT_NUM];     
complex<float> twiddles[FFT_NUM<2? 1 : FFT_NUM/2];

void RADIX2_BFLY_double_buffer_quarter_CY(complex<float>* data_ld, complex<float>* data_st, int i0, int i1, bool inv_i1_enable, bool tw_enable, complex<float> tw) 
{
    #pragma HLS inline
    complex<float> d0 = data_ld[i0];                     
    complex<float> d1 = data_ld[i1];
    float d0_real, d0_imag, d1_real, d1_imag, d2_real, d2_imag, d3_real, d3_imag;
    if(inv_i1_enable){
        // d1 = complex<float>(d1.imag(), -d1.real());
        d1_real = d1.imag();
        d1_imag = - d1.real();
    }
    else if(tw_enable){
        #pragma HLS bind_op variable=d1_real op=fsub impl=fabric 
        #pragma HLS bind_op variable=d1_imag op=fadd impl=fabric 
        float a, b, c, d;
        a = d1.real(); b = d1.imag(); c = tw.real(); d = tw.imag();
        float ac, bd, ad, bc; 
        ac = a*c;
        bd = b*d;
        ad = a*d;
        bc = b*c;
        d1_real = ac - bd; 
        d1_imag = ad + bc;         
    }else{
        d1_real = d1.real();
        d1_imag = d1.imag();
    }

    #pragma HLS bind_op variable=d2_real op=fadd impl=fabric 
    #pragma HLS bind_op variable=d2_imag op=fadd impl=fabric 
    #pragma HLS bind_op variable=d3_real op=fsub impl=fabric 
    #pragma HLS bind_op variable=d3_imag op=fsub impl=fabric 

    d0_real = d0.real();
    d0_imag = d0.imag();
    d2_real = d0_real + d1_real;
    d2_imag = d0_imag + d1_imag;
    d3_real = d0_real - d1_real;
    d3_imag = d0_imag - d1_imag;
    data_st[i0] = complex<float>(d2_real, d2_imag);
    data_st[i1] = complex<float>(d3_real, d3_imag);

}


void output_result_array_to_stream (complex<float> data_6[FFT_NUM], hls::stream<hls::vector<complex<float>, UF*2>> & dataOut
){
    #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
    PostP_Fwd_loop: for (int i = 0; i < FFT_NUM/(UF*2); i++)  { 
        #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
        #pragma HLS pipeline                      
        hls::vector<complex<float>, UF*2> temp;
        for (int u = 0; u < UF*2; u++) {
            temp[u] = data_6[i*UF*2+u];
        }
        dataOut.write(temp);
    }
}


template<int N>  ap_uint<N> bit_reverse ( ap_uint<N> input ){
    #pragma HLS inline
    ap_uint<N> reversed;
    Loop_Reverse: for (int bit_i = 0; bit_i < N; bit_i++) {
        #pragma HLS UNROLL
        reversed.range(bit_i, bit_i) = input.range(N-1-bit_i, N-1-bit_i);
    }
    return reversed;
}


void reverse_input_stream_UF4 (
    hls::stream<hls::vector<complex<float>, UF*2>> & dataIn,
    hls::stream<hls::vector<complex<float>, UF*2>> & reverse_in_stream_vector
){

    complex<float> data_in_cyclic[UF*2][FFT_NUM/(UF*2)];

    // complex<float> data_rev_stream[UF*2][FFT_NUM/(UF*2)];
    // #pragma HLS array_partition variable=data_rev_stream type=complete dim=1

    complex<float> data_rev_stream_0[FFT_NUM/(UF*2)];
    complex<float> data_rev_stream_1[FFT_NUM/(UF*2)];
    complex<float> data_rev_stream_2[FFT_NUM/(UF*2)];
    complex<float> data_rev_stream_3[FFT_NUM/(UF*2)];
    complex<float> data_rev_stream_4[FFT_NUM/(UF*2)];
    complex<float> data_rev_stream_5[FFT_NUM/(UF*2)];
    complex<float> data_rev_stream_6[FFT_NUM/(UF*2)];
    complex<float> data_rev_stream_7[FFT_NUM/(UF*2)];

    
    #pragma HLS array_partition variable=data_in_cyclic type=complete dim=1 
    #pragma HLS array_partition variable=data_in_cyclic type=cyclic factor=UF dim=2
    
    #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
    const int TIME_STEP =  FFT_NUM/(UF*2);
    const int PAR =  UF*2;

    READ_STREAM_INPUT: for (int i = 0; i < TIME_STEP; i++){ 
        #pragma HLS pipeline
        #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
        hls::vector<complex<float>, UF*2> temp;
        temp = dataIn.read();
        ap_uint<EXP2_FFT> original[UF*2];
        #pragma HLS array_partition variable=original type=complete dim=1 
        for (int u = 0; u < UF*2; u++) {
            original[u] = i*UF*2+u;
        }
        ap_uint<EXP2_FFT> reversed[UF*2];
        #pragma HLS array_partition variable=reversed type=complete dim=1 
        for (int u = 0; u < UF*2; u++) {
            reversed[u] = bit_reverse<EXP2_FFT>(original[u]);
        }
        data_rev_stream_0[reversed[0]%TIME_STEP] = temp[0];
        data_rev_stream_4[reversed[1]%TIME_STEP] = temp[1];
        data_rev_stream_2[reversed[2]%TIME_STEP] = temp[2];
        data_rev_stream_6[reversed[3]%TIME_STEP] = temp[3];
        data_rev_stream_1[reversed[4]%TIME_STEP] = temp[4];
        data_rev_stream_5[reversed[5]%TIME_STEP] = temp[5];
        data_rev_stream_3[reversed[6]%TIME_STEP] = temp[6];
        data_rev_stream_7[reversed[7]%TIME_STEP] = temp[7];
    }

    

    // complex<float> data_rev_stream[UF*2][FFT_NUM/(UF*2)];
    // // print data_rev_stream
    // for (int i = 0; i < UF*2; i++){
    //     for(int j = 0; j < TIME_STEP; j++){
    //         data_rev_stream[i][j] = i * TIME_STEP + j;
    //     }
    // }

    
    // for(int j = 0; j < TIME_STEP; j++){
    //     data_rev_stream_0[j] = data_rev_stream[0][j];
    //     data_rev_stream_1[j] = data_rev_stream[1][j];
    //     data_rev_stream_2[j] = data_rev_stream[2][j];
    //     data_rev_stream_3[j] = data_rev_stream[3][j];
    //     data_rev_stream_4[j] = data_rev_stream[4][j];
    //     data_rev_stream_5[j] = data_rev_stream[5][j];
    //     data_rev_stream_6[j] = data_rev_stream[6][j];
    //     data_rev_stream_7[j] = data_rev_stream[7][j];
    // }
    
    
    FROM_BLOCK_TO_CYCLIC: for (int i = 0; i < TIME_STEP; i= i + 1){ 
        #pragma HLS pipeline
        #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
        int offset[UF*2];
        #pragma HLS array_partition variable=offset type=complete dim=1
        for (int u = 0; u < UF*2; u++) {
            offset[u] = (i+u)%TIME_STEP;
        }
        int cyclic_offset[UF*2];
        #pragma HLS array_partition variable=cyclic_offset type=complete dim=1
        complex<float> block_data[UF*2];
        #pragma HLS array_partition variable=block_data type=complete dim=1
        // for (int u = 0; u < UF*2; u++) {
        //     block_data[u] = data_rev_stream[u][offset[u]];
        // }
        block_data[0] = data_rev_stream_0[offset[0]];
        block_data[1] = data_rev_stream_1[offset[1]];
        block_data[2] = data_rev_stream_2[offset[2]];
        block_data[3] = data_rev_stream_3[offset[3]];
        block_data[4] = data_rev_stream_4[offset[4]];
        block_data[5] = data_rev_stream_5[offset[5]];
        block_data[6] = data_rev_stream_6[offset[6]];
        block_data[7] = data_rev_stream_7[offset[7]];

        complex<float> cyclic_data[UF*2];
        #pragma HLS array_partition variable=cyclic_data type=complete dim=1

        if (i%PAR ==0 ){
            cyclic_data[0] = block_data[0];
            cyclic_data[1] = block_data[1];
            cyclic_data[2] = block_data[2];
            cyclic_data[3] = block_data[3];
            cyclic_data[4] = block_data[4];
            cyclic_data[5] = block_data[5];
            cyclic_data[6] = block_data[6];
            cyclic_data[7] = block_data[7];
            
            cyclic_offset[0] = i/PAR ;
            cyclic_offset[1] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
            cyclic_offset[2] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR;
            cyclic_offset[3] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
            cyclic_offset[4] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
            cyclic_offset[5] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            cyclic_offset[6] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR;
            cyclic_offset[7] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR;

        }else if (i%PAR ==1 ){
            cyclic_data[0] = block_data[7];
            cyclic_data[1] = block_data[0];
            cyclic_data[2] = block_data[1];
            cyclic_data[3] = block_data[2];
            cyclic_data[4] = block_data[3];
            cyclic_data[5] = block_data[4];
            cyclic_data[6] = block_data[5];
            cyclic_data[7] = block_data[6];

            cyclic_offset[0] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR;
            cyclic_offset[1] = i/PAR; 
            cyclic_offset[2] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
            cyclic_offset[3] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR; 
            cyclic_offset[4] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
            cyclic_offset[5] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
            cyclic_offset[6] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            cyclic_offset[7] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR;

        }else if (i%PAR ==2 ){
            cyclic_data[0] = block_data[6];
            cyclic_data[1] = block_data[7];
            cyclic_data[2] = block_data[0];
            cyclic_data[3] = block_data[1];
            cyclic_data[4] = block_data[2];
            cyclic_data[5] = block_data[3];
            cyclic_data[6] = block_data[4];
            cyclic_data[7] = block_data[5];

            cyclic_offset[0] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR; 
            cyclic_offset[1] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR;
            cyclic_offset[2] = i/PAR;  
            cyclic_offset[3] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
            cyclic_offset[4] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR;
            cyclic_offset[5] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
            cyclic_offset[6] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
            cyclic_offset[7] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            
        }else if (i%PAR ==3 ){
            cyclic_data[0] = block_data[5];
            cyclic_data[1] = block_data[6];
            cyclic_data[2] = block_data[7];
            cyclic_data[3] = block_data[0];
            cyclic_data[4] = block_data[1];
            cyclic_data[5] = block_data[2];
            cyclic_data[6] = block_data[3];
            cyclic_data[7] = block_data[4];

            cyclic_offset[0] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            cyclic_offset[1] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR;
            cyclic_offset[2] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR; 
            cyclic_offset[3] = i/PAR; 
            cyclic_offset[4] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
            cyclic_offset[5] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR;
            cyclic_offset[6] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
            cyclic_offset[7] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
        }else if  (i%PAR ==4 ){
            cyclic_data[0] = block_data[4];
            cyclic_data[1] = block_data[5];
            cyclic_data[2] = block_data[6];
            cyclic_data[3] = block_data[7];
            cyclic_data[4] = block_data[0];
            cyclic_data[5] = block_data[1];
            cyclic_data[6] = block_data[2];
            cyclic_data[7] = block_data[3];

            cyclic_offset[0] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
            cyclic_offset[1] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            cyclic_offset[2] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR;
            cyclic_offset[3] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR;
            cyclic_offset[4] = i/PAR;
            cyclic_offset[5] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
            cyclic_offset[6] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR;
            cyclic_offset[7] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
        }else if  (i%PAR ==5 ){
            cyclic_data[0] = block_data[3];
            cyclic_data[1] = block_data[4];
            cyclic_data[2] = block_data[5];
            cyclic_data[3] = block_data[6];
            cyclic_data[4] = block_data[7];
            cyclic_data[5] = block_data[0];
            cyclic_data[6] = block_data[1];
            cyclic_data[7] = block_data[2];

            cyclic_offset[0] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
            cyclic_offset[1] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
            cyclic_offset[2] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            cyclic_offset[3] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR;
            cyclic_offset[4] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR;
            cyclic_offset[5] = i/PAR;
            cyclic_offset[6] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
            cyclic_offset[7] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR;
        }else if  (i%PAR ==6 ){
            cyclic_data[0] = block_data[2];
            cyclic_data[1] = block_data[3];
            cyclic_data[2] = block_data[4];
            cyclic_data[3] = block_data[5];
            cyclic_data[4] = block_data[6];
            cyclic_data[5] = block_data[7];
            cyclic_data[6] = block_data[0];
            cyclic_data[7] = block_data[1];

            cyclic_offset[0] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR;
            cyclic_offset[1] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
            cyclic_offset[2] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
            cyclic_offset[3] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            cyclic_offset[4] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR;
            cyclic_offset[5] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR;
            cyclic_offset[6] = i/PAR;
            cyclic_offset[7] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
        }else if  (i%PAR ==7 ){
            cyclic_data[0] = block_data[1];
            cyclic_data[1] = block_data[2];
            cyclic_data[2] = block_data[3];
            cyclic_data[3] = block_data[4];
            cyclic_data[4] = block_data[5];
            cyclic_data[5] = block_data[6];
            cyclic_data[6] = block_data[7];
            cyclic_data[7] = block_data[0];

            cyclic_offset[0] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
            cyclic_offset[1] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR;
            cyclic_offset[2] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
            cyclic_offset[3] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
            cyclic_offset[4] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            cyclic_offset[5] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR;
            cyclic_offset[6] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR;
            cyclic_offset[7] = i/PAR;
        }

        for (int u = 0; u < UF*2; u++) {
            data_in_cyclic[u][cyclic_offset[u]] = cyclic_data[u];
        }

        // std::cout << "reference " ;  
        // for (int u = 0; u < UF*2; u++) {
        //     std::cout <<  offset[u] << " ";
        // }
        // std::cout << " current " ;
        // for (int u = 0; u < UF*2; u++) {
        //     std::cout <<  cyclic_offset[u] << " ";
        // }
        // std::cout << " recover reference  " ;
        // for (int u = 0; u < UF*2; u++) {
        //     std::cout <<  offset[u]+u*TIME_STEP << " ";
        // }
        // std::cout << " recover current " ;
        // for (int u = 0; u < UF*2; u++) {
        //     std::cout <<  cyclic_offset[u]*PAR+u << " ";
        // }
        // std::cout << std::endl;

    }


    // print data_in_cyclic
    // for(int j = 0; j < TIME_STEP; j++){
    //     for (int i = 0; i < UF*2; i++){
    //         int index = j*UF*2+i;
    //         // if(data_in_cyclic[i][j] != data_rev_stream[index/TIME_STEP][index%TIME_STEP]) cout << "!!! " << i << " " << j << " " << endl;
    //         cout << "index " << index << ": rev "<< data_rev_stream[index/TIME_STEP][index%TIME_STEP] << " cyclic " << data_in_cyclic[i][j] << endl;
    //     }
    // }

    // for (int i = 0; i < TIME_STEP; i= i + 1){ 
    //     for (int u = 0; u < UF*2; u++) {
    //         // data_rev_stream[u][i] = data_in_cyclic[u][i];
    //         int index = i * UF*2 + u;
    //         data_in_cyclic[u][i] = data_rev_stream[index/32][index%32];
    //     }
    // }

    STREAM_OUT_REVERSE: for (int i = 0; i < TIME_STEP; i= i + 1){ 
        #pragma HLS pipeline
        #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle  
        hls::vector<complex<float>, UF*2> temp;
        for (int u = 0; u < UF*2; u++) {
            temp[u] = data_in_cyclic[u][i];
        }
        reverse_in_stream_vector.write(temp);
    }
}


template<int stage> void FFT_stage_spatial_unroll(complex<float> data_ld[FFT_NUM], complex<float> data_st[FFT_NUM]){
    int bflySize = 1 << stage; //from 2 to FFT_NUM
    int bflyStep = bflySize >> 1; //from 1 to FFT_NUM/2
    int indexMultiply = FFT_NUM >> stage;
    
    if(bflyStep < UF){
        L_Pair_loop: 
        for (uint16_t m = 0; m < FFT_NUM; m += bflySize) {
            #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
            #pragma HLS unroll factor=UF>>(stage-1)
            #pragma HLS pipeline
            L_Group_loop: 
            for (uint16_t k = 0; k < bflyStep; ++k) {
                // Butterfly computations
                uint16_t index = indexMultiply * k;
                #pragma HLS bind_op variable=index op=mul impl=fabric
                auto tw = twiddles[index];   
                uint16_t i0 = m + k; 
                uint16_t i1 = m + k + bflyStep;
                RADIX2_BFLY_double_buffer_quarter_CY(data_ld, data_st, i0, i1, k > 0 && k == bflyStep>>1, k > 0, tw);  
            }
        }
    }
    else if (bflySize == FFT_NUM){
            R_Group_loop_bflySize_equal_FFT_NUM : 
            for (uint16_t k = 0; k < bflyStep; ++k) {
                #pragma HLS pipeline
                #pragma HLS unroll factor=UF
                #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
                // Butterfly computations
                uint16_t index = indexMultiply * k;
                #pragma HLS bind_op variable=index op=mul impl=fabric
                auto tw = twiddles[index];   
                uint16_t i0 = 0 + k; 
                uint16_t i1 = 0 + k + bflyStep;
                RADIX2_BFLY_double_buffer_quarter_CY(data_ld, data_st, i0, i1, k > 0 && k == bflyStep>>1, k > 0, tw);  
            }
    }
    else{
        R_Pair_loop: 
        for (uint16_t m = 0; m < FFT_NUM; m += bflySize) {
            #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
            R_Group_loop: 
            for (uint16_t k = 0; k < bflyStep; ++k) {
                #pragma HLS pipeline
                #pragma HLS unroll factor=UF
                // Butterfly computations
                uint16_t index = indexMultiply * k;
                #pragma HLS bind_op variable=index op=mul impl=fabric
                auto tw = twiddles[index];   
                uint16_t i0 = m + k; 
                uint16_t i1 = m + k + bflyStep;
                #pragma HLS array_partition variable=twiddles complete
                RADIX2_BFLY_double_buffer_quarter_CY(data_ld, data_st, i0, i1, k > 0 && k == bflyStep>>1, k > 0, tw);                  
            }
        }
    }
}


void RADIX2_BFLY_double_buffer_quarter_onlycompute(complex<float> d0, complex<float> d1, complex<float>& data_out0, complex<float>& data_out1, bool inv_i1_enable, bool tw_enable, complex<float> tw) 
{
#pragma HLS inline
// complex<float> d0 = data_ld[i0];                     
// complex<float> d1 = data_ld[i1];
float d0_real, d0_imag, d1_real, d1_imag, d2_real, d2_imag, d3_real, d3_imag;
if(inv_i1_enable){
    // d1 = complex<float>(d1.imag(), -d1.real());
    d1_real = d1.imag();
    d1_imag = - d1.real();
}
else if(tw_enable){
    #pragma HLS bind_op variable=d1_real op=fsub impl=fabric 
    #pragma HLS bind_op variable=d1_imag op=fadd impl=fabric 
    float a, b, c, d;
    a = d1.real(); b = d1.imag(); c = tw.real(); d = tw.imag();
    float ac, bd, ad, bc; 
    ac = a*c;
    bd = b*d;
    ad = a*d;
    bc = b*c;
    // ac = a;
    // bd = b;
    // ad = c;
    // bc = d;
    d1_real = ac - bd; 
    d1_imag = ad + bc;         
}else{
    d1_real = d1.real();
    d1_imag = d1.imag();
}

// #pragma HLS bind_op variable=data_st op=fadd impl=dsp
// data_st[i0] = d0 + d1;
// #pragma HLS bind_op variable=data_st op=fsub impl=dsp                     
// data_st[i1] = d0 - d1;     

#pragma HLS bind_op variable=d2_real op=fadd impl=fabric 
#pragma HLS bind_op variable=d2_imag op=fadd impl=fabric 
#pragma HLS bind_op variable=d3_real op=fsub impl=fabric 
#pragma HLS bind_op variable=d3_imag op=fsub impl=fabric 

d0_real = d0.real();
d0_imag = d0.imag();
d2_real = d0_real + d1_real;
d2_imag = d0_imag + d1_imag;
d3_real = d0_real - d1_real;
d3_imag = d0_imag - d1_imag;
// data_st[i0] = complex<float>(d2_real, d2_imag);
// data_st[i1] = complex<float>(d3_real, d3_imag);

data_out0 = complex<float>(d2_real, d2_imag);
data_out1 = complex<float>(d3_real, d3_imag);
}



void FFT_Stage1_vectorstream_parameterize(
    hls::stream<hls::vector<complex<float>, UF*2>> & reverse_in_stream_vector,
    hls::stream<hls::vector<complex<float>, UF*2>> & data_s1_stream_vector
){
    #pragma HLS dataflow disable_start_propagation
    #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
    FFT_Stage1: for (int m = 0; m < FFT_NUM/(2*UF); m += 1) {
        #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
        #pragma HLS pipeline 
        // Butterfly computations
        auto tw = complex<float>(0,0); 
        hls::vector<complex<float>, UF*2> data = reverse_in_stream_vector.read();
        hls::vector<complex<float>, UF*2> data_out;
        for (int i = 0; i < UF; i++){
            auto data0 = data[i*2];
            auto data1 = data[i*2+1];
            complex<float> data_out0, data_out1;
            RADIX2_BFLY_double_buffer_quarter_onlycompute(data0, data1, data_out0, data_out1, false, false, tw); 
            data_out[i*2] = data_out0;
            data_out[i*2+1] = data_out1;
        }
        data_s1_stream_vector.write(data_out);
    }

}


void FFT_Stage2_vectorstreamIn_arrayOut_parametize(
    hls::stream<hls::vector<complex<float>, UF*2>> & data_s1_stream_vector,
    complex<float> data_2[FFT_NUM]
){

#pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
FFT_Stage2:  for (int m = 0; m < FFT_NUM; m += 2*UF) {
        #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
        #pragma HLS pipeline
        hls::vector<complex<float>, UF*2> data = data_s1_stream_vector.read();
        for (int i = 0; i < UF*2; i+= 4){
            auto index0 = 64*0;
            auto index1 = 64*1;
            auto tw0 = twiddles[index0];   
            auto tw1 = twiddles[index1];   
            auto data0 = data[i];
            auto data1 = data[i+2];
            auto data2 = data[i+1];
            auto data3 = data[i+3];
            complex<float> data_out0, data_out1, data_out2, data_out3;
            RADIX2_BFLY_double_buffer_quarter_onlycompute(data0, data1, data_out0, data_out1, false,  0 > 0, tw0); 
            RADIX2_BFLY_double_buffer_quarter_onlycompute(data2, data3, data_out2, data_out3, true,  1 > 0, tw1); 
            data_2[m+i] =  data_out0;
            data_2[m+i+2] =  data_out1;
            data_2[m+i+1] =  data_out2;
            data_2[m+i+3] =  data_out3;
        }
    }
}


void FFT_DIT_spatial_unroll_CY_stream_vector( 
    hls::stream<hls::vector<complex<float>, UF*2>> & dataIn,  
    hls::stream<hls::vector<complex<float>, UF*2>> & dataOut
){

    #pragma HLS dataflow disable_start_propagation
    #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle

    static complex<float> data_0[FFT_NUM];
    static complex<float> data_1[FFT_NUM];
    static complex<float> data_2[FFT_NUM];
    static complex<float> data_3[FFT_NUM];
    static complex<float> data_4[FFT_NUM];
    static complex<float> data_5[FFT_NUM];
    static complex<float> data_6[FFT_NUM];
    static complex<float> data_7[FFT_NUM];
    static complex<float> data_8[FFT_NUM];


    #pragma HLS array_partition variable=data_0 type=cyclic factor=UF*2 dim=1
    #pragma HLS bind_storage variable=data_0 type=RAM_2P impl=LUTRAM
    #pragma HLS array_partition variable=data_1 type=cyclic factor=UF*2 dim=1
    #pragma HLS bind_storage variable=data_1 type=RAM_2P impl=LUTRAM
    #pragma HLS array_partition variable=data_2 type=cyclic factor=UF dim=1
    #pragma HLS array_partition variable=data_3 type=cyclic factor=UF dim=1
    #pragma HLS array_partition variable=data_4 type=cyclic factor=UF dim=1
    #pragma HLS array_partition variable=data_5 type=cyclic factor=UF dim=1
    #pragma HLS array_partition variable=data_6 type=cyclic factor=UF dim=1
    #pragma HLS array_partition variable=data_7 type=cyclic factor=UF dim=1
    #pragma HLS array_partition variable=data_8 type=cyclic factor=UF dim=1



    hls::stream<hls::vector<complex<float>, UF*2>> reverse_in_stream_vector; 
    hls::stream<hls::vector<complex<float>, UF*2>> data_s1_stream_vector; 

    reverse_input_stream_UF4(dataIn, reverse_in_stream_vector);
    FFT_Stage1_vectorstream_parameterize (reverse_in_stream_vector, data_s1_stream_vector);
    FFT_Stage2_vectorstreamIn_arrayOut_parametize (data_s1_stream_vector, data_0);
    FFT_stage_spatial_unroll<3>(data_0, data_1);
    FFT_stage_spatial_unroll<4>(data_1, data_2);
    FFT_stage_spatial_unroll<5>(data_2, data_3);
    FFT_stage_spatial_unroll<6>(data_3, data_4);
    FFT_stage_spatial_unroll<7>(data_4, data_5);
    FFT_stage_spatial_unroll<8>(data_5, data_6);
    FFT_stage_spatial_unroll<9>(data_6, data_7);
    FFT_stage_spatial_unroll<10>(data_7, data_8);
    output_result_array_to_stream (data_8, dataOut);

}


void FFT_TOP(hls::stream<hls::vector<complex<float>, UF*2>> & in, 
    hls::stream<hls::vector<complex<float>, UF*2>> & out
){

    #pragma HLS dataflow disable_start_propagation
    LOOP_TWIDDLES: for (int i = 0; i < FFT_NUM/2; ++i) {
        double angle = -2 * PI * i / FFT_NUM;
        auto tw = complex<float>(cos(angle), sin(angle));
        twiddles[i] = (complex<float>)tw;
    }
    LOOP_REVIDTAB: for (int i = 0; i < FFT_NUM; ++i) {
        int reversed = 0;
        int number = i;
        for (int j = 0; j < EXP2_FFT; ++j) {
            reversed = reversed * 2 + (number & 1);
            number >>= 1;
        }
        revIdxTab[i] = reversed;
    }
    FFT_DIT_spatial_unroll_CY_stream_vector(in, out);
}