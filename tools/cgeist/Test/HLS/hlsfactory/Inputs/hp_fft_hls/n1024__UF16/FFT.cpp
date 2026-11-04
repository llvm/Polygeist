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


void reverse_read_stream_input (
    hls::stream<hls::vector<complex<float>, UF*2>> & dataIn,
    complex<float> data_rev_stream[UF*2][FFT_NUM/(UF*2)]
){

    const int TIME_STEP = FFT_NUM/(UF*2);
    const int PAR = UF*2;

    READ_STREAM_INPUT: for (int i = 0; i < TIME_STEP; i++){
        #pragma HLS pipeline II=1
        // #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
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
        data_rev_stream[0][reversed[0]%TIME_STEP] = temp[0];
        data_rev_stream[16][reversed[1]%TIME_STEP] = temp[1];
        data_rev_stream[8][reversed[2]%TIME_STEP] = temp[2];
        data_rev_stream[24][reversed[3]%TIME_STEP] = temp[3];
        data_rev_stream[4][reversed[4]%TIME_STEP] = temp[4];
        data_rev_stream[20][reversed[5]%TIME_STEP] = temp[5];
        data_rev_stream[12][reversed[6]%TIME_STEP] = temp[6];
        data_rev_stream[28][reversed[7]%TIME_STEP] = temp[7];
        data_rev_stream[2][reversed[8]%TIME_STEP] = temp[8];
        data_rev_stream[18][reversed[9]%TIME_STEP] = temp[9];
        data_rev_stream[10][reversed[10]%TIME_STEP] = temp[10];
        data_rev_stream[26][reversed[11]%TIME_STEP] = temp[11];
        data_rev_stream[6][reversed[12]%TIME_STEP] = temp[12];
        data_rev_stream[22][reversed[13]%TIME_STEP] = temp[13];
        data_rev_stream[14][reversed[14]%TIME_STEP] = temp[14];
        data_rev_stream[30][reversed[15]%TIME_STEP] = temp[15];
        data_rev_stream[1][reversed[16]%TIME_STEP] = temp[16];
        data_rev_stream[17][reversed[17]%TIME_STEP] = temp[17];
        data_rev_stream[9][reversed[18]%TIME_STEP] = temp[18];
        data_rev_stream[25][reversed[19]%TIME_STEP] = temp[19];
        data_rev_stream[5][reversed[20]%TIME_STEP] = temp[20];
        data_rev_stream[21][reversed[21]%TIME_STEP] = temp[21];
        data_rev_stream[13][reversed[22]%TIME_STEP] = temp[22];
        data_rev_stream[29][reversed[23]%TIME_STEP] = temp[23];
        data_rev_stream[3][reversed[24]%TIME_STEP] = temp[24];
        data_rev_stream[19][reversed[25]%TIME_STEP] = temp[25];
        data_rev_stream[11][reversed[26]%TIME_STEP] = temp[26];
        data_rev_stream[27][reversed[27]%TIME_STEP] = temp[27];
        data_rev_stream[7][reversed[28]%TIME_STEP] = temp[28];
        data_rev_stream[23][reversed[29]%TIME_STEP] = temp[29];
        data_rev_stream[15][reversed[30]%TIME_STEP] = temp[30];
        data_rev_stream[31][reversed[31]%TIME_STEP] = temp[31];
    }

    
}


void reverse_from_block_to_cyclic (
    complex<float> data_rev_stream[UF*2][FFT_NUM/(UF*2)],
    complex<float> data_in_cyclic[UF*2][FFT_NUM/(UF*2)]
){
    const int TIME_STEP =  FFT_NUM/(UF*2);
    const int PAR =  UF*2;
    #pragma inline off


    FROM_BLOCK_TO_CYCLIC: for (int i = 0; i < TIME_STEP; i= i + 1){ 
        #pragma HLS pipeline II=1
        // #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
        int offset[UF*2];
        #pragma HLS array_partition variable=offset type=complete dim=1
        for (int u = 0; u < UF*2; u++) {
            offset[u] = (i+u)%TIME_STEP;
        }
        int cyclic_offset[UF*2];
        #pragma HLS array_partition variable=cyclic_offset type=complete dim=1
        complex<float> block_data[UF*2];
        #pragma HLS array_partition variable=block_data type=complete dim=1
        for (int u = 0; u < UF*2; u++) {
            block_data[u] = data_rev_stream[u][offset[u]];
        }
    
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
            cyclic_data[8] = block_data[8];
            cyclic_data[9] = block_data[9];
            cyclic_data[10] = block_data[10];
            cyclic_data[11] = block_data[11];
            cyclic_data[12] = block_data[12];
            cyclic_data[13] = block_data[13];
            cyclic_data[14] = block_data[14];
            cyclic_data[15] = block_data[15];
            cyclic_data[16] = block_data[16];
            cyclic_data[17] = block_data[17];
            cyclic_data[18] = block_data[18];
            cyclic_data[19] = block_data[19];
            cyclic_data[20] = block_data[20];
            cyclic_data[21] = block_data[21];
            cyclic_data[22] = block_data[22];
            cyclic_data[23] = block_data[23];
            cyclic_data[24] = block_data[24];
            cyclic_data[25] = block_data[25];
            cyclic_data[26] = block_data[26];
            cyclic_data[27] = block_data[27];
            cyclic_data[28] = block_data[28];
            cyclic_data[29] = block_data[29];
            cyclic_data[30] = block_data[30];
            cyclic_data[31] = block_data[31];

            cyclic_offset[0] = i/PAR ;
            cyclic_offset[1] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
            cyclic_offset[2] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR;
            cyclic_offset[3] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
            cyclic_offset[4] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
            cyclic_offset[5] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            cyclic_offset[6] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR;
            cyclic_offset[7] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR;
            cyclic_offset[8] = ((i+8)%TIME_STEP+TIME_STEP*8)/PAR;
            cyclic_offset[9] = ((i+9)%TIME_STEP+TIME_STEP*9)/PAR;
            cyclic_offset[10] = ((i+10)%TIME_STEP+TIME_STEP*10)/PAR;
            cyclic_offset[11] = ((i+11)%TIME_STEP+TIME_STEP*11)/PAR;
            cyclic_offset[12] = ((i+12)%TIME_STEP+TIME_STEP*12)/PAR;
            cyclic_offset[13] = ((i+13)%TIME_STEP+TIME_STEP*13)/PAR;
            cyclic_offset[14] = ((i+14)%TIME_STEP+TIME_STEP*14)/PAR;
            cyclic_offset[15] = ((i+15)%TIME_STEP+TIME_STEP*15)/PAR;
            cyclic_offset[16] = ((i+16)%TIME_STEP+TIME_STEP*16)/PAR;
            cyclic_offset[17] = ((i+17)%TIME_STEP+TIME_STEP*17)/PAR;
            cyclic_offset[18] = ((i+18)%TIME_STEP+TIME_STEP*18)/PAR;
            cyclic_offset[19] = ((i+19)%TIME_STEP+TIME_STEP*19)/PAR;
            cyclic_offset[20] = ((i+20)%TIME_STEP+TIME_STEP*20)/PAR;
            cyclic_offset[21] = ((i+21)%TIME_STEP+TIME_STEP*21)/PAR;
            cyclic_offset[22] = ((i+22)%TIME_STEP+TIME_STEP*22)/PAR;
            cyclic_offset[23] = ((i+23)%TIME_STEP+TIME_STEP*23)/PAR;
            cyclic_offset[24] = ((i+24)%TIME_STEP+TIME_STEP*24)/PAR;
            cyclic_offset[25] = ((i+25)%TIME_STEP+TIME_STEP*25)/PAR;
            cyclic_offset[26] = ((i+26)%TIME_STEP+TIME_STEP*26)/PAR;
            cyclic_offset[27] = ((i+27)%TIME_STEP+TIME_STEP*27)/PAR;
            cyclic_offset[28] = ((i+28)%TIME_STEP+TIME_STEP*28)/PAR;
            cyclic_offset[29] = ((i+29)%TIME_STEP+TIME_STEP*29)/PAR;
            cyclic_offset[30] = ((i+30)%TIME_STEP+TIME_STEP*30)/PAR;
            cyclic_offset[31] = ((i+31)%TIME_STEP+TIME_STEP*31)/PAR;

        }else if (i%PAR ==1 ){
            cyclic_data[0] = block_data[31];
            cyclic_data[1] = block_data[0];
            cyclic_data[2] = block_data[1];
            cyclic_data[3] = block_data[2];
            cyclic_data[4] = block_data[3];
            cyclic_data[5] = block_data[4];
            cyclic_data[6] = block_data[5];
            cyclic_data[7] = block_data[6];
            cyclic_data[8] = block_data[7];
            cyclic_data[9] = block_data[8];
            cyclic_data[10] = block_data[9];
            cyclic_data[11] = block_data[10];
            cyclic_data[12] = block_data[11];
            cyclic_data[13] = block_data[12];
            cyclic_data[14] = block_data[13];
            cyclic_data[15] = block_data[14];
            cyclic_data[16] = block_data[15];
            cyclic_data[17] = block_data[16];
            cyclic_data[18] = block_data[17];
            cyclic_data[19] = block_data[18];
            cyclic_data[20] = block_data[19];
            cyclic_data[21] = block_data[20];
            cyclic_data[22] = block_data[21];
            cyclic_data[23] = block_data[22];
            cyclic_data[24] = block_data[23];
            cyclic_data[25] = block_data[24];
            cyclic_data[26] = block_data[25];
            cyclic_data[27] = block_data[26];
            cyclic_data[28] = block_data[27];
            cyclic_data[29] = block_data[28];
            cyclic_data[30] = block_data[29];
            cyclic_data[31] = block_data[30];
            
            cyclic_offset[0] = ((i+31)%TIME_STEP+TIME_STEP*31)/PAR;
            cyclic_offset[1] = i/PAR;
            cyclic_offset[2] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
            cyclic_offset[3] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR;
            cyclic_offset[4] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
            cyclic_offset[5] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
            cyclic_offset[6] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            cyclic_offset[7] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR;
            cyclic_offset[8] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR;
            cyclic_offset[9] = ((i+8)%TIME_STEP+TIME_STEP*8)/PAR;
            cyclic_offset[10] = ((i+9)%TIME_STEP+TIME_STEP*9)/PAR;
            cyclic_offset[11] = ((i+10)%TIME_STEP+TIME_STEP*10)/PAR;
            cyclic_offset[12] = ((i+11)%TIME_STEP+TIME_STEP*11)/PAR;
            cyclic_offset[13] = ((i+12)%TIME_STEP+TIME_STEP*12)/PAR;
            cyclic_offset[14] = ((i+13)%TIME_STEP+TIME_STEP*13)/PAR;
            cyclic_offset[15] = ((i+14)%TIME_STEP+TIME_STEP*14)/PAR;
            cyclic_offset[16] = ((i+15)%TIME_STEP+TIME_STEP*15)/PAR;
            cyclic_offset[17] = ((i+16)%TIME_STEP+TIME_STEP*16)/PAR;
            cyclic_offset[18] = ((i+17)%TIME_STEP+TIME_STEP*17)/PAR;
            cyclic_offset[19] = ((i+18)%TIME_STEP+TIME_STEP*18)/PAR;
            cyclic_offset[20] = ((i+19)%TIME_STEP+TIME_STEP*19)/PAR;
            cyclic_offset[21] = ((i+20)%TIME_STEP+TIME_STEP*20)/PAR;
            cyclic_offset[22] = ((i+21)%TIME_STEP+TIME_STEP*21)/PAR;
            cyclic_offset[23] = ((i+22)%TIME_STEP+TIME_STEP*22)/PAR;
            cyclic_offset[24] = ((i+23)%TIME_STEP+TIME_STEP*23)/PAR;
            cyclic_offset[25] = ((i+24)%TIME_STEP+TIME_STEP*24)/PAR;
            cyclic_offset[26] = ((i+25)%TIME_STEP+TIME_STEP*25)/PAR;
            cyclic_offset[27] = ((i+26)%TIME_STEP+TIME_STEP*26)/PAR;
            cyclic_offset[28] = ((i+27)%TIME_STEP+TIME_STEP*27)/PAR;
            cyclic_offset[29] = ((i+28)%TIME_STEP+TIME_STEP*28)/PAR;
            cyclic_offset[30] = ((i+29)%TIME_STEP+TIME_STEP*29)/PAR;
            cyclic_offset[31] = ((i+30)%TIME_STEP+TIME_STEP*30)/PAR;
            
        }else if (i%PAR ==2 ){
            cyclic_data[0] = block_data[30];
            cyclic_data[1] = block_data[31];
            cyclic_data[2] = block_data[0];
            cyclic_data[3] = block_data[1];
            cyclic_data[4] = block_data[2];
            cyclic_data[5] = block_data[3];
            cyclic_data[6] = block_data[4];
            cyclic_data[7] = block_data[5];
            cyclic_data[8] = block_data[6];
            cyclic_data[9] = block_data[7];
            cyclic_data[10] = block_data[8];
            cyclic_data[11] = block_data[9];
            cyclic_data[12] = block_data[10];
            cyclic_data[13] = block_data[11];
            cyclic_data[14] = block_data[12];
            cyclic_data[15] = block_data[13];
            cyclic_data[16] = block_data[14];
            cyclic_data[17] = block_data[15];
            cyclic_data[18] = block_data[16];
            cyclic_data[19] = block_data[17];
            cyclic_data[20] = block_data[18];
            cyclic_data[21] = block_data[19];
            cyclic_data[22] = block_data[20];
            cyclic_data[23] = block_data[21];
            cyclic_data[24] = block_data[22];
            cyclic_data[25] = block_data[23];
            cyclic_data[26] = block_data[24];
            cyclic_data[27] = block_data[25];
            cyclic_data[28] = block_data[26];
            cyclic_data[29] = block_data[27];
            cyclic_data[30] = block_data[28];
            cyclic_data[31] = block_data[29];

        
            cyclic_offset[0] = ((i+14)%TIME_STEP+TIME_STEP*30)/PAR;
            cyclic_offset[1] = ((i+15)%TIME_STEP+TIME_STEP*31)/PAR;
            cyclic_offset[2] = i/PAR;
            cyclic_offset[3] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
            cyclic_offset[4] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR;
            cyclic_offset[5] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
            cyclic_offset[6] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
            cyclic_offset[7] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            cyclic_offset[8] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR;
            cyclic_offset[9] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR;
            cyclic_offset[10] = ((i+8)%TIME_STEP+TIME_STEP*8)/PAR;
            cyclic_offset[11] = ((i+9)%TIME_STEP+TIME_STEP*9)/PAR;
            cyclic_offset[12] = ((i+10)%TIME_STEP+TIME_STEP*10)/PAR;
            cyclic_offset[13] = ((i+11)%TIME_STEP+TIME_STEP*11)/PAR;
            cyclic_offset[14] = ((i+12)%TIME_STEP+TIME_STEP*12)/PAR;
            cyclic_offset[15] = ((i+13)%TIME_STEP+TIME_STEP*13)/PAR;
            cyclic_offset[16] = ((i+14)%TIME_STEP+TIME_STEP*14)/PAR;
            cyclic_offset[17] = ((i+15)%TIME_STEP+TIME_STEP*15)/PAR;
            cyclic_offset[18] = ((i+16)%TIME_STEP+TIME_STEP*16)/PAR;
            cyclic_offset[19] = ((i+17)%TIME_STEP+TIME_STEP*17)/PAR;
            cyclic_offset[20] = ((i+18)%TIME_STEP+TIME_STEP*18)/PAR;
            cyclic_offset[21] = ((i+19)%TIME_STEP+TIME_STEP*19)/PAR;
            cyclic_offset[22] = ((i+20)%TIME_STEP+TIME_STEP*20)/PAR;
            cyclic_offset[23] = ((i+21)%TIME_STEP+TIME_STEP*21)/PAR;
            cyclic_offset[24] = ((i+22)%TIME_STEP+TIME_STEP*22)/PAR;
            cyclic_offset[25] = ((i+23)%TIME_STEP+TIME_STEP*23)/PAR;
            cyclic_offset[26] = ((i+24)%TIME_STEP+TIME_STEP*24)/PAR;
            cyclic_offset[27] = ((i+25)%TIME_STEP+TIME_STEP*25)/PAR;
            cyclic_offset[28] = ((i+26)%TIME_STEP+TIME_STEP*26)/PAR;
            cyclic_offset[29] = ((i+27)%TIME_STEP+TIME_STEP*27)/PAR;
            cyclic_offset[30] = ((i+28)%TIME_STEP+TIME_STEP*28)/PAR;
            cyclic_offset[31] = ((i+29)%TIME_STEP+TIME_STEP*29)/PAR;

        }else if (i%PAR ==3 ){
            cyclic_data[0] = block_data[29];
            cyclic_data[1] = block_data[30];
            cyclic_data[2] = block_data[31];
            cyclic_data[3] = block_data[0];
            cyclic_data[4] = block_data[1];
            cyclic_data[5] = block_data[2];
            cyclic_data[6] = block_data[3];
            cyclic_data[7] = block_data[4];
            cyclic_data[8] = block_data[5];
            cyclic_data[9] = block_data[6];
            cyclic_data[10] = block_data[7];
            cyclic_data[11] = block_data[8];
            cyclic_data[12] = block_data[9];
            cyclic_data[13] = block_data[10];
            cyclic_data[14] = block_data[11];
            cyclic_data[15] = block_data[12];
            cyclic_data[16] = block_data[13];
            cyclic_data[17] = block_data[14];
            cyclic_data[18] = block_data[15];
            cyclic_data[19] = block_data[16];
            cyclic_data[20] = block_data[17];
            cyclic_data[21] = block_data[18];
            cyclic_data[22] = block_data[19];
            cyclic_data[23] = block_data[20];
            cyclic_data[24] = block_data[21];
            cyclic_data[25] = block_data[22];
            cyclic_data[26] = block_data[23];
            cyclic_data[27] = block_data[24];
            cyclic_data[28] = block_data[25];
            cyclic_data[29] = block_data[26];
            cyclic_data[30] = block_data[27];
            cyclic_data[31] = block_data[28];

            cyclic_offset[0] = ((i+29)%TIME_STEP+TIME_STEP*29)/PAR;
            cyclic_offset[1] = ((i+30)%TIME_STEP+TIME_STEP*30)/PAR;
            cyclic_offset[2] = ((i+31)%TIME_STEP+TIME_STEP*31)/PAR;
            cyclic_offset[3] = i/PAR;
            cyclic_offset[4] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
            cyclic_offset[5] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR;
            cyclic_offset[6] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
            cyclic_offset[7] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
            cyclic_offset[8] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            cyclic_offset[9] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR;
            cyclic_offset[10] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR;
            cyclic_offset[11] = ((i+8)%TIME_STEP+TIME_STEP*8)/PAR;
            cyclic_offset[12] = ((i+9)%TIME_STEP+TIME_STEP*9)/PAR;
            cyclic_offset[13] = ((i+10)%TIME_STEP+TIME_STEP*10)/PAR;
            cyclic_offset[14] = ((i+11)%TIME_STEP+TIME_STEP*11)/PAR;
            cyclic_offset[15] = ((i+12)%TIME_STEP+TIME_STEP*12)/PAR;
            cyclic_offset[16] = ((i+13)%TIME_STEP+TIME_STEP*13)/PAR;
            cyclic_offset[17] = ((i+14)%TIME_STEP+TIME_STEP*14)/PAR;
            cyclic_offset[18] = ((i+15)%TIME_STEP+TIME_STEP*15)/PAR;
            cyclic_offset[19] = ((i+16)%TIME_STEP+TIME_STEP*16)/PAR;
            cyclic_offset[20] = ((i+17)%TIME_STEP+TIME_STEP*17)/PAR;
            cyclic_offset[21] = ((i+18)%TIME_STEP+TIME_STEP*18)/PAR;
            cyclic_offset[22] = ((i+19)%TIME_STEP+TIME_STEP*19)/PAR;
            cyclic_offset[23] = ((i+20)%TIME_STEP+TIME_STEP*20)/PAR;
            cyclic_offset[24] = ((i+21)%TIME_STEP+TIME_STEP*21)/PAR;
            cyclic_offset[25] = ((i+22)%TIME_STEP+TIME_STEP*22)/PAR;
            cyclic_offset[26] = ((i+23)%TIME_STEP+TIME_STEP*23)/PAR;
            cyclic_offset[27] = ((i+24)%TIME_STEP+TIME_STEP*24)/PAR;
            cyclic_offset[28] = ((i+25)%TIME_STEP+TIME_STEP*25)/PAR;
            cyclic_offset[29] = ((i+26)%TIME_STEP+TIME_STEP*26)/PAR;
            cyclic_offset[30] = ((i+27)%TIME_STEP+TIME_STEP*27)/PAR;
            cyclic_offset[31] = ((i+28)%TIME_STEP+TIME_STEP*28)/PAR;

        }else if (i%PAR ==4 ){
             
            cyclic_data[0] = block_data[28];
            cyclic_data[1] = block_data[29];
            cyclic_data[2] = block_data[30];
            cyclic_data[3] = block_data[31];
            cyclic_data[4] = block_data[0];
            cyclic_data[5] = block_data[1];
            cyclic_data[6] = block_data[2];
            cyclic_data[7] = block_data[3];
            cyclic_data[8] = block_data[4];
            cyclic_data[9] = block_data[5];
            cyclic_data[10] = block_data[6];
            cyclic_data[11] = block_data[7];
            cyclic_data[12] = block_data[8];
            cyclic_data[13] = block_data[9];
            cyclic_data[14] = block_data[10];
            cyclic_data[15] = block_data[11];
            cyclic_data[16] = block_data[12];
            cyclic_data[17] = block_data[13];
            cyclic_data[18] = block_data[14];
            cyclic_data[19] = block_data[15];
            cyclic_data[20] = block_data[16];
            cyclic_data[21] = block_data[17];
            cyclic_data[22] = block_data[18];
            cyclic_data[23] = block_data[19];
            cyclic_data[24] = block_data[20];
            cyclic_data[25] = block_data[21];
            cyclic_data[26] = block_data[22];
            cyclic_data[27] = block_data[23];
            cyclic_data[28] = block_data[24];
            cyclic_data[29] = block_data[25];
            cyclic_data[30] = block_data[26];
            cyclic_data[31] = block_data[27];

            cyclic_offset[0] = ((i+28)%TIME_STEP+TIME_STEP*28)/PAR;
            cyclic_offset[1] = ((i+29)%TIME_STEP+TIME_STEP*29)/PAR;
            cyclic_offset[2] = ((i+30)%TIME_STEP+TIME_STEP*30)/PAR;
            cyclic_offset[3] = ((i+31)%TIME_STEP+TIME_STEP*31)/PAR;
            cyclic_offset[4] = i/PAR;
            cyclic_offset[5] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
            cyclic_offset[6] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR;
            cyclic_offset[7] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
            cyclic_offset[8] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
            cyclic_offset[9] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            cyclic_offset[10] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR;
            cyclic_offset[11] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR;
            cyclic_offset[12] = ((i+8)%TIME_STEP+TIME_STEP*8)/PAR;
            cyclic_offset[13] = ((i+9)%TIME_STEP+TIME_STEP*9)/PAR;
            cyclic_offset[14] = ((i+10)%TIME_STEP+TIME_STEP*10)/PAR;
            cyclic_offset[15] = ((i+11)%TIME_STEP+TIME_STEP*11)/PAR;
            cyclic_offset[16] = ((i+12)%TIME_STEP+TIME_STEP*12)/PAR;
            cyclic_offset[17] = ((i+13)%TIME_STEP+TIME_STEP*13)/PAR;
            cyclic_offset[18] = ((i+14)%TIME_STEP+TIME_STEP*14)/PAR;
            cyclic_offset[19] = ((i+15)%TIME_STEP+TIME_STEP*15)/PAR;
            cyclic_offset[20] = ((i+16)%TIME_STEP+TIME_STEP*16)/PAR;
            cyclic_offset[21] = ((i+17)%TIME_STEP+TIME_STEP*17)/PAR;
            cyclic_offset[22] = ((i+18)%TIME_STEP+TIME_STEP*18)/PAR;
            cyclic_offset[23] = ((i+19)%TIME_STEP+TIME_STEP*19)/PAR;
            cyclic_offset[24] = ((i+20)%TIME_STEP+TIME_STEP*20)/PAR;
            cyclic_offset[25] = ((i+21)%TIME_STEP+TIME_STEP*21)/PAR;
            cyclic_offset[26] = ((i+22)%TIME_STEP+TIME_STEP*22)/PAR;
            cyclic_offset[27] = ((i+23)%TIME_STEP+TIME_STEP*23)/PAR;
            cyclic_offset[28] = ((i+24)%TIME_STEP+TIME_STEP*24)/PAR;
            cyclic_offset[29] = ((i+25)%TIME_STEP+TIME_STEP*25)/PAR;
            cyclic_offset[30] = ((i+26)%TIME_STEP+TIME_STEP*26)/PAR;
            cyclic_offset[31] = ((i+27)%TIME_STEP+TIME_STEP*27)/PAR;

        }else if (i%PAR ==5 ){

            cyclic_data[0] = block_data[27];
            cyclic_data[1] = block_data[28];
            cyclic_data[2] = block_data[29];
            cyclic_data[3] = block_data[30];
            cyclic_data[4] = block_data[31];
            cyclic_data[5] = block_data[0];
            cyclic_data[6] = block_data[1];
            cyclic_data[7] = block_data[2];
            cyclic_data[8] = block_data[3];
            cyclic_data[9] = block_data[4];
            cyclic_data[10] = block_data[5];
            cyclic_data[11] = block_data[6];
            cyclic_data[12] = block_data[7];
            cyclic_data[13] = block_data[8];
            cyclic_data[14] = block_data[9];
            cyclic_data[15] = block_data[10];
            cyclic_data[16] = block_data[11];
            cyclic_data[17] = block_data[12];
            cyclic_data[18] = block_data[13];
            cyclic_data[19] = block_data[14];
            cyclic_data[20] = block_data[15];
            cyclic_data[21] = block_data[16];
            cyclic_data[22] = block_data[17];
            cyclic_data[23] = block_data[18];
            cyclic_data[24] = block_data[19];
            cyclic_data[25] = block_data[20];
            cyclic_data[26] = block_data[21];
            cyclic_data[27] = block_data[22];
            cyclic_data[28] = block_data[23];
            cyclic_data[29] = block_data[24];
            cyclic_data[30] = block_data[25];
            cyclic_data[31] = block_data[26];

            cyclic_offset[0] = ((i+27)%TIME_STEP+TIME_STEP*27)/PAR;
            cyclic_offset[1] = ((i+28)%TIME_STEP+TIME_STEP*28)/PAR;
            cyclic_offset[2] = ((i+29)%TIME_STEP+TIME_STEP*29)/PAR;
            cyclic_offset[3] = ((i+30)%TIME_STEP+TIME_STEP*30)/PAR;
            cyclic_offset[4] = ((i+31)%TIME_STEP+TIME_STEP*31)/PAR;
            cyclic_offset[5] = i/PAR;
            cyclic_offset[6] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
            cyclic_offset[7] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR;
            cyclic_offset[8] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
            cyclic_offset[9] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
            cyclic_offset[10] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            cyclic_offset[11] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR;
            cyclic_offset[12] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR;
            cyclic_offset[13] = ((i+8)%TIME_STEP+TIME_STEP*8)/PAR;
            cyclic_offset[14] = ((i+9)%TIME_STEP+TIME_STEP*9)/PAR;
            cyclic_offset[15] = ((i+10)%TIME_STEP+TIME_STEP*10)/PAR;
            cyclic_offset[16] = ((i+11)%TIME_STEP+TIME_STEP*11)/PAR;
            cyclic_offset[17] = ((i+12)%TIME_STEP+TIME_STEP*12)/PAR;
            cyclic_offset[18] = ((i+13)%TIME_STEP+TIME_STEP*13)/PAR;
            cyclic_offset[19] = ((i+14)%TIME_STEP+TIME_STEP*14)/PAR;
            cyclic_offset[20] = ((i+15)%TIME_STEP+TIME_STEP*15)/PAR;
            cyclic_offset[21] = ((i+16)%TIME_STEP+TIME_STEP*16)/PAR;
            cyclic_offset[22] = ((i+17)%TIME_STEP+TIME_STEP*17)/PAR;
            cyclic_offset[23] = ((i+18)%TIME_STEP+TIME_STEP*18)/PAR;
            cyclic_offset[24] = ((i+19)%TIME_STEP+TIME_STEP*19)/PAR;
            cyclic_offset[25] = ((i+20)%TIME_STEP+TIME_STEP*20)/PAR;
            cyclic_offset[26] = ((i+21)%TIME_STEP+TIME_STEP*21)/PAR;
            cyclic_offset[27] = ((i+22)%TIME_STEP+TIME_STEP*22)/PAR;
            cyclic_offset[28] = ((i+23)%TIME_STEP+TIME_STEP*23)/PAR;
            cyclic_offset[29] = ((i+24)%TIME_STEP+TIME_STEP*24)/PAR;
            cyclic_offset[30] = ((i+25)%TIME_STEP+TIME_STEP*25)/PAR;
            cyclic_offset[31] = ((i+26)%TIME_STEP+TIME_STEP*26)/PAR;

                 
            

        }else if (i%PAR ==6 ){
            cyclic_data[0] = block_data[26];
            cyclic_data[1] = block_data[27];
            cyclic_data[2] = block_data[28];
            cyclic_data[3] = block_data[29];
            cyclic_data[4] = block_data[30];
            cyclic_data[5] = block_data[31];
            cyclic_data[6] = block_data[0];
            cyclic_data[7] = block_data[1];
            cyclic_data[8] = block_data[2];
            cyclic_data[9] = block_data[3];
            cyclic_data[10] = block_data[4];
            cyclic_data[11] = block_data[5];
            cyclic_data[12] = block_data[6];
            cyclic_data[13] = block_data[7];
            cyclic_data[14] = block_data[8];
            cyclic_data[15] = block_data[9];
            cyclic_data[16] = block_data[10];
            cyclic_data[17] = block_data[11];
            cyclic_data[18] = block_data[12];
            cyclic_data[19] = block_data[13];
            cyclic_data[20] = block_data[14];
            cyclic_data[21] = block_data[15];
            cyclic_data[22] = block_data[16];
            cyclic_data[23] = block_data[17];
            cyclic_data[24] = block_data[18];
            cyclic_data[25] = block_data[19];
            cyclic_data[26] = block_data[20];
            cyclic_data[27] = block_data[21];
            cyclic_data[28] = block_data[22];
            cyclic_data[29] = block_data[23];
            cyclic_data[30] = block_data[24];
            cyclic_data[31] = block_data[25];

            cyclic_offset[0] = ((i+26)%TIME_STEP+TIME_STEP*26)/PAR;
            cyclic_offset[1] = ((i+27)%TIME_STEP+TIME_STEP*27)/PAR;
            cyclic_offset[2] = ((i+28)%TIME_STEP+TIME_STEP*28)/PAR;
            cyclic_offset[3] = ((i+29)%TIME_STEP+TIME_STEP*29)/PAR;
            cyclic_offset[4] = ((i+30)%TIME_STEP+TIME_STEP*30)/PAR;
            cyclic_offset[5] = ((i+31)%TIME_STEP+TIME_STEP*31)/PAR;
            cyclic_offset[6] = i/PAR;
            cyclic_offset[7] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
            cyclic_offset[8] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR;
            cyclic_offset[9] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
            cyclic_offset[10] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
            cyclic_offset[11] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            cyclic_offset[12] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR;
            cyclic_offset[13] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR;
            cyclic_offset[14] = ((i+8)%TIME_STEP+TIME_STEP*8)/PAR;
            cyclic_offset[15] = ((i+9)%TIME_STEP+TIME_STEP*9)/PAR;
            cyclic_offset[16] = ((i+10)%TIME_STEP+TIME_STEP*10)/PAR;
            cyclic_offset[17] = ((i+11)%TIME_STEP+TIME_STEP*11)/PAR;
            cyclic_offset[18] = ((i+12)%TIME_STEP+TIME_STEP*12)/PAR;
            cyclic_offset[19] = ((i+13)%TIME_STEP+TIME_STEP*13)/PAR;
            cyclic_offset[20] = ((i+14)%TIME_STEP+TIME_STEP*14)/PAR;
            cyclic_offset[21] = ((i+15)%TIME_STEP+TIME_STEP*15)/PAR;
            cyclic_offset[22] = ((i+16)%TIME_STEP+TIME_STEP*16)/PAR;
            cyclic_offset[23] = ((i+17)%TIME_STEP+TIME_STEP*17)/PAR;
            cyclic_offset[24] = ((i+18)%TIME_STEP+TIME_STEP*18)/PAR;
            cyclic_offset[25] = ((i+19)%TIME_STEP+TIME_STEP*19)/PAR;
            cyclic_offset[26] = ((i+20)%TIME_STEP+TIME_STEP*20)/PAR;
            cyclic_offset[27] = ((i+21)%TIME_STEP+TIME_STEP*21)/PAR;
            cyclic_offset[28] = ((i+22)%TIME_STEP+TIME_STEP*22)/PAR;
            cyclic_offset[29] = ((i+23)%TIME_STEP+TIME_STEP*23)/PAR;
            cyclic_offset[30] = ((i+24)%TIME_STEP+TIME_STEP*24)/PAR;
            cyclic_offset[31] = ((i+25)%TIME_STEP+TIME_STEP*25)/PAR;
                     
            
        }else if (i%PAR == 7){

            cyclic_data[0] = block_data[25];
            cyclic_data[1] = block_data[26];
            cyclic_data[2] = block_data[27];
            cyclic_data[3] = block_data[28];
            cyclic_data[4] = block_data[29];
            cyclic_data[5] = block_data[30];
            cyclic_data[6] = block_data[31];
            cyclic_data[7] = block_data[0];
            cyclic_data[8] = block_data[1];
            cyclic_data[9] = block_data[2];
            cyclic_data[10] = block_data[3];
            cyclic_data[11] = block_data[4];
            cyclic_data[12] = block_data[5];
            cyclic_data[13] = block_data[6];
            cyclic_data[14] = block_data[7];
            cyclic_data[15] = block_data[8];
            cyclic_data[16] = block_data[9];
            cyclic_data[17] = block_data[10];
            cyclic_data[18] = block_data[11];
            cyclic_data[19] = block_data[12];
            cyclic_data[20] = block_data[13];
            cyclic_data[21] = block_data[14];
            cyclic_data[22] = block_data[15];
            cyclic_data[23] = block_data[16];
            cyclic_data[24] = block_data[17];
            cyclic_data[25] = block_data[18];
            cyclic_data[26] = block_data[19];
            cyclic_data[27] = block_data[20];
            cyclic_data[28] = block_data[21];
            cyclic_data[29] = block_data[22];
            cyclic_data[30] = block_data[23];
            cyclic_data[31] = block_data[24];

            cyclic_offset[0] = ((i+25)%TIME_STEP+TIME_STEP*25)/PAR;
            cyclic_offset[1] = ((i+26)%TIME_STEP+TIME_STEP*26)/PAR;
            cyclic_offset[2] = ((i+27)%TIME_STEP+TIME_STEP*27)/PAR;
            cyclic_offset[3] = ((i+28)%TIME_STEP+TIME_STEP*28)/PAR;
            cyclic_offset[4] = ((i+29)%TIME_STEP+TIME_STEP*29)/PAR;
            cyclic_offset[5] = ((i+30)%TIME_STEP+TIME_STEP*30)/PAR;
            cyclic_offset[6] = ((i+31)%TIME_STEP+TIME_STEP*31)/PAR;
            cyclic_offset[7] = i/PAR;
            cyclic_offset[8] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
            cyclic_offset[9] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR;
            cyclic_offset[10] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
            cyclic_offset[11] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
            cyclic_offset[12] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            cyclic_offset[13] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR;
            cyclic_offset[14] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR;
            cyclic_offset[15] = ((i+8)%TIME_STEP+TIME_STEP*8)/PAR;
            cyclic_offset[16] = ((i+9)%TIME_STEP+TIME_STEP*9)/PAR;
            cyclic_offset[17] = ((i+10)%TIME_STEP+TIME_STEP*10)/PAR;
            cyclic_offset[18] = ((i+11)%TIME_STEP+TIME_STEP*11)/PAR;
            cyclic_offset[19] = ((i+12)%TIME_STEP+TIME_STEP*12)/PAR;
            cyclic_offset[20] = ((i+13)%TIME_STEP+TIME_STEP*13)/PAR;
            cyclic_offset[21] = ((i+14)%TIME_STEP+TIME_STEP*14)/PAR;
            cyclic_offset[22] = ((i+15)%TIME_STEP+TIME_STEP*15)/PAR;
            cyclic_offset[23] = ((i+16)%TIME_STEP+TIME_STEP*16)/PAR;
            cyclic_offset[24] = ((i+17)%TIME_STEP+TIME_STEP*17)/PAR;
            cyclic_offset[25] = ((i+18)%TIME_STEP+TIME_STEP*18)/PAR;
            cyclic_offset[26] = ((i+19)%TIME_STEP+TIME_STEP*19)/PAR;
            cyclic_offset[27] = ((i+20)%TIME_STEP+TIME_STEP*20)/PAR;
            cyclic_offset[28] = ((i+21)%TIME_STEP+TIME_STEP*21)/PAR;
            cyclic_offset[29] = ((i+22)%TIME_STEP+TIME_STEP*22)/PAR;
            cyclic_offset[30] = ((i+23)%TIME_STEP+TIME_STEP*23)/PAR;
            cyclic_offset[31] = ((i+24)%TIME_STEP+TIME_STEP*24)/PAR;
                             
         
        }else if (i%PAR == 8){

            cyclic_data[0] = block_data[24];
            cyclic_data[1] = block_data[25];
            cyclic_data[2] = block_data[26];
            cyclic_data[3] = block_data[27];
            cyclic_data[4] = block_data[28];
            cyclic_data[5] = block_data[29];
            cyclic_data[6] = block_data[30];
            cyclic_data[7] = block_data[31];
            cyclic_data[8] = block_data[0];
            cyclic_data[9] = block_data[1];
            cyclic_data[10] = block_data[2];
            cyclic_data[11] = block_data[3];
            cyclic_data[12] = block_data[4];
            cyclic_data[13] = block_data[5];
            cyclic_data[14] = block_data[6];
            cyclic_data[15] = block_data[7];
            cyclic_data[16] = block_data[8];
            cyclic_data[17] = block_data[9];
            cyclic_data[18] = block_data[10];
            cyclic_data[19] = block_data[11];
            cyclic_data[20] = block_data[12];
            cyclic_data[21] = block_data[13];
            cyclic_data[22] = block_data[14];
            cyclic_data[23] = block_data[15];
            cyclic_data[24] = block_data[16];
            cyclic_data[25] = block_data[17];
            cyclic_data[26] = block_data[18];
            cyclic_data[27] = block_data[19];
            cyclic_data[28] = block_data[20];
            cyclic_data[29] = block_data[21];
            cyclic_data[30] = block_data[22];
            cyclic_data[31] = block_data[23];

            cyclic_offset[0] = ((i+24)%TIME_STEP+TIME_STEP*24)/PAR;
            cyclic_offset[1] = ((i+25)%TIME_STEP+TIME_STEP*25)/PAR;
            cyclic_offset[2] = ((i+26)%TIME_STEP+TIME_STEP*26)/PAR;
            cyclic_offset[3] = ((i+27)%TIME_STEP+TIME_STEP*27)/PAR;
            cyclic_offset[4] = ((i+28)%TIME_STEP+TIME_STEP*28)/PAR;
            cyclic_offset[5] = ((i+29)%TIME_STEP+TIME_STEP*29)/PAR;
            cyclic_offset[6] = ((i+30)%TIME_STEP+TIME_STEP*30)/PAR;
            cyclic_offset[7] = ((i+31)%TIME_STEP+TIME_STEP*31)/PAR;
            cyclic_offset[8] = i/PAR;
            cyclic_offset[9] = ((i+1)%TIME_STEP+TIME_STEP)/PAR;
            cyclic_offset[10] = ((i+2)%TIME_STEP+TIME_STEP*2)/PAR;
            cyclic_offset[11] = ((i+3)%TIME_STEP+TIME_STEP*3)/PAR;
            cyclic_offset[12] = ((i+4)%TIME_STEP+TIME_STEP*4)/PAR;
            cyclic_offset[13] = ((i+5)%TIME_STEP+TIME_STEP*5)/PAR;
            cyclic_offset[14] = ((i+6)%TIME_STEP+TIME_STEP*6)/PAR;
            cyclic_offset[15] = ((i+7)%TIME_STEP+TIME_STEP*7)/PAR;
            cyclic_offset[16] = ((i+8)%TIME_STEP+TIME_STEP*8)/PAR;
            cyclic_offset[17] = ((i+9)%TIME_STEP+TIME_STEP*9)/PAR;
            cyclic_offset[18] = ((i+10)%TIME_STEP+TIME_STEP*10)/PAR;
            cyclic_offset[19] = ((i+11)%TIME_STEP+TIME_STEP*11)/PAR;
            cyclic_offset[20] = ((i+12)%TIME_STEP+TIME_STEP*12)/PAR;
            cyclic_offset[21] = ((i+13)%TIME_STEP+TIME_STEP*13)/PAR;
            cyclic_offset[22] = ((i+14)%TIME_STEP+TIME_STEP*14)/PAR;
            cyclic_offset[23] = ((i+15)%TIME_STEP+TIME_STEP*15)/PAR;
            cyclic_offset[24] = ((i+16)%TIME_STEP+TIME_STEP*16)/PAR;
            cyclic_offset[25] = ((i+17)%TIME_STEP+TIME_STEP*17)/PAR;
            cyclic_offset[26] = ((i+18)%TIME_STEP+TIME_STEP*18)/PAR;
            cyclic_offset[27] = ((i+19)%TIME_STEP+TIME_STEP*19)/PAR;
            cyclic_offset[28] = ((i+20)%TIME_STEP+TIME_STEP*20)/PAR;
            cyclic_offset[29] = ((i+21)%TIME_STEP+TIME_STEP*21)/PAR;
            cyclic_offset[30] = ((i+22)%TIME_STEP+TIME_STEP*22)/PAR;
            cyclic_offset[31] = ((i+23)%TIME_STEP+TIME_STEP*23)/PAR;
       
        }else if (i%PAR == 9){

            cyclic_data[0] = block_data[23];
            cyclic_data[1] = block_data[24];
            cyclic_data[2] = block_data[25];
            cyclic_data[3] = block_data[26];
            cyclic_data[4] = block_data[27];
            cyclic_data[5] = block_data[28];
            cyclic_data[6] = block_data[29];
            cyclic_data[7] = block_data[30];
            cyclic_data[8] = block_data[31];
            cyclic_data[9] = block_data[0];
            cyclic_data[10] = block_data[1];
            cyclic_data[11] = block_data[2];
            cyclic_data[12] = block_data[3];
            cyclic_data[13] = block_data[4];
            cyclic_data[14] = block_data[5];
            cyclic_data[15] = block_data[6];
            cyclic_data[16] = block_data[7];
            cyclic_data[17] = block_data[8];
            cyclic_data[18] = block_data[9];
            cyclic_data[19] = block_data[10];
            cyclic_data[20] = block_data[11];
            cyclic_data[21] = block_data[12];
            cyclic_data[22] = block_data[13];
            cyclic_data[23] = block_data[14];
            cyclic_data[24] = block_data[15];
            cyclic_data[25] = block_data[16];
            cyclic_data[26] = block_data[17];
            cyclic_data[27] = block_data[18];
            cyclic_data[28] = block_data[19];
            cyclic_data[29] = block_data[20];
            cyclic_data[30] = block_data[21];
            cyclic_data[31] = block_data[22];

            cyclic_offset[0] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[1] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[2] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[3] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[4] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[5] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[6] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[7] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[8] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[9] = i / PAR;
            cyclic_offset[10] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[11] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[12] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[13] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[14] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[15] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[16] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[17] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[18] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[19] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[20] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[21] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[22] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[23] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[24] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[25] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[26] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[27] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[28] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[29] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[30] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[31] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
                                     

        }else if (i%PAR == 10){

            cyclic_data[0] = block_data[22];
            cyclic_data[1] = block_data[23];
            cyclic_data[2] = block_data[24];
            cyclic_data[3] = block_data[25];
            cyclic_data[4] = block_data[26];
            cyclic_data[5] = block_data[27];
            cyclic_data[6] = block_data[28];
            cyclic_data[7] = block_data[29];
            cyclic_data[8] = block_data[30];
            cyclic_data[9] = block_data[31];
            cyclic_data[10] = block_data[0];
            cyclic_data[11] = block_data[1];
            cyclic_data[12] = block_data[2];
            cyclic_data[13] = block_data[3];
            cyclic_data[14] = block_data[4];
            cyclic_data[15] = block_data[5];
            cyclic_data[16] = block_data[6];
            cyclic_data[17] = block_data[7];
            cyclic_data[18] = block_data[8];
            cyclic_data[19] = block_data[9];
            cyclic_data[20] = block_data[10];
            cyclic_data[21] = block_data[11];
            cyclic_data[22] = block_data[12];
            cyclic_data[23] = block_data[13];
            cyclic_data[24] = block_data[14];
            cyclic_data[25] = block_data[15];
            cyclic_data[26] = block_data[16];
            cyclic_data[27] = block_data[17];
            cyclic_data[28] = block_data[18];
            cyclic_data[29] = block_data[19];
            cyclic_data[30] = block_data[20];
            cyclic_data[31] = block_data[21];

            cyclic_offset[0] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[1] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[2] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[3] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[4] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[5] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[6] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[7] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[8] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[9] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[10] = i / PAR;
            cyclic_offset[11] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[12] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[13] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[14] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[15] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[16] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[17] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[18] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[19] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[20] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[21] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[22] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[23] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[24] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[25] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[26] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[27] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[28] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[29] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[30] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[31] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
                                         
         
        }else if (i%PAR == 11){
            cyclic_data[0] = block_data[21];
            cyclic_data[1] = block_data[22];
            cyclic_data[2] = block_data[23];
            cyclic_data[3] = block_data[24];
            cyclic_data[4] = block_data[25];
            cyclic_data[5] = block_data[26];
            cyclic_data[6] = block_data[27];
            cyclic_data[7] = block_data[28];
            cyclic_data[8] = block_data[29];
            cyclic_data[9] = block_data[30];
            cyclic_data[10] = block_data[31];
            cyclic_data[11] = block_data[0];
            cyclic_data[12] = block_data[1];
            cyclic_data[13] = block_data[2];
            cyclic_data[14] = block_data[3];
            cyclic_data[15] = block_data[4];
            cyclic_data[16] = block_data[5];
            cyclic_data[17] = block_data[6];
            cyclic_data[18] = block_data[7];
            cyclic_data[19] = block_data[8];
            cyclic_data[20] = block_data[9];
            cyclic_data[21] = block_data[10];
            cyclic_data[22] = block_data[11];
            cyclic_data[23] = block_data[12];
            cyclic_data[24] = block_data[13];
            cyclic_data[25] = block_data[14];
            cyclic_data[26] = block_data[15];
            cyclic_data[27] = block_data[16];
            cyclic_data[28] = block_data[17];
            cyclic_data[29] = block_data[18];
            cyclic_data[30] = block_data[19];
            cyclic_data[31] = block_data[20];

            cyclic_offset[0] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[1] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[2] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[3] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[4] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[5] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[6] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[7] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[8] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[9] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[10] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[11] = i / PAR;
            cyclic_offset[12] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[13] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[14] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[15] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[16] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[17] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[18] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[19] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[20] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[21] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[22] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[23] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[24] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[25] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[26] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[27] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[28] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[29] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[30] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[31] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
                  
        }else if (i%PAR == 12){
            cyclic_data[0] = block_data[20];
            cyclic_data[1] = block_data[21];
            cyclic_data[2] = block_data[22];
            cyclic_data[3] = block_data[23];
            cyclic_data[4] = block_data[24];
            cyclic_data[5] = block_data[25];
            cyclic_data[6] = block_data[26];
            cyclic_data[7] = block_data[27];
            cyclic_data[8] = block_data[28];
            cyclic_data[9] = block_data[29];
            cyclic_data[10] = block_data[30];
            cyclic_data[11] = block_data[31];
            cyclic_data[12] = block_data[0];
            cyclic_data[13] = block_data[1];
            cyclic_data[14] = block_data[2];
            cyclic_data[15] = block_data[3];
            cyclic_data[16] = block_data[4];
            cyclic_data[17] = block_data[5];
            cyclic_data[18] = block_data[6];
            cyclic_data[19] = block_data[7];
            cyclic_data[20] = block_data[8];
            cyclic_data[21] = block_data[9];
            cyclic_data[22] = block_data[10];
            cyclic_data[23] = block_data[11];
            cyclic_data[24] = block_data[12];
            cyclic_data[25] = block_data[13];
            cyclic_data[26] = block_data[14];
            cyclic_data[27] = block_data[15];
            cyclic_data[28] = block_data[16];
            cyclic_data[29] = block_data[17];
            cyclic_data[30] = block_data[18];
            cyclic_data[31] = block_data[19];

            cyclic_offset[0] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[1] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[2] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[3] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[4] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[5] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[6] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[7] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[8] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[9] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[10] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[11] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[12] = i / PAR;
            cyclic_offset[13] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[14] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[15] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[16] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[17] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[18] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[19] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[20] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[21] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[22] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[23] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[24] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[25] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[26] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[27] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[28] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[29] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[30] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[31] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
                                                 

        }else if (i%PAR == 13){
            cyclic_data[0] = block_data[19];
            cyclic_data[1] = block_data[20];
            cyclic_data[2] = block_data[21];
            cyclic_data[3] = block_data[22];
            cyclic_data[4] = block_data[23];
            cyclic_data[5] = block_data[24];
            cyclic_data[6] = block_data[25];
            cyclic_data[7] = block_data[26];
            cyclic_data[8] = block_data[27];
            cyclic_data[9] = block_data[28];
            cyclic_data[10] = block_data[29];
            cyclic_data[11] = block_data[30];
            cyclic_data[12] = block_data[31];
            cyclic_data[13] = block_data[0];
            cyclic_data[14] = block_data[1];
            cyclic_data[15] = block_data[2];
            cyclic_data[16] = block_data[3];
            cyclic_data[17] = block_data[4];
            cyclic_data[18] = block_data[5];
            cyclic_data[19] = block_data[6];
            cyclic_data[20] = block_data[7];
            cyclic_data[21] = block_data[8];
            cyclic_data[22] = block_data[9];
            cyclic_data[23] = block_data[10];
            cyclic_data[24] = block_data[11];
            cyclic_data[25] = block_data[12];
            cyclic_data[26] = block_data[13];
            cyclic_data[27] = block_data[14];
            cyclic_data[28] = block_data[15];
            cyclic_data[29] = block_data[16];
            cyclic_data[30] = block_data[17];
            cyclic_data[31] = block_data[18];

            cyclic_offset[0] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[1] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[2] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[3] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[4] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[5] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[6] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[7] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[8] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[9] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[10] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[11] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[12] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[13] = i / PAR;
            cyclic_offset[14] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[15] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[16] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[17] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[18] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[19] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[20] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[21] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[22] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[23] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[24] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[25] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[26] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[27] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[28] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[29] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[30] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[31] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
          
        }else if (i%PAR == 14){
            cyclic_data[0] = block_data[18];
            cyclic_data[1] = block_data[19];
            cyclic_data[2] = block_data[20];
            cyclic_data[3] = block_data[21];
            cyclic_data[4] = block_data[22];
            cyclic_data[5] = block_data[23];
            cyclic_data[6] = block_data[24];
            cyclic_data[7] = block_data[25];
            cyclic_data[8] = block_data[26];
            cyclic_data[9] = block_data[27];
            cyclic_data[10] = block_data[28];
            cyclic_data[11] = block_data[29];
            cyclic_data[12] = block_data[30];
            cyclic_data[13] = block_data[31];
            cyclic_data[14] = block_data[0];
            cyclic_data[15] = block_data[1];
            cyclic_data[16] = block_data[2];
            cyclic_data[17] = block_data[3];
            cyclic_data[18] = block_data[4];
            cyclic_data[19] = block_data[5];
            cyclic_data[20] = block_data[6];
            cyclic_data[21] = block_data[7];
            cyclic_data[22] = block_data[8];
            cyclic_data[23] = block_data[9];
            cyclic_data[24] = block_data[10];
            cyclic_data[25] = block_data[11];
            cyclic_data[26] = block_data[12];
            cyclic_data[27] = block_data[13];
            cyclic_data[28] = block_data[14];
            cyclic_data[29] = block_data[15];
            cyclic_data[30] = block_data[16];
            cyclic_data[31] = block_data[17];

            cyclic_offset[0] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[1] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[2] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[3] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[4] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[5] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[6] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[7] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[8] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[9] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[10] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[11] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[12] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[13] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[14] = i / PAR;
            cyclic_offset[15] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[16] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[17] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[18] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[19] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[20] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[21] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[22] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[23] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[24] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[25] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[26] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[27] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[28] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[29] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[30] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[31] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
        

        }else if (i%PAR == 15){
            cyclic_data[0] = block_data[17];
            cyclic_data[1] = block_data[18];
            cyclic_data[2] = block_data[19];
            cyclic_data[3] = block_data[20];
            cyclic_data[4] = block_data[21];
            cyclic_data[5] = block_data[22];
            cyclic_data[6] = block_data[23];
            cyclic_data[7] = block_data[24];
            cyclic_data[8] = block_data[25];
            cyclic_data[9] = block_data[26];
            cyclic_data[10] = block_data[27];
            cyclic_data[11] = block_data[28];
            cyclic_data[12] = block_data[29];
            cyclic_data[13] = block_data[30];
            cyclic_data[14] = block_data[31];
            cyclic_data[15] = block_data[0];
            cyclic_data[16] = block_data[1];
            cyclic_data[17] = block_data[2];
            cyclic_data[18] = block_data[3];
            cyclic_data[19] = block_data[4];
            cyclic_data[20] = block_data[5];
            cyclic_data[21] = block_data[6];
            cyclic_data[22] = block_data[7];
            cyclic_data[23] = block_data[8];
            cyclic_data[24] = block_data[9];
            cyclic_data[25] = block_data[10];
            cyclic_data[26] = block_data[11];
            cyclic_data[27] = block_data[12];
            cyclic_data[28] = block_data[13];
            cyclic_data[29] = block_data[14];
            cyclic_data[30] = block_data[15];
            cyclic_data[31] = block_data[16];

            cyclic_offset[0] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[1] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[2] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[3] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[4] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[5] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[6] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[7] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[8] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[9] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[10] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[11] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[12] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[13] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[14] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[15] = i / PAR;
            cyclic_offset[16] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[17] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[18] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[19] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[20] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[21] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[22] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[23] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[24] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[25] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[26] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[27] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[28] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[29] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[30] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[31] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
                        
        }else if (i%PAR == 16){

            cyclic_data[0] = block_data[16];
            cyclic_data[1] = block_data[17];
            cyclic_data[2] = block_data[18];
            cyclic_data[3] = block_data[19];
            cyclic_data[4] = block_data[20];
            cyclic_data[5] = block_data[21];
            cyclic_data[6] = block_data[22];
            cyclic_data[7] = block_data[23];
            cyclic_data[8] = block_data[24];
            cyclic_data[9] = block_data[25];
            cyclic_data[10] = block_data[26];
            cyclic_data[11] = block_data[27];
            cyclic_data[12] = block_data[28];
            cyclic_data[13] = block_data[29];
            cyclic_data[14] = block_data[30];
            cyclic_data[15] = block_data[31];
            cyclic_data[16] = block_data[0];
            cyclic_data[17] = block_data[1];
            cyclic_data[18] = block_data[2];
            cyclic_data[19] = block_data[3];
            cyclic_data[20] = block_data[4];
            cyclic_data[21] = block_data[5];
            cyclic_data[22] = block_data[6];
            cyclic_data[23] = block_data[7];
            cyclic_data[24] = block_data[8];
            cyclic_data[25] = block_data[9];
            cyclic_data[26] = block_data[10];
            cyclic_data[27] = block_data[11];
            cyclic_data[28] = block_data[12];
            cyclic_data[29] = block_data[13];
            cyclic_data[30] = block_data[14];
            cyclic_data[31] = block_data[15];

            cyclic_offset[0] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[1] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[2] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[3] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[4] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[5] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[6] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[7] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[8] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[9] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[10] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[11] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[12] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[13] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[14] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[15] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[16] = i / PAR;
            cyclic_offset[17] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[18] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[19] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[20] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[21] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[22] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[23] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[24] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[25] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[26] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[27] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[28] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[29] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[30] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[31] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;


        }else if (i%PAR == 17){
            cyclic_data[0] = block_data[15];
            cyclic_data[1] = block_data[16];
            cyclic_data[2] = block_data[17];
            cyclic_data[3] = block_data[18];
            cyclic_data[4] = block_data[19];
            cyclic_data[5] = block_data[20];
            cyclic_data[6] = block_data[21];
            cyclic_data[7] = block_data[22];
            cyclic_data[8] = block_data[23];
            cyclic_data[9] = block_data[24];
            cyclic_data[10] = block_data[25];
            cyclic_data[11] = block_data[26];
            cyclic_data[12] = block_data[27];
            cyclic_data[13] = block_data[28];
            cyclic_data[14] = block_data[29];
            cyclic_data[15] = block_data[30];
            cyclic_data[16] = block_data[31];
            cyclic_data[17] = block_data[0];
            cyclic_data[18] = block_data[1];
            cyclic_data[19] = block_data[2];
            cyclic_data[20] = block_data[3];
            cyclic_data[21] = block_data[4];
            cyclic_data[22] = block_data[5];
            cyclic_data[23] = block_data[6];
            cyclic_data[24] = block_data[7];
            cyclic_data[25] = block_data[8];
            cyclic_data[26] = block_data[9];
            cyclic_data[27] = block_data[10];
            cyclic_data[28] = block_data[11];
            cyclic_data[29] = block_data[12];
            cyclic_data[30] = block_data[13];
            cyclic_data[31] = block_data[14];

            cyclic_offset[0] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[1] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[2] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[3] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[4] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[5] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[6] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[7] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[8] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[9] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[10] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[11] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[12] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[13] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[14] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[15] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[16] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[17] = i / PAR;
            cyclic_offset[18] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[19] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[20] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[21] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[22] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[23] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[24] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[25] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[26] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[27] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[28] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[29] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[30] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[31] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;


        }else if (i%PAR == 18){

            cyclic_data[0] = block_data[14];
            cyclic_data[1] = block_data[15];
            cyclic_data[2] = block_data[16];
            cyclic_data[3] = block_data[17];
            cyclic_data[4] = block_data[18];
            cyclic_data[5] = block_data[19];
            cyclic_data[6] = block_data[20];
            cyclic_data[7] = block_data[21];
            cyclic_data[8] = block_data[22];
            cyclic_data[9] = block_data[23];
            cyclic_data[10] = block_data[24];
            cyclic_data[11] = block_data[25];
            cyclic_data[12] = block_data[26];
            cyclic_data[13] = block_data[27];
            cyclic_data[14] = block_data[28];
            cyclic_data[15] = block_data[29];
            cyclic_data[16] = block_data[30];
            cyclic_data[17] = block_data[31];
            cyclic_data[18] = block_data[0];
            cyclic_data[19] = block_data[1];
            cyclic_data[20] = block_data[2];
            cyclic_data[21] = block_data[3];
            cyclic_data[22] = block_data[4];
            cyclic_data[23] = block_data[5];
            cyclic_data[24] = block_data[6];
            cyclic_data[25] = block_data[7];
            cyclic_data[26] = block_data[8];
            cyclic_data[27] = block_data[9];
            cyclic_data[28] = block_data[10];
            cyclic_data[29] = block_data[11];
            cyclic_data[30] = block_data[12];
            cyclic_data[31] = block_data[13];

            cyclic_offset[0] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[1] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[2] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[3] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[4] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[5] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[6] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[7] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[8] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[9] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[10] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[11] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[12] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[13] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[14] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[15] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[16] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[17] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[18] = i / PAR;
            cyclic_offset[19] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[20] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[21] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[22] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[23] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[24] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[25] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[26] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[27] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[28] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[29] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[30] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[31] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;

        }else if (i%PAR == 19){
            cyclic_data[0] = block_data[13];
            cyclic_data[1] = block_data[14];
            cyclic_data[2] = block_data[15];
            cyclic_data[3] = block_data[16];
            cyclic_data[4] = block_data[17];
            cyclic_data[5] = block_data[18];
            cyclic_data[6] = block_data[19];
            cyclic_data[7] = block_data[20];
            cyclic_data[8] = block_data[21];
            cyclic_data[9] = block_data[22];
            cyclic_data[10] = block_data[23];
            cyclic_data[11] = block_data[24];
            cyclic_data[12] = block_data[25];
            cyclic_data[13] = block_data[26];
            cyclic_data[14] = block_data[27];
            cyclic_data[15] = block_data[28];
            cyclic_data[16] = block_data[29];
            cyclic_data[17] = block_data[30];
            cyclic_data[18] = block_data[31];
            cyclic_data[19] = block_data[0];
            cyclic_data[20] = block_data[1];
            cyclic_data[21] = block_data[2];
            cyclic_data[22] = block_data[3];
            cyclic_data[23] = block_data[4];
            cyclic_data[24] = block_data[5];
            cyclic_data[25] = block_data[6];
            cyclic_data[26] = block_data[7];
            cyclic_data[27] = block_data[8];
            cyclic_data[28] = block_data[9];
            cyclic_data[29] = block_data[10];
            cyclic_data[30] = block_data[11];
            cyclic_data[31] = block_data[12];

            cyclic_offset[0] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[1] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[2] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[3] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[4] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[5] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[6] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[7] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[8] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[9] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[10] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[11] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[12] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[13] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[14] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[15] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[16] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[17] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[18] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[19] = i / PAR;
            cyclic_offset[20] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[21] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[22] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[23] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[24] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[25] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[26] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[27] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[28] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[29] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[30] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[31] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;

        }else if (i%PAR == 20){
            cyclic_data[0] = block_data[12];
            cyclic_data[1] = block_data[13];
            cyclic_data[2] = block_data[14];
            cyclic_data[3] = block_data[15];
            cyclic_data[4] = block_data[16];
            cyclic_data[5] = block_data[17];
            cyclic_data[6] = block_data[18];
            cyclic_data[7] = block_data[19];
            cyclic_data[8] = block_data[20];
            cyclic_data[9] = block_data[21];
            cyclic_data[10] = block_data[22];
            cyclic_data[11] = block_data[23];
            cyclic_data[12] = block_data[24];
            cyclic_data[13] = block_data[25];
            cyclic_data[14] = block_data[26];
            cyclic_data[15] = block_data[27];
            cyclic_data[16] = block_data[28];
            cyclic_data[17] = block_data[29];
            cyclic_data[18] = block_data[30];
            cyclic_data[19] = block_data[31];
            cyclic_data[20] = block_data[0];
            cyclic_data[21] = block_data[1];
            cyclic_data[22] = block_data[2];
            cyclic_data[23] = block_data[3];
            cyclic_data[24] = block_data[4];
            cyclic_data[25] = block_data[5];
            cyclic_data[26] = block_data[6];
            cyclic_data[27] = block_data[7];
            cyclic_data[28] = block_data[8];
            cyclic_data[29] = block_data[9];
            cyclic_data[30] = block_data[10];
            cyclic_data[31] = block_data[11];

            cyclic_offset[0] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[1] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[2] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[3] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[4] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[5] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[6] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[7] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[8] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[9] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[10] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[11] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[12] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[13] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[14] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[15] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[16] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[17] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[18] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[19] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[20] = i / PAR;
            cyclic_offset[21] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[22] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[23] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[24] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[25] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[26] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[27] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[28] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[29] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[30] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[31] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            

        }else if (i%PAR == 21){
            cyclic_data[0] = block_data[11];
            cyclic_data[1] = block_data[12];
            cyclic_data[2] = block_data[13];
            cyclic_data[3] = block_data[14];
            cyclic_data[4] = block_data[15];
            cyclic_data[5] = block_data[16];
            cyclic_data[6] = block_data[17];
            cyclic_data[7] = block_data[18];
            cyclic_data[8] = block_data[19];
            cyclic_data[9] = block_data[20];
            cyclic_data[10] = block_data[21];
            cyclic_data[11] = block_data[22];
            cyclic_data[12] = block_data[23];
            cyclic_data[13] = block_data[24];
            cyclic_data[14] = block_data[25];
            cyclic_data[15] = block_data[26];
            cyclic_data[16] = block_data[27];
            cyclic_data[17] = block_data[28];
            cyclic_data[18] = block_data[29];
            cyclic_data[19] = block_data[30];
            cyclic_data[20] = block_data[31];
            cyclic_data[21] = block_data[0];
            cyclic_data[22] = block_data[1];
            cyclic_data[23] = block_data[2];
            cyclic_data[24] = block_data[3];
            cyclic_data[25] = block_data[4];
            cyclic_data[26] = block_data[5];
            cyclic_data[27] = block_data[6];
            cyclic_data[28] = block_data[7];
            cyclic_data[29] = block_data[8];
            cyclic_data[30] = block_data[9];
            cyclic_data[31] = block_data[10];

            cyclic_offset[0] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[1] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[2] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[3] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[4] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[5] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[6] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[7] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[8] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[9] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[10] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[11] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[12] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[13] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[14] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[15] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[16] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[17] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[18] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[19] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[20] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[21] = i / PAR;
            cyclic_offset[22] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[23] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[24] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[25] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[26] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[27] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[28] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[29] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[30] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[31] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;

        }else if (i%PAR == 22){
            cyclic_data[0] = block_data[10];
            cyclic_data[1] = block_data[11];
            cyclic_data[2] = block_data[12];
            cyclic_data[3] = block_data[13];
            cyclic_data[4] = block_data[14];
            cyclic_data[5] = block_data[15];
            cyclic_data[6] = block_data[16];
            cyclic_data[7] = block_data[17];
            cyclic_data[8] = block_data[18];
            cyclic_data[9] = block_data[19];
            cyclic_data[10] = block_data[20];
            cyclic_data[11] = block_data[21];
            cyclic_data[12] = block_data[22];
            cyclic_data[13] = block_data[23];
            cyclic_data[14] = block_data[24];
            cyclic_data[15] = block_data[25];
            cyclic_data[16] = block_data[26];
            cyclic_data[17] = block_data[27];
            cyclic_data[18] = block_data[28];
            cyclic_data[19] = block_data[29];
            cyclic_data[20] = block_data[30];
            cyclic_data[21] = block_data[31];
            cyclic_data[22] = block_data[0];
            cyclic_data[23] = block_data[1];
            cyclic_data[24] = block_data[2];
            cyclic_data[25] = block_data[3];
            cyclic_data[26] = block_data[4];
            cyclic_data[27] = block_data[5];
            cyclic_data[28] = block_data[6];
            cyclic_data[29] = block_data[7];
            cyclic_data[30] = block_data[8];
            cyclic_data[31] = block_data[9];

            cyclic_offset[0] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[1] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[2] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[3] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[4] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[5] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[6] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[7] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[8] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[9] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[10] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[11] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[12] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[13] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[14] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[15] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[16] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[17] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[18] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[19] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[20] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[21] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[22] = i / PAR;
            cyclic_offset[23] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[24] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[25] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[26] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[27] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[28] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[29] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[30] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[31] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;

        }else if (i%PAR == 23){
            cyclic_data[0] = block_data[9];
            cyclic_data[1] = block_data[10];
            cyclic_data[2] = block_data[11];
            cyclic_data[3] = block_data[12];
            cyclic_data[4] = block_data[13];
            cyclic_data[5] = block_data[14];
            cyclic_data[6] = block_data[15];
            cyclic_data[7] = block_data[16];
            cyclic_data[8] = block_data[17];
            cyclic_data[9] = block_data[18];
            cyclic_data[10] = block_data[19];
            cyclic_data[11] = block_data[20];
            cyclic_data[12] = block_data[21];
            cyclic_data[13] = block_data[22];
            cyclic_data[14] = block_data[23];
            cyclic_data[15] = block_data[24];
            cyclic_data[16] = block_data[25];
            cyclic_data[17] = block_data[26];
            cyclic_data[18] = block_data[27];
            cyclic_data[19] = block_data[28];
            cyclic_data[20] = block_data[29];
            cyclic_data[21] = block_data[30];
            cyclic_data[22] = block_data[31];
            cyclic_data[23] = block_data[0];
            cyclic_data[24] = block_data[1];
            cyclic_data[25] = block_data[2];
            cyclic_data[26] = block_data[3];
            cyclic_data[27] = block_data[4];
            cyclic_data[28] = block_data[5];
            cyclic_data[29] = block_data[6];
            cyclic_data[30] = block_data[7];
            cyclic_data[31] = block_data[8];

            cyclic_offset[0] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[1] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[2] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[3] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[4] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[5] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[6] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[7] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[8] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[9] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[10] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[11] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[12] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[13] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[14] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[15] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[16] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[17] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[18] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[19] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[20] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[21] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[22] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[23] = i / PAR;
            cyclic_offset[24] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[25] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[26] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[27] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[28] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[29] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[30] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[31] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;

        }else if (i%PAR == 24){
            cyclic_data[0] = block_data[8];
            cyclic_data[1] = block_data[9];
            cyclic_data[2] = block_data[10];
            cyclic_data[3] = block_data[11];
            cyclic_data[4] = block_data[12];
            cyclic_data[5] = block_data[13];
            cyclic_data[6] = block_data[14];
            cyclic_data[7] = block_data[15];
            cyclic_data[8] = block_data[16];
            cyclic_data[9] = block_data[17];
            cyclic_data[10] = block_data[18];
            cyclic_data[11] = block_data[19];
            cyclic_data[12] = block_data[20];
            cyclic_data[13] = block_data[21];
            cyclic_data[14] = block_data[22];
            cyclic_data[15] = block_data[23];
            cyclic_data[16] = block_data[24];
            cyclic_data[17] = block_data[25];
            cyclic_data[18] = block_data[26];
            cyclic_data[19] = block_data[27];
            cyclic_data[20] = block_data[28];
            cyclic_data[21] = block_data[29];
            cyclic_data[22] = block_data[30];
            cyclic_data[23] = block_data[31];
            cyclic_data[24] = block_data[0];
            cyclic_data[25] = block_data[1];
            cyclic_data[26] = block_data[2];
            cyclic_data[27] = block_data[3];
            cyclic_data[28] = block_data[4];
            cyclic_data[29] = block_data[5];
            cyclic_data[30] = block_data[6];
            cyclic_data[31] = block_data[7];

            cyclic_offset[0] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[1] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[2] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[3] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[4] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[5] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[6] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[7] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[8] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[9] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[10] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[11] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[12] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[13] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[14] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[15] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[16] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[17] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[18] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[19] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[20] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[21] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[22] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[23] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[24] = i / PAR;
            cyclic_offset[25] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[26] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[27] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[28] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[29] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[30] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[31] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;

        }else if (i%PAR == 25){
            cyclic_data[0] = block_data[7];
            cyclic_data[1] = block_data[8];
            cyclic_data[2] = block_data[9];
            cyclic_data[3] = block_data[10];
            cyclic_data[4] = block_data[11];
            cyclic_data[5] = block_data[12];
            cyclic_data[6] = block_data[13];
            cyclic_data[7] = block_data[14];
            cyclic_data[8] = block_data[15];
            cyclic_data[9] = block_data[16];
            cyclic_data[10] = block_data[17];
            cyclic_data[11] = block_data[18];
            cyclic_data[12] = block_data[19];
            cyclic_data[13] = block_data[20];
            cyclic_data[14] = block_data[21];
            cyclic_data[15] = block_data[22];
            cyclic_data[16] = block_data[23];
            cyclic_data[17] = block_data[24];
            cyclic_data[18] = block_data[25];
            cyclic_data[19] = block_data[26];
            cyclic_data[20] = block_data[27];
            cyclic_data[21] = block_data[28];
            cyclic_data[22] = block_data[29];
            cyclic_data[23] = block_data[30];
            cyclic_data[24] = block_data[31];
            cyclic_data[25] = block_data[0];
            cyclic_data[26] = block_data[1];
            cyclic_data[27] = block_data[2];
            cyclic_data[28] = block_data[3];
            cyclic_data[29] = block_data[4];
            cyclic_data[30] = block_data[5];
            cyclic_data[31] = block_data[6];

            cyclic_offset[0] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[1] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[2] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[3] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[4] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[5] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[6] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[7] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[8] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[9] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[10] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[11] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[12] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[13] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[14] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[15] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[16] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[17] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[18] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[19] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[20] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[21] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[22] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[23] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[24] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[25] = i / PAR;
            cyclic_offset[26] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[27] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[28] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[29] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[30] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[31] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;

        }else if (i%PAR == 26){
            cyclic_data[0] = block_data[6];
            cyclic_data[1] = block_data[7];
            cyclic_data[2] = block_data[8];
            cyclic_data[3] = block_data[9];
            cyclic_data[4] = block_data[10];
            cyclic_data[5] = block_data[11];
            cyclic_data[6] = block_data[12];
            cyclic_data[7] = block_data[13];
            cyclic_data[8] = block_data[14];
            cyclic_data[9] = block_data[15];
            cyclic_data[10] = block_data[16];
            cyclic_data[11] = block_data[17];
            cyclic_data[12] = block_data[18];
            cyclic_data[13] = block_data[19];
            cyclic_data[14] = block_data[20];
            cyclic_data[15] = block_data[21];
            cyclic_data[16] = block_data[22];
            cyclic_data[17] = block_data[23];
            cyclic_data[18] = block_data[24];
            cyclic_data[19] = block_data[25];
            cyclic_data[20] = block_data[26];
            cyclic_data[21] = block_data[27];
            cyclic_data[22] = block_data[28];
            cyclic_data[23] = block_data[29];
            cyclic_data[24] = block_data[30];
            cyclic_data[25] = block_data[31];
            cyclic_data[26] = block_data[0];
            cyclic_data[27] = block_data[1];
            cyclic_data[28] = block_data[2];
            cyclic_data[29] = block_data[3];
            cyclic_data[30] = block_data[4];
            cyclic_data[31] = block_data[5];

            cyclic_offset[0] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[1] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[2] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[3] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[4] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[5] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[6] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[7] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[8] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[9] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[10] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[11] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[12] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[13] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[14] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[15] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[16] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[17] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[18] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[19] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[20] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[21] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[22] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[23] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[24] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[25] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[26] = i / PAR;
            cyclic_offset[27] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[28] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[29] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[30] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[31] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;

        }else if (i%PAR == 27){
            cyclic_data[0] = block_data[5];
            cyclic_data[1] = block_data[6];
            cyclic_data[2] = block_data[7];
            cyclic_data[3] = block_data[8];
            cyclic_data[4] = block_data[9];
            cyclic_data[5] = block_data[10];
            cyclic_data[6] = block_data[11];
            cyclic_data[7] = block_data[12];
            cyclic_data[8] = block_data[13];
            cyclic_data[9] = block_data[14];
            cyclic_data[10] = block_data[15];
            cyclic_data[11] = block_data[16];
            cyclic_data[12] = block_data[17];
            cyclic_data[13] = block_data[18];
            cyclic_data[14] = block_data[19];
            cyclic_data[15] = block_data[20];
            cyclic_data[16] = block_data[21];
            cyclic_data[17] = block_data[22];
            cyclic_data[18] = block_data[23];
            cyclic_data[19] = block_data[24];
            cyclic_data[20] = block_data[25];
            cyclic_data[21] = block_data[26];
            cyclic_data[22] = block_data[27];
            cyclic_data[23] = block_data[28];
            cyclic_data[24] = block_data[29];
            cyclic_data[25] = block_data[30];
            cyclic_data[26] = block_data[31];
            cyclic_data[27] = block_data[0];
            cyclic_data[28] = block_data[1];
            cyclic_data[29] = block_data[2];
            cyclic_data[30] = block_data[3];
            cyclic_data[31] = block_data[4];

            cyclic_offset[0] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[1] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[2] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[3] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[4] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[5] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[6] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[7] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[8] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[9] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[10] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[11] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[12] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[13] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[14] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[15] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[16] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[17] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[18] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[19] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[20] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[21] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[22] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[23] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[24] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[25] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[26] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[27] = i / PAR;
            cyclic_offset[28] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[29] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[30] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[31] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;

        }else if (i%PAR == 28){
            cyclic_data[0] = block_data[4];
            cyclic_data[1] = block_data[5];
            cyclic_data[2] = block_data[6];
            cyclic_data[3] = block_data[7];
            cyclic_data[4] = block_data[8];
            cyclic_data[5] = block_data[9];
            cyclic_data[6] = block_data[10];
            cyclic_data[7] = block_data[11];
            cyclic_data[8] = block_data[12];
            cyclic_data[9] = block_data[13];
            cyclic_data[10] = block_data[14];
            cyclic_data[11] = block_data[15];
            cyclic_data[12] = block_data[16];
            cyclic_data[13] = block_data[17];
            cyclic_data[14] = block_data[18];
            cyclic_data[15] = block_data[19];
            cyclic_data[16] = block_data[20];
            cyclic_data[17] = block_data[21];
            cyclic_data[18] = block_data[22];
            cyclic_data[19] = block_data[23];
            cyclic_data[20] = block_data[24];
            cyclic_data[21] = block_data[25];
            cyclic_data[22] = block_data[26];
            cyclic_data[23] = block_data[27];
            cyclic_data[24] = block_data[28];
            cyclic_data[25] = block_data[29];
            cyclic_data[26] = block_data[30];
            cyclic_data[27] = block_data[31];
            cyclic_data[28] = block_data[0];
            cyclic_data[29] = block_data[1];
            cyclic_data[30] = block_data[2];
            cyclic_data[31] = block_data[3];

            cyclic_offset[0] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[1] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[2] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[3] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[4] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[5] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[6] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[7] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[8] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[9] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[10] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[11] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[12] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[13] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[14] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[15] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[16] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[17] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[18] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[19] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[20] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[21] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[22] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[23] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[24] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[25] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[26] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[27] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[28] = i / PAR;
            cyclic_offset[29] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[30] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[31] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;

        }else if (i%PAR == 29){
            cyclic_data[0] = block_data[3];
            cyclic_data[1] = block_data[4];
            cyclic_data[2] = block_data[5];
            cyclic_data[3] = block_data[6];
            cyclic_data[4] = block_data[7];
            cyclic_data[5] = block_data[8];
            cyclic_data[6] = block_data[9];
            cyclic_data[7] = block_data[10];
            cyclic_data[8] = block_data[11];
            cyclic_data[9] = block_data[12];
            cyclic_data[10] = block_data[13];
            cyclic_data[11] = block_data[14];
            cyclic_data[12] = block_data[15];
            cyclic_data[13] = block_data[16];
            cyclic_data[14] = block_data[17];
            cyclic_data[15] = block_data[18];
            cyclic_data[16] = block_data[19];
            cyclic_data[17] = block_data[20];
            cyclic_data[18] = block_data[21];
            cyclic_data[19] = block_data[22];
            cyclic_data[20] = block_data[23];
            cyclic_data[21] = block_data[24];
            cyclic_data[22] = block_data[25];
            cyclic_data[23] = block_data[26];
            cyclic_data[24] = block_data[27];
            cyclic_data[25] = block_data[28];
            cyclic_data[26] = block_data[29];
            cyclic_data[27] = block_data[30];
            cyclic_data[28] = block_data[31];
            cyclic_data[29] = block_data[0];
            cyclic_data[30] = block_data[1];
            cyclic_data[31] = block_data[2];

            cyclic_offset[0] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[1] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[2] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[3] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[4] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[5] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[6] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[7] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[8] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[9] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[10] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[11] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[12] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[13] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[14] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[15] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[16] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[17] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[18] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[19] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[20] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[21] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[22] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[23] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[24] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[25] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[26] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[27] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[28] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[29] = i / PAR;
            cyclic_offset[30] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;
            cyclic_offset[31] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;

        }else if (i%PAR == 30){
            cyclic_data[0] = block_data[2];
            cyclic_data[1] = block_data[3];
            cyclic_data[2] = block_data[4];
            cyclic_data[3] = block_data[5];
            cyclic_data[4] = block_data[6];
            cyclic_data[5] = block_data[7];
            cyclic_data[6] = block_data[8];
            cyclic_data[7] = block_data[9];
            cyclic_data[8] = block_data[10];
            cyclic_data[9] = block_data[11];
            cyclic_data[10] = block_data[12];
            cyclic_data[11] = block_data[13];
            cyclic_data[12] = block_data[14];
            cyclic_data[13] = block_data[15];
            cyclic_data[14] = block_data[16];
            cyclic_data[15] = block_data[17];
            cyclic_data[16] = block_data[18];
            cyclic_data[17] = block_data[19];
            cyclic_data[18] = block_data[20];
            cyclic_data[19] = block_data[21];
            cyclic_data[20] = block_data[22];
            cyclic_data[21] = block_data[23];
            cyclic_data[22] = block_data[24];
            cyclic_data[23] = block_data[25];
            cyclic_data[24] = block_data[26];
            cyclic_data[25] = block_data[27];
            cyclic_data[26] = block_data[28];
            cyclic_data[27] = block_data[29];
            cyclic_data[28] = block_data[30];
            cyclic_data[29] = block_data[31];
            cyclic_data[30] = block_data[0];
            cyclic_data[31] = block_data[1];

            cyclic_offset[0] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[1] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[2] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[3] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[4] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[5] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[6] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[7] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[8] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[9] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[10] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[11] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[12] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[13] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[14] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[15] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[16] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[17] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[18] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[19] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[20] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[21] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[22] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[23] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[24] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[25] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[26] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[27] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[28] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[29] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[30] = i / PAR;
            cyclic_offset[31] = ((i + 1) % TIME_STEP + TIME_STEP) / PAR;

        }else if (i%PAR == 31){
            cyclic_data[0] = block_data[1];
            cyclic_data[1] = block_data[2];
            cyclic_data[2] = block_data[3];
            cyclic_data[3] = block_data[4];
            cyclic_data[4] = block_data[5];
            cyclic_data[5] = block_data[6];
            cyclic_data[6] = block_data[7];
            cyclic_data[7] = block_data[8];
            cyclic_data[8] = block_data[9];
            cyclic_data[9] = block_data[10];
            cyclic_data[10] = block_data[11];
            cyclic_data[11] = block_data[12];
            cyclic_data[12] = block_data[13];
            cyclic_data[13] = block_data[14];
            cyclic_data[14] = block_data[15];
            cyclic_data[15] = block_data[16];
            cyclic_data[16] = block_data[17];
            cyclic_data[17] = block_data[18];
            cyclic_data[18] = block_data[19];
            cyclic_data[19] = block_data[20];
            cyclic_data[20] = block_data[21];
            cyclic_data[21] = block_data[22];
            cyclic_data[22] = block_data[23];
            cyclic_data[23] = block_data[24];
            cyclic_data[24] = block_data[25];
            cyclic_data[25] = block_data[26];
            cyclic_data[26] = block_data[27];
            cyclic_data[27] = block_data[28];
            cyclic_data[28] = block_data[29];
            cyclic_data[29] = block_data[30];
            cyclic_data[30] = block_data[31];
            cyclic_data[31] = block_data[0];

            cyclic_offset[0] = ((i + 1) % TIME_STEP + TIME_STEP * 1) / PAR;
            cyclic_offset[1] = ((i + 2) % TIME_STEP + TIME_STEP * 2) / PAR;
            cyclic_offset[2] = ((i + 3) % TIME_STEP + TIME_STEP * 3) / PAR;
            cyclic_offset[3] = ((i + 4) % TIME_STEP + TIME_STEP * 4) / PAR;
            cyclic_offset[4] = ((i + 5) % TIME_STEP + TIME_STEP * 5) / PAR;
            cyclic_offset[5] = ((i + 6) % TIME_STEP + TIME_STEP * 6) / PAR;
            cyclic_offset[6] = ((i + 7) % TIME_STEP + TIME_STEP * 7) / PAR;
            cyclic_offset[7] = ((i + 8) % TIME_STEP + TIME_STEP * 8) / PAR;
            cyclic_offset[8] = ((i + 9) % TIME_STEP + TIME_STEP * 9) / PAR;
            cyclic_offset[9] = ((i + 10) % TIME_STEP + TIME_STEP * 10) / PAR;
            cyclic_offset[10] = ((i + 11) % TIME_STEP + TIME_STEP * 11) / PAR;
            cyclic_offset[11] = ((i + 12) % TIME_STEP + TIME_STEP * 12) / PAR;
            cyclic_offset[12] = ((i + 13) % TIME_STEP + TIME_STEP * 13) / PAR;
            cyclic_offset[13] = ((i + 14) % TIME_STEP + TIME_STEP * 14) / PAR;
            cyclic_offset[14] = ((i + 15) % TIME_STEP + TIME_STEP * 15) / PAR;
            cyclic_offset[15] = ((i + 16) % TIME_STEP + TIME_STEP * 16) / PAR;
            cyclic_offset[16] = ((i + 17) % TIME_STEP + TIME_STEP * 17) / PAR;
            cyclic_offset[17] = ((i + 18) % TIME_STEP + TIME_STEP * 18) / PAR;
            cyclic_offset[18] = ((i + 19) % TIME_STEP + TIME_STEP * 19) / PAR;
            cyclic_offset[19] = ((i + 20) % TIME_STEP + TIME_STEP * 20) / PAR;
            cyclic_offset[20] = ((i + 21) % TIME_STEP + TIME_STEP * 21) / PAR;
            cyclic_offset[21] = ((i + 22) % TIME_STEP + TIME_STEP * 22) / PAR;
            cyclic_offset[22] = ((i + 23) % TIME_STEP + TIME_STEP * 23) / PAR;
            cyclic_offset[23] = ((i + 24) % TIME_STEP + TIME_STEP * 24) / PAR;
            cyclic_offset[24] = ((i + 25) % TIME_STEP + TIME_STEP * 25) / PAR;
            cyclic_offset[25] = ((i + 26) % TIME_STEP + TIME_STEP * 26) / PAR;
            cyclic_offset[26] = ((i + 27) % TIME_STEP + TIME_STEP * 27) / PAR;
            cyclic_offset[27] = ((i + 28) % TIME_STEP + TIME_STEP * 28) / PAR;
            cyclic_offset[28] = ((i + 29) % TIME_STEP + TIME_STEP * 29) / PAR;
            cyclic_offset[29] = ((i + 30) % TIME_STEP + TIME_STEP * 30) / PAR;
            cyclic_offset[30] = ((i + 31) % TIME_STEP + TIME_STEP * 31) / PAR;
            cyclic_offset[31] = i / PAR;
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

}


void reverse_write_stream_output (
    complex<float> data_in_cyclic[UF*2][FFT_NUM/(UF*2)],
    hls::stream<hls::vector<complex<float>, UF*2>> & reverse_in_stream_vector
) {
    const int TIME_STEP =  FFT_NUM/(UF*2);
    const int PAR =  UF*2;
    #pragma inline off

    STREAM_OUT_REVERSE: for (int i = 0; i < TIME_STEP; i= i + 1){ 
        #pragma HLS pipeline II=1
        // #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle  
        hls::vector<complex<float>, UF*2> temp;
        for (int u = 0; u < UF*2; u++) {
            temp[u] = data_in_cyclic[u][i];
        }
        reverse_in_stream_vector.write(temp);
    }


}

void reverse_input_stream_UF16 (
    hls::stream<hls::vector<complex<float>, UF*2>> & dataIn,
    hls::stream<hls::vector<complex<float>, UF*2>> & reverse_in_stream_vector
){

    static complex<float> data_in_cyclic[UF*2][FFT_NUM/(UF*2)];

    static complex<float> data_rev_stream[UF*2][FFT_NUM/(UF*2)];
    
    #pragma HLS array_partition variable=data_rev_stream type=complete dim=1 
    #pragma HLS array_partition variable=data_in_cyclic type=complete dim=1 

    #pragma HLS array_partition variable=data_rev_stream type=complete dim=2
    #pragma HLS array_partition variable=data_in_cyclic type=complete dim=2

    #pragma HLS stream type=pipo variable=data_in_cyclic depth=3
    #pragma HLS stream type=pipo variable=data_rev_stream  depth=3

    #pragma HLS pipeline II=FFT_NUM/(2*UF)

    const int TIME_STEP = FFT_NUM/(UF*2);
    const int PAR = UF*2;



    READ_STREAM_INPUT: for (int i = 0; i < TIME_STEP; i++){
        #pragma HLS pipeline II=1
        // #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
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
        data_rev_stream[0][reversed[0]%TIME_STEP] = temp[0];
        data_rev_stream[16][reversed[1]%TIME_STEP] = temp[1];
        data_rev_stream[8][reversed[2]%TIME_STEP] = temp[2];
        data_rev_stream[24][reversed[3]%TIME_STEP] = temp[3];
        data_rev_stream[4][reversed[4]%TIME_STEP] = temp[4];
        data_rev_stream[20][reversed[5]%TIME_STEP] = temp[5];
        data_rev_stream[12][reversed[6]%TIME_STEP] = temp[6];
        data_rev_stream[28][reversed[7]%TIME_STEP] = temp[7];
        data_rev_stream[2][reversed[8]%TIME_STEP] = temp[8];
        data_rev_stream[18][reversed[9]%TIME_STEP] = temp[9];
        data_rev_stream[10][reversed[10]%TIME_STEP] = temp[10];
        data_rev_stream[26][reversed[11]%TIME_STEP] = temp[11];
        data_rev_stream[6][reversed[12]%TIME_STEP] = temp[12];
        data_rev_stream[22][reversed[13]%TIME_STEP] = temp[13];
        data_rev_stream[14][reversed[14]%TIME_STEP] = temp[14];
        data_rev_stream[30][reversed[15]%TIME_STEP] = temp[15];
        data_rev_stream[1][reversed[16]%TIME_STEP] = temp[16];
        data_rev_stream[17][reversed[17]%TIME_STEP] = temp[17];
        data_rev_stream[9][reversed[18]%TIME_STEP] = temp[18];
        data_rev_stream[25][reversed[19]%TIME_STEP] = temp[19];
        data_rev_stream[5][reversed[20]%TIME_STEP] = temp[20];
        data_rev_stream[21][reversed[21]%TIME_STEP] = temp[21];
        data_rev_stream[13][reversed[22]%TIME_STEP] = temp[22];
        data_rev_stream[29][reversed[23]%TIME_STEP] = temp[23];
        data_rev_stream[3][reversed[24]%TIME_STEP] = temp[24];
        data_rev_stream[19][reversed[25]%TIME_STEP] = temp[25];
        data_rev_stream[11][reversed[26]%TIME_STEP] = temp[26];
        data_rev_stream[27][reversed[27]%TIME_STEP] = temp[27];
        data_rev_stream[7][reversed[28]%TIME_STEP] = temp[28];
        data_rev_stream[23][reversed[29]%TIME_STEP] = temp[29];
        data_rev_stream[15][reversed[30]%TIME_STEP] = temp[30];
        data_rev_stream[31][reversed[31]%TIME_STEP] = temp[31];
    }



    FROM_BLOCK_TO_CYCLIC_SIMPLE: for (int i = 0; i < TIME_STEP; i= i + 1){ 
        #pragma HLS pipeline II=1
        #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
        for (int u = 0; u < UF*2; u++) {
            int index = i * UF*2 + u;
            data_in_cyclic[u][i] = data_rev_stream[index/TIME_STEP][index%TIME_STEP];
        }
    }

    STREAM_OUT_REVERSE: for (int i = 0; i < TIME_STEP; i= i + 1){ 
        #pragma HLS pipeline II=1
        #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle  
        hls::vector<complex<float>, UF*2> temp;
        for (int u = 0; u < UF*2; u++) {
            temp[u] = data_in_cyclic[u][i];
        }
        reverse_in_stream_vector.write(temp);
    }



    // reverse_read_stream_input(dataIn, data_rev_stream);
    // reverse_from_block_to_cyclic(data_rev_stream,data_in_cyclic);
    // reverse_write_stream_output (data_in_cyclic, reverse_in_stream_vector);

    // initialize data_rev_stream
    // for (int i = 0; i < UF*2; i++){
    //     for(int j = 0; j < TIME_STEP; j++){
    //         data_rev_stream[i][j] = i * TIME_STEP + j;
    //     }
    // }


    // // print data_in_cyclic
    // for(int j = 0; j < TIME_STEP; j++){
    //     for (int i = 0; i < UF*2; i++){
    //         int index = j*UF*2+i;
    //         if(data_in_cyclic[i][j] != data_rev_stream[index/TIME_STEP][index%TIME_STEP]) {
    //             cout << "!!! time step" << i << " unroll id" << j << " " << endl;
    //             cout << "index " << index << ": rev "<< data_rev_stream[index/TIME_STEP][index%TIME_STEP] << " cyclic " << data_in_cyclic[i][j] << endl;
    //         }
    //     }
    // }


    // for (int i = 0; i < TIME_STEP; i= i + 1){ 
    //     for (int u = 0; u < UF*2; u++) {
    //         int index = i * UF*2 + u;
    //         data_in_cyclic[u][i] = data_rev_stream[index/TIME_STEP][index%TIME_STEP];
    //     }
    // }


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
    // #pragma HLS performance target_ti=FFT_NUM/(2*UF) unit=cycle
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
    #pragma HLS array_partition variable=data_2 type=cyclic factor=UF*2 dim=1
    #pragma HLS bind_storage variable=data_2 type=RAM_2P impl=LUTRAM
    // #pragma HLS array_partition variable=data_1 type=cyclic factor=UF dim=1
    // #pragma HLS array_partition variable=data_2 type=cyclic factor=UF dim=1
    // #pragma HLS array_partition variable=data_3 type=cyclic factor=UF dim=1
    #pragma HLS array_partition variable=data_3 type=cyclic factor=UF*2 dim=1
    #pragma HLS bind_storage variable=data_3 type=RAM_2P impl=LUTRAM
    #pragma HLS array_partition variable=data_4 type=cyclic factor=UF dim=1
    #pragma HLS array_partition variable=data_5 type=cyclic factor=UF dim=1
    #pragma HLS array_partition variable=data_6 type=cyclic factor=UF dim=1
    #pragma HLS array_partition variable=data_7 type=cyclic factor=UF dim=1
    #pragma HLS array_partition variable=data_8 type=cyclic factor=UF dim=1



    hls::stream<hls::vector<complex<float>, UF*2>> reverse_in_stream_vector; 
    hls::stream<hls::vector<complex<float>, UF*2>> data_s1_stream_vector; 

    reverse_input_stream_UF16(dataIn, reverse_in_stream_vector);
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