#include <cstdint>

#include <tapa.h>
#include "ntt.h"

// /********************************************************************************************
// * Polynomial load store tasks *
// ********************************************************************************************/

void fwd_load_inv_store_poly_1_limbs(uint64_t  poly_mmap,
                 tapa::istreams<WORD, 2*V_BUG_SIZE>& inv_in_poly_L0_stream0,
                 tapa::istreams<WORD, 2*V_BUG_SIZE>& inv_in_poly_L0_stream1,
                 tapa::istreams<WORD, 2*V_BUG_SIZE>& inv_in_poly_L0_stream2,
                 tapa::istreams<WORD, 2*V_BUG_SIZE>& inv_in_poly_L0_stream3,
                 tapa::ostreams<WORD, 2*V_BUG_SIZE>& fwd_out_poly_L0_stream0,
                 tapa::ostreams<WORD, 2*V_BUG_SIZE>& fwd_out_poly_L0_stream1,
                 tapa::ostreams<WORD, 2*V_BUG_SIZE>& fwd_out_poly_L0_stream2,
                 tapa::ostreams<WORD, 2*V_BUG_SIZE>& fwd_out_poly_L0_stream3,
                 bool direction,
                 VAR_TYPE_16 iter) ;


void fwd_store_inv_load_poly_1_limbs(uint64_t  poly_mmap,
                 tapa::istreams<WORD, 2*V_BUG_SIZE>& fwd_in_poly_L0_stream0,
                 tapa::istreams<WORD, 2*V_BUG_SIZE>& fwd_in_poly_L0_stream1,
                 tapa::istreams<WORD, 2*V_BUG_SIZE>& fwd_in_poly_L0_stream2,
                 tapa::istreams<WORD, 2*V_BUG_SIZE>& fwd_in_poly_L0_stream3,
                 tapa::ostreams<WORD, 2*V_BUG_SIZE>& inv_out_poly_L0_stream0,
                 tapa::ostreams<WORD, 2*V_BUG_SIZE>& inv_out_poly_L0_stream1,
                 tapa::ostreams<WORD, 2*V_BUG_SIZE>& inv_out_poly_L0_stream2,
                 tapa::ostreams<WORD, 2*V_BUG_SIZE>& inv_out_poly_L0_stream3,
                 bool direction,
                 VAR_TYPE_16 iter) ;


// /********************************************************************************************
// * TF load tasks *
// ********************************************************************************************/

void Mmap2Stream_tf_0_1_limbs(uint64_t  tf_mmap,
                 tapa::ostreams<DWORD, TF_CONCAT_FACTOR_PER_PARA_LIMB_PORT/2>& tf_stream_L0_0,
                 bool direction,
                 VAR_TYPE_16 iter) ;

// /********************************************************************************************
// * Function to multiply with two inverse *
// ********************************************************************************************/

void multiplyTwoInverse(WORD *val, WORD *twoInverse){
  #pragma HLS inline
  *val = (*val>>1) + (*val & 1)*(*twoInverse);
}

// /********************************************************************************************
// * Modulo multiplication functions *
// ********************************************************************************************/

ap_uint<64> hardcoded_fullMul_0(ap_uint<32> u, ap_uint<32> v){
  #pragma HLS inline

  ap_uint<17> u_0 = u & ((1<<17)-1);
  u = u >> 17;
  ap_uint<15> u_1 = u & ((1<<15)-1);

  ap_uint<27> v_0 = v & ((1<<27)-1);
  v = v >> 27;
  ap_uint<5> v_1 = v & ((1<<5)-1);
  
  ap_uint<44> mul_0 = ((ap_uint<44>)(u_0) * v_0); //<<0
  #pragma HLS bind_op variable=mul_0 op=mul impl=dsp
  ap_uint<42> mul_1 = ((ap_uint<42>)(u_1) * v_0); //<<17
  #pragma HLS bind_op variable=mul_1 op=mul impl=dsp

  ap_uint<22> mul_2 = ((ap_uint<22>)(u_0) * v_1); //<<27
  #pragma HLS bind_op variable=mul_2 op=mul impl=dsp
  ap_uint<20> mul_3 = ((ap_uint<20>)(u_1) * v_1); //<<44
  #pragma HLS bind_op variable=mul_3 op=mul impl=dsp

  
  ap_uint<64> res = ( (((ap_uint<64>)mul_0)<<0) + (((ap_uint<64>)mul_1)<<17) + (((ap_uint<64>)mul_2)<<27) + (((ap_uint<64>)mul_3)<<44) );
  return res;
}

ap_uint<33> hardcoded_fullMul_1(ap_uint<35> u, ap_uint<32> v){
  #pragma HLS inline

  ap_uint<18> u_0 = u & ((1<<18)-1);
  u = u >> 18;
  ap_uint<17> u_1 = u & ((1<<17)-1);

  ap_uint<26> v_0 = v & ((1<<26)-1);
  v = v >> 26;
  ap_uint<6> v_1 = v & ((1<<6)-1);
  
  ap_uint<44> mul_0 = ((ap_uint<44>)(u_0) * v_0); //<<0
  #pragma HLS bind_op variable=mul_0 op=mul impl=dsp
  ap_uint<43> mul_1 = ((ap_uint<43>)(u_1) * v_0); //<<18
  #pragma HLS bind_op variable=mul_1 op=mul impl=dsp

  ap_uint<24> mul_2 = ((ap_uint<24>)(u_0) * v_1); //<<26
  #pragma HLS bind_op variable=mul_2 op=mul impl=dsp
  ap_uint<23> mul_3 = ((ap_uint<23>)(u_1) * v_1); //<<44
  #pragma HLS bind_op variable=mul_3 op=mul impl=dsp

  
  ap_uint<33> res = ( (((ap_uint<67>)mul_0)<<0) + (((ap_uint<67>)mul_1)<<18) + (((ap_uint<67>)mul_2)<<26) + (((ap_uint<67>)mul_3)<<44) ) >> (WORD_SIZE+2);
  return res;
}

ap_uint<34> hardcoded_lowerHalfMul_2(ap_uint<33> u, ap_uint<32> v){
  #pragma HLS inline

  ap_uint<17> u_0 = u & ((1<<17)-1);
  u = u >> 17;
  ap_uint<16> u_1 = u & ((1<<16)-1);

  ap_uint<27> v_0 = v & ((1<<27)-1);
  v = v >> 27;
  ap_uint<5> v_1 = v & ((1<<5)-1);
  
  ap_uint<44> mul_0 = ((ap_uint<44>)(u_0) * v_0); //<<0
  #pragma HLS bind_op variable=mul_0 op=mul impl=dsp
  ap_uint<43> mul_1 = ((ap_uint<43>)(u_1) * v_0); //<<17
  #pragma HLS bind_op variable=mul_1 op=mul impl=dsp

  ap_uint<22> mul_2 = ((ap_uint<22>)(u_0) * v_1); //<<27
  #pragma HLS bind_op variable=mul_2 op=mul impl=dsp

  
  ap_uint<34> res = ( (((ap_uint<34>)mul_0)<<0) + (((ap_uint<34>)mul_1)<<17) + (((ap_uint<34>)mul_2)<<27) ) & WORD_SIZE_PLUS2_MASK;
  return res;
}

WORD barrett_reduce(WORD u, WORD v, WORD mod, WORD_PLUS3 factor)
{
  #pragma HLS inline

  DWORD U = hardcoded_fullMul_0(u,v);
  WORD_PLUS2 Y = (U) & WORD_SIZE_PLUS2_MASK;

  WORD V = U >> WORD_SIZE;
  WORD_PLUS1 W_1 = hardcoded_fullMul_1(factor, V);

  WORD_PLUS2 X_1 = hardcoded_lowerHalfMul_2(W_1, mod);

  WORD_PLUS1 Z;
  if(X_1<=Y){
    Z = Y - X_1;
  }
  else{
    Z = (WORD_PLUS3(1)<<(WORD_SIZE+2)) - X_1 + Y;
  }

  if(Z >= (3*mod)){
    Z = Z - 3*mod;
  }
  else if(Z >= (2*mod)){
    Z = Z - 2*mod;
  }
  else if(Z >= mod){
    Z = Z - mod;
  }

	return (WORD)Z;
}

// /********************************************************************************************
// * Butterfly Unit(BU) *
// ********************************************************************************************/

void BU(
    tapa::istream<WORD>& fwd_inVal0, 
    tapa::istream<WORD>& fwd_inVal1, 
    tapa::istream<WORD>& inv_inVal0,
    tapa::istream<WORD>& inv_inVal1, 
    tapa::istream<WORD>& tf_inVal,
    tapa::ostream<WORD>& fwd_outVal0,
    tapa::ostream<WORD>& fwd_outVal1,
    tapa::ostream<WORD>& inv_outVal0,
    tapa::ostream<WORD>& inv_outVal1,
    WORD q,
    WORD twoInverse,
    WORD_PLUS3 factor,
    bool direction,
    VAR_TYPE_16 iter,
    VAR_TYPE_16 BUG_id,
    VAR_TYPE_16 layer_id
    );

// /********************************************************************************************
// * Shuffler tasks *
// ********************************************************************************************/

void shuffler_in(tapa::istreams<WORD, 2*V_BUG_SIZE>& fwd_inVal,
                  tapa::istreams<WORD, 2*V_BUG_SIZE>& inv_inVal,
                   tapa::ostreams<WORD, 2*V_BUG_SIZE>& fwd_outVal,
                   tapa::ostreams<WORD, 2*V_BUG_SIZE>& inv_outVal,
                   bool direction,
                   VAR_TYPE_16 iter,
                   VAR_TYPE_16 task_idx
                    ){
#pragma HLS disaggregate variable = fwd_inVal
#pragma HLS array_partition variable = fwd_inVal complete
#pragma HLS interface ap_fifo port = fwd_inVal[0]._
#pragma HLS aggregate variable = fwd_inVal[0]._ bit
#pragma HLS interface ap_fifo port = fwd_inVal[0]._peek
#pragma HLS aggregate variable = fwd_inVal[0]._peek bit
void(fwd_inVal[0]._.empty());
void(fwd_inVal[0]._peek.empty());
#pragma HLS interface ap_fifo port = fwd_inVal[1]._
#pragma HLS aggregate variable = fwd_inVal[1]._ bit
#pragma HLS interface ap_fifo port = fwd_inVal[1]._peek
#pragma HLS aggregate variable = fwd_inVal[1]._peek bit
void(fwd_inVal[1]._.empty());
void(fwd_inVal[1]._peek.empty());
#pragma HLS interface ap_fifo port = fwd_inVal[2]._
#pragma HLS aggregate variable = fwd_inVal[2]._ bit
#pragma HLS interface ap_fifo port = fwd_inVal[2]._peek
#pragma HLS aggregate variable = fwd_inVal[2]._peek bit
void(fwd_inVal[2]._.empty());
void(fwd_inVal[2]._peek.empty());
#pragma HLS interface ap_fifo port = fwd_inVal[3]._
#pragma HLS aggregate variable = fwd_inVal[3]._ bit
#pragma HLS interface ap_fifo port = fwd_inVal[3]._peek
#pragma HLS aggregate variable = fwd_inVal[3]._peek bit
void(fwd_inVal[3]._.empty());
void(fwd_inVal[3]._peek.empty());
#pragma HLS interface ap_fifo port = fwd_inVal[4]._
#pragma HLS aggregate variable = fwd_inVal[4]._ bit
#pragma HLS interface ap_fifo port = fwd_inVal[4]._peek
#pragma HLS aggregate variable = fwd_inVal[4]._peek bit
void(fwd_inVal[4]._.empty());
void(fwd_inVal[4]._peek.empty());
#pragma HLS interface ap_fifo port = fwd_inVal[5]._
#pragma HLS aggregate variable = fwd_inVal[5]._ bit
#pragma HLS interface ap_fifo port = fwd_inVal[5]._peek
#pragma HLS aggregate variable = fwd_inVal[5]._peek bit
void(fwd_inVal[5]._.empty());
void(fwd_inVal[5]._peek.empty());
#pragma HLS interface ap_fifo port = fwd_inVal[6]._
#pragma HLS aggregate variable = fwd_inVal[6]._ bit
#pragma HLS interface ap_fifo port = fwd_inVal[6]._peek
#pragma HLS aggregate variable = fwd_inVal[6]._peek bit
void(fwd_inVal[6]._.empty());
void(fwd_inVal[6]._peek.empty());
#pragma HLS interface ap_fifo port = fwd_inVal[7]._
#pragma HLS aggregate variable = fwd_inVal[7]._ bit
#pragma HLS interface ap_fifo port = fwd_inVal[7]._peek
#pragma HLS aggregate variable = fwd_inVal[7]._peek bit
void(fwd_inVal[7]._.empty());
void(fwd_inVal[7]._peek.empty());
#pragma HLS interface ap_fifo port = fwd_inVal[8]._
#pragma HLS aggregate variable = fwd_inVal[8]._ bit
#pragma HLS interface ap_fifo port = fwd_inVal[8]._peek
#pragma HLS aggregate variable = fwd_inVal[8]._peek bit
void(fwd_inVal[8]._.empty());
void(fwd_inVal[8]._peek.empty());
#pragma HLS interface ap_fifo port = fwd_inVal[9]._
#pragma HLS aggregate variable = fwd_inVal[9]._ bit
#pragma HLS interface ap_fifo port = fwd_inVal[9]._peek
#pragma HLS aggregate variable = fwd_inVal[9]._peek bit
void(fwd_inVal[9]._.empty());
void(fwd_inVal[9]._peek.empty());
#pragma HLS interface ap_fifo port = fwd_inVal[10]._
#pragma HLS aggregate variable = fwd_inVal[10]._ bit
#pragma HLS interface ap_fifo port = fwd_inVal[10]._peek
#pragma HLS aggregate variable = fwd_inVal[10]._peek bit
void(fwd_inVal[10]._.empty());
void(fwd_inVal[10]._peek.empty());
#pragma HLS interface ap_fifo port = fwd_inVal[11]._
#pragma HLS aggregate variable = fwd_inVal[11]._ bit
#pragma HLS interface ap_fifo port = fwd_inVal[11]._peek
#pragma HLS aggregate variable = fwd_inVal[11]._peek bit
void(fwd_inVal[11]._.empty());
void(fwd_inVal[11]._peek.empty());
#pragma HLS interface ap_fifo port = fwd_inVal[12]._
#pragma HLS aggregate variable = fwd_inVal[12]._ bit
#pragma HLS interface ap_fifo port = fwd_inVal[12]._peek
#pragma HLS aggregate variable = fwd_inVal[12]._peek bit
void(fwd_inVal[12]._.empty());
void(fwd_inVal[12]._peek.empty());
#pragma HLS interface ap_fifo port = fwd_inVal[13]._
#pragma HLS aggregate variable = fwd_inVal[13]._ bit
#pragma HLS interface ap_fifo port = fwd_inVal[13]._peek
#pragma HLS aggregate variable = fwd_inVal[13]._peek bit
void(fwd_inVal[13]._.empty());
void(fwd_inVal[13]._peek.empty());
#pragma HLS interface ap_fifo port = fwd_inVal[14]._
#pragma HLS aggregate variable = fwd_inVal[14]._ bit
#pragma HLS interface ap_fifo port = fwd_inVal[14]._peek
#pragma HLS aggregate variable = fwd_inVal[14]._peek bit
void(fwd_inVal[14]._.empty());
void(fwd_inVal[14]._peek.empty());
#pragma HLS interface ap_fifo port = fwd_inVal[15]._
#pragma HLS aggregate variable = fwd_inVal[15]._ bit
#pragma HLS interface ap_fifo port = fwd_inVal[15]._peek
#pragma HLS aggregate variable = fwd_inVal[15]._peek bit
void(fwd_inVal[15]._.empty());
void(fwd_inVal[15]._peek.empty());

#pragma HLS disaggregate variable = inv_inVal
#pragma HLS array_partition variable = inv_inVal complete
#pragma HLS interface ap_fifo port = inv_inVal[0]._
#pragma HLS aggregate variable = inv_inVal[0]._ bit
#pragma HLS interface ap_fifo port = inv_inVal[0]._peek
#pragma HLS aggregate variable = inv_inVal[0]._peek bit
void(inv_inVal[0]._.empty());
void(inv_inVal[0]._peek.empty());
#pragma HLS interface ap_fifo port = inv_inVal[1]._
#pragma HLS aggregate variable = inv_inVal[1]._ bit
#pragma HLS interface ap_fifo port = inv_inVal[1]._peek
#pragma HLS aggregate variable = inv_inVal[1]._peek bit
void(inv_inVal[1]._.empty());
void(inv_inVal[1]._peek.empty());
#pragma HLS interface ap_fifo port = inv_inVal[2]._
#pragma HLS aggregate variable = inv_inVal[2]._ bit
#pragma HLS interface ap_fifo port = inv_inVal[2]._peek
#pragma HLS aggregate variable = inv_inVal[2]._peek bit
void(inv_inVal[2]._.empty());
void(inv_inVal[2]._peek.empty());
#pragma HLS interface ap_fifo port = inv_inVal[3]._
#pragma HLS aggregate variable = inv_inVal[3]._ bit
#pragma HLS interface ap_fifo port = inv_inVal[3]._peek
#pragma HLS aggregate variable = inv_inVal[3]._peek bit
void(inv_inVal[3]._.empty());
void(inv_inVal[3]._peek.empty());
#pragma HLS interface ap_fifo port = inv_inVal[4]._
#pragma HLS aggregate variable = inv_inVal[4]._ bit
#pragma HLS interface ap_fifo port = inv_inVal[4]._peek
#pragma HLS aggregate variable = inv_inVal[4]._peek bit
void(inv_inVal[4]._.empty());
void(inv_inVal[4]._peek.empty());
#pragma HLS interface ap_fifo port = inv_inVal[5]._
#pragma HLS aggregate variable = inv_inVal[5]._ bit
#pragma HLS interface ap_fifo port = inv_inVal[5]._peek
#pragma HLS aggregate variable = inv_inVal[5]._peek bit
void(inv_inVal[5]._.empty());
void(inv_inVal[5]._peek.empty());
#pragma HLS interface ap_fifo port = inv_inVal[6]._
#pragma HLS aggregate variable = inv_inVal[6]._ bit
#pragma HLS interface ap_fifo port = inv_inVal[6]._peek
#pragma HLS aggregate variable = inv_inVal[6]._peek bit
void(inv_inVal[6]._.empty());
void(inv_inVal[6]._peek.empty());
#pragma HLS interface ap_fifo port = inv_inVal[7]._
#pragma HLS aggregate variable = inv_inVal[7]._ bit
#pragma HLS interface ap_fifo port = inv_inVal[7]._peek
#pragma HLS aggregate variable = inv_inVal[7]._peek bit
void(inv_inVal[7]._.empty());
void(inv_inVal[7]._peek.empty());
#pragma HLS interface ap_fifo port = inv_inVal[8]._
#pragma HLS aggregate variable = inv_inVal[8]._ bit
#pragma HLS interface ap_fifo port = inv_inVal[8]._peek
#pragma HLS aggregate variable = inv_inVal[8]._peek bit
void(inv_inVal[8]._.empty());
void(inv_inVal[8]._peek.empty());
#pragma HLS interface ap_fifo port = inv_inVal[9]._
#pragma HLS aggregate variable = inv_inVal[9]._ bit
#pragma HLS interface ap_fifo port = inv_inVal[9]._peek
#pragma HLS aggregate variable = inv_inVal[9]._peek bit
void(inv_inVal[9]._.empty());
void(inv_inVal[9]._peek.empty());
#pragma HLS interface ap_fifo port = inv_inVal[10]._
#pragma HLS aggregate variable = inv_inVal[10]._ bit
#pragma HLS interface ap_fifo port = inv_inVal[10]._peek
#pragma HLS aggregate variable = inv_inVal[10]._peek bit
void(inv_inVal[10]._.empty());
void(inv_inVal[10]._peek.empty());
#pragma HLS interface ap_fifo port = inv_inVal[11]._
#pragma HLS aggregate variable = inv_inVal[11]._ bit
#pragma HLS interface ap_fifo port = inv_inVal[11]._peek
#pragma HLS aggregate variable = inv_inVal[11]._peek bit
void(inv_inVal[11]._.empty());
void(inv_inVal[11]._peek.empty());
#pragma HLS interface ap_fifo port = inv_inVal[12]._
#pragma HLS aggregate variable = inv_inVal[12]._ bit
#pragma HLS interface ap_fifo port = inv_inVal[12]._peek
#pragma HLS aggregate variable = inv_inVal[12]._peek bit
void(inv_inVal[12]._.empty());
void(inv_inVal[12]._peek.empty());
#pragma HLS interface ap_fifo port = inv_inVal[13]._
#pragma HLS aggregate variable = inv_inVal[13]._ bit
#pragma HLS interface ap_fifo port = inv_inVal[13]._peek
#pragma HLS aggregate variable = inv_inVal[13]._peek bit
void(inv_inVal[13]._.empty());
void(inv_inVal[13]._peek.empty());
#pragma HLS interface ap_fifo port = inv_inVal[14]._
#pragma HLS aggregate variable = inv_inVal[14]._ bit
#pragma HLS interface ap_fifo port = inv_inVal[14]._peek
#pragma HLS aggregate variable = inv_inVal[14]._peek bit
void(inv_inVal[14]._.empty());
void(inv_inVal[14]._peek.empty());
#pragma HLS interface ap_fifo port = inv_inVal[15]._
#pragma HLS aggregate variable = inv_inVal[15]._ bit
#pragma HLS interface ap_fifo port = inv_inVal[15]._peek
#pragma HLS aggregate variable = inv_inVal[15]._peek bit
void(inv_inVal[15]._.empty());
void(inv_inVal[15]._peek.empty());

#pragma HLS disaggregate variable = fwd_outVal
#pragma HLS array_partition variable = fwd_outVal complete
#pragma HLS interface ap_fifo port = fwd_outVal[0]._
#pragma HLS aggregate variable = fwd_outVal[0]._ bit
void(fwd_outVal[0]._.full());
#pragma HLS interface ap_fifo port = fwd_outVal[1]._
#pragma HLS aggregate variable = fwd_outVal[1]._ bit
void(fwd_outVal[1]._.full());
#pragma HLS interface ap_fifo port = fwd_outVal[2]._
#pragma HLS aggregate variable = fwd_outVal[2]._ bit
void(fwd_outVal[2]._.full());
#pragma HLS interface ap_fifo port = fwd_outVal[3]._
#pragma HLS aggregate variable = fwd_outVal[3]._ bit
void(fwd_outVal[3]._.full());
#pragma HLS interface ap_fifo port = fwd_outVal[4]._
#pragma HLS aggregate variable = fwd_outVal[4]._ bit
void(fwd_outVal[4]._.full());
#pragma HLS interface ap_fifo port = fwd_outVal[5]._
#pragma HLS aggregate variable = fwd_outVal[5]._ bit
void(fwd_outVal[5]._.full());
#pragma HLS interface ap_fifo port = fwd_outVal[6]._
#pragma HLS aggregate variable = fwd_outVal[6]._ bit
void(fwd_outVal[6]._.full());
#pragma HLS interface ap_fifo port = fwd_outVal[7]._
#pragma HLS aggregate variable = fwd_outVal[7]._ bit
void(fwd_outVal[7]._.full());
#pragma HLS interface ap_fifo port = fwd_outVal[8]._
#pragma HLS aggregate variable = fwd_outVal[8]._ bit
void(fwd_outVal[8]._.full());
#pragma HLS interface ap_fifo port = fwd_outVal[9]._
#pragma HLS aggregate variable = fwd_outVal[9]._ bit
void(fwd_outVal[9]._.full());
#pragma HLS interface ap_fifo port = fwd_outVal[10]._
#pragma HLS aggregate variable = fwd_outVal[10]._ bit
void(fwd_outVal[10]._.full());
#pragma HLS interface ap_fifo port = fwd_outVal[11]._
#pragma HLS aggregate variable = fwd_outVal[11]._ bit
void(fwd_outVal[11]._.full());
#pragma HLS interface ap_fifo port = fwd_outVal[12]._
#pragma HLS aggregate variable = fwd_outVal[12]._ bit
void(fwd_outVal[12]._.full());
#pragma HLS interface ap_fifo port = fwd_outVal[13]._
#pragma HLS aggregate variable = fwd_outVal[13]._ bit
void(fwd_outVal[13]._.full());
#pragma HLS interface ap_fifo port = fwd_outVal[14]._
#pragma HLS aggregate variable = fwd_outVal[14]._ bit
void(fwd_outVal[14]._.full());
#pragma HLS interface ap_fifo port = fwd_outVal[15]._
#pragma HLS aggregate variable = fwd_outVal[15]._ bit
void(fwd_outVal[15]._.full());

#pragma HLS disaggregate variable = inv_outVal
#pragma HLS array_partition variable = inv_outVal complete
#pragma HLS interface ap_fifo port = inv_outVal[0]._
#pragma HLS aggregate variable = inv_outVal[0]._ bit
void(inv_outVal[0]._.full());
#pragma HLS interface ap_fifo port = inv_outVal[1]._
#pragma HLS aggregate variable = inv_outVal[1]._ bit
void(inv_outVal[1]._.full());
#pragma HLS interface ap_fifo port = inv_outVal[2]._
#pragma HLS aggregate variable = inv_outVal[2]._ bit
void(inv_outVal[2]._.full());
#pragma HLS interface ap_fifo port = inv_outVal[3]._
#pragma HLS aggregate variable = inv_outVal[3]._ bit
void(inv_outVal[3]._.full());
#pragma HLS interface ap_fifo port = inv_outVal[4]._
#pragma HLS aggregate variable = inv_outVal[4]._ bit
void(inv_outVal[4]._.full());
#pragma HLS interface ap_fifo port = inv_outVal[5]._
#pragma HLS aggregate variable = inv_outVal[5]._ bit
void(inv_outVal[5]._.full());
#pragma HLS interface ap_fifo port = inv_outVal[6]._
#pragma HLS aggregate variable = inv_outVal[6]._ bit
void(inv_outVal[6]._.full());
#pragma HLS interface ap_fifo port = inv_outVal[7]._
#pragma HLS aggregate variable = inv_outVal[7]._ bit
void(inv_outVal[7]._.full());
#pragma HLS interface ap_fifo port = inv_outVal[8]._
#pragma HLS aggregate variable = inv_outVal[8]._ bit
void(inv_outVal[8]._.full());
#pragma HLS interface ap_fifo port = inv_outVal[9]._
#pragma HLS aggregate variable = inv_outVal[9]._ bit
void(inv_outVal[9]._.full());
#pragma HLS interface ap_fifo port = inv_outVal[10]._
#pragma HLS aggregate variable = inv_outVal[10]._ bit
void(inv_outVal[10]._.full());
#pragma HLS interface ap_fifo port = inv_outVal[11]._
#pragma HLS aggregate variable = inv_outVal[11]._ bit
void(inv_outVal[11]._.full());
#pragma HLS interface ap_fifo port = inv_outVal[12]._
#pragma HLS aggregate variable = inv_outVal[12]._ bit
void(inv_outVal[12]._.full());
#pragma HLS interface ap_fifo port = inv_outVal[13]._
#pragma HLS aggregate variable = inv_outVal[13]._ bit
void(inv_outVal[13]._.full());
#pragma HLS interface ap_fifo port = inv_outVal[14]._
#pragma HLS aggregate variable = inv_outVal[14]._ bit
void(inv_outVal[14]._.full());
#pragma HLS interface ap_fifo port = inv_outVal[15]._
#pragma HLS aggregate variable = inv_outVal[15]._ bit
void(inv_outVal[15]._.full());




  
  const VAR_TYPE_16 dataCount = (N/(V_TOTAL_DATA));
  const VAR_TYPE_8 shuffleLimit = ( (VAR_TYPE_32)( (logN+(H_BUG_SIZE-1))/H_BUG_SIZE ) ) - 1;
  VAR_TYPE_8 max_data_offset = (V_BUG_SIZE*2);

  VAR_TYPE_8 mini_shift = 0;
  VAR_TYPE_8 major_shift = 0;
  VAR_TYPE_8 shift_amount = 0;

  WORD inValRegArr[2*V_BUG_SIZE];
  WORD outValRegArr[2*V_BUG_SIZE];

  //==== Auto generated variables start ====//
  VAR_TYPE_8 log_shift_freq_table[shuffleLimit] = {0, 2};
  VAR_TYPE_8 log_mini_shift_reset_freq_table[shuffleLimit] = {0, 0};
  VAR_TYPE_8 log_mini_shift_amount_table[shuffleLimit] = {0, 0};
  VAR_TYPE_8 log_shift_jump_table[shuffleLimit] = {2, 0};
  //==== Auto generated variables end ====//

  bool inpStreamNotEmpty;
  VAR_TYPE_8 comp_shuffler_id;
  for(VAR_TYPE_16 iterCount=0; iterCount<(iter+1); iterCount++){
    for(VAR_TYPE_8 shuffler_id=0, inv_shuffler_id=shuffleLimit-1; shuffler_id<shuffleLimit; shuffler_id++, inv_shuffler_id--){
      IN_SHUF_COMP:for(VAR_TYPE_16 i=0; i<dataCount;){
        #pragma HLS PIPELINE II = 1

        if(direction){
          comp_shuffler_id = shuffler_id;
        }
        else{
          comp_shuffler_id = shuffleLimit - shuffler_id - 1;
        }

        VAR_TYPE_8 log_shift_freq = log_shift_freq_table[comp_shuffler_id];
        VAR_TYPE_8 log_mini_shift_reset_freq = log_mini_shift_reset_freq_table[comp_shuffler_id];
        VAR_TYPE_8 log_shift_jump = log_shift_jump_table[comp_shuffler_id];
        VAR_TYPE_8 log_mini_shift_amount = log_mini_shift_amount_table[comp_shuffler_id];

        inpStreamNotEmpty = ( !fwd_inVal[0].empty() ) || ( !inv_inVal[0].empty() ) ;
        
        //Read data from input stream
        for(VAR_TYPE_8 j=1; j<2*V_BUG_SIZE; j++){
          #pragma HLS UNROLL
          inpStreamNotEmpty &= ( !fwd_inVal[j].empty() ) || ( !inv_inVal[j].empty() );
        }

        if(inpStreamNotEmpty){

          for(VAR_TYPE_8 j=0; j<2*V_BUG_SIZE; j++){
            #pragma HLS UNROLL
            if(direction){
              inValRegArr[j] = fwd_inVal[j].read();
            }
            else{
              inValRegArr[j] = inv_inVal[j].read();
            }
          }

          //shift amount calculate
          major_shift = ( ( i & ( (1<<log_shift_freq) - 1 ) ) == 0) ? (((i >> log_shift_freq) << log_shift_jump) % (max_data_offset)) : (major_shift);
          mini_shift = ( ( i & ( (1<<log_mini_shift_reset_freq) - 1 ) ) == 0 ) ? (0) : ( ( i & ( (1<<log_mini_shift_reset_freq) - 1 ) ) << log_mini_shift_amount );
          if(direction){
            shift_amount = major_shift + mini_shift;
          }
          else{
            shift_amount = (max_data_offset - (major_shift + mini_shift))%max_data_offset;
          }

          //shift
          for(VAR_TYPE_16 j=0; j<V_BUG_SIZE*2; j++){
            #pragma HLS UNROLL
            outValRegArr[(j+shift_amount)%max_data_offset] = inValRegArr[j];
          }

          //write data to out stream
          for(VAR_TYPE_16 j=0; j<V_BUG_SIZE*2; j++){
            #pragma HLS UNROLL
            if(direction){
              fwd_outVal[j].write(outValRegArr[j]);
            }
            else{
              inv_outVal[j].write(outValRegArr[j]);
            }
          }
          i++;
        }
      }
    }
  }
}

void shuffler_buf(tapa::istreams<WORD, 2*V_BUG_SIZE>& fwd_inVal,
                  tapa::istreams<WORD, 2*V_BUG_SIZE>& inv_inVal,
                   tapa::ostreams<WORD, 2*V_BUG_SIZE>& fwd_outVal,
                   tapa::ostreams<WORD, 2*V_BUG_SIZE>& inv_outVal,
                    bool direction,
                   VAR_TYPE_16 BUG_id,
                   VAR_TYPE_16 iter);

void shuffler_out_shift(tapa::istreams<WORD, 2*V_BUG_SIZE>& fwd_inVal,
                        tapa::istreams<WORD, 2*V_BUG_SIZE>& inv_inVal_inter_0,
                        tapa::istreams<WORD, 2*V_BUG_SIZE>& inv_inVal_intra,
                        tapa::ostreams<WORD, 2*V_BUG_SIZE>& fwd_outVal_inter_0,
                        tapa::ostreams<WORD, 2*V_BUG_SIZE>& fwd_outVal_intra,
                        tapa::ostreams<WORD, 2*V_BUG_SIZE>& inv_outVal,
                        bool direction,
                        VAR_TYPE_16 iter,
                        VAR_TYPE_16 task_id);

void shuffler_out_shuff(
                  tapa::istream<WORD>& fwd_inVal_inter0_0,
                  tapa::istream<WORD>& fwd_inVal_inter0_1,
                  tapa::istream<WORD>& fwd_inVal_inter0_2,
                  tapa::istream<WORD>& fwd_inVal_inter0_3,
                  tapa::istream<WORD>& fwd_inVal_inter0_4,
                  tapa::istream<WORD>& fwd_inVal_inter0_5,
                  tapa::istream<WORD>& fwd_inVal_inter0_6,
                  tapa::istream<WORD>& fwd_inVal_inter0_7,
                  tapa::istream<WORD>& fwd_inVal_inter0_8,
                  tapa::istream<WORD>& fwd_inVal_inter0_9,
                  tapa::istream<WORD>& fwd_inVal_inter0_10,
                  tapa::istream<WORD>& fwd_inVal_inter0_11,
                  tapa::istream<WORD>& fwd_inVal_inter0_12,
                  tapa::istream<WORD>& fwd_inVal_inter0_13,
                  tapa::istream<WORD>& fwd_inVal_inter0_14,
                  tapa::istream<WORD>& fwd_inVal_inter0_15,
                  tapa::istreams<WORD, 2*V_BUG_SIZE>& fwd_inVal_intra,
                  tapa::istreams<WORD, 2*V_BUG_SIZE>& inv_inVal,
                  tapa::ostreams<WORD, 2*V_BUG_SIZE>& fwd_outVal,
                  tapa::ostreams<WORD, 2*V_BUG_SIZE>& inv_outVal_intra,
                  tapa::ostream<WORD>& inv_outVal_inter0_0,
                  tapa::ostream<WORD>& inv_outVal_inter0_1,
                  tapa::ostream<WORD>& inv_outVal_inter0_2,
                  tapa::ostream<WORD>& inv_outVal_inter0_3,
                  tapa::ostream<WORD>& inv_outVal_inter0_4,
                  tapa::ostream<WORD>& inv_outVal_inter0_5,
                  tapa::ostream<WORD>& inv_outVal_inter0_6,
                  tapa::ostream<WORD>& inv_outVal_inter0_7,
                  tapa::ostream<WORD>& inv_outVal_inter0_8,
                  tapa::ostream<WORD>& inv_outVal_inter0_9,
                  tapa::ostream<WORD>& inv_outVal_inter0_10,
                  tapa::ostream<WORD>& inv_outVal_inter0_11,
                  tapa::ostream<WORD>& inv_outVal_inter0_12,
                  tapa::ostream<WORD>& inv_outVal_inter0_13,
                  tapa::ostream<WORD>& inv_outVal_inter0_14,
                  tapa::ostream<WORD>& inv_outVal_inter0_15,
                    bool direction,
                   VAR_TYPE_16 iter
                    );


// /********************************************************************************************
// * TF buffer tasks *
// ********************************************************************************************/

void TFBuf_loadTF_wiFW(
           tapa::istream<DWORD>& loadTFStream,
           tapa::ostream<DWORD>& loadTFToNext,
           WORD TFArr[N/2],
           VAR_TYPE_16 tf_load_counter, 
           VAR_TYPE_16 this_buf_limit){

  #pragma HLS inline off

  //load
  TF_LOAD:for(VAR_TYPE_16 i=0; i<tf_load_counter;){
    #pragma HLS PIPELINE II = 1

    bool inpStreamNotEmpty = !loadTFStream.empty();

    if(inpStreamNotEmpty){
      if(i<this_buf_limit){
        DWORD val = loadTFStream.read();   
        TFArr[2*i] = (WORD)(val & ((((DWORD)1) << WORD_SIZE) - 1));
        TFArr[2*i+1] = (WORD)(val >> WORD_SIZE);
      }
      else{
        loadTFToNext.write((loadTFStream.read()));
      }
      i++;
    }
  }
}

void TFBuf_loadTF_woFW(
           tapa::istream<DWORD>& loadTFStream,
           WORD TFArr[N/2],
           VAR_TYPE_16 tf_load_counter){
  
  #pragma HLS inline off

  //load
  TF_LOAD:for(VAR_TYPE_16 i=0; i<tf_load_counter;){
    #pragma HLS PIPELINE II = 1

    bool inpStreamNotEmpty = !loadTFStream.empty();

    if(inpStreamNotEmpty){
      DWORD val = loadTFStream.read();   
      TFArr[2*i] = (WORD)(val & ((((DWORD)1) << WORD_SIZE) - 1));
      TFArr[2*i+1] = (WORD)(val >> WORD_SIZE);
      i++;
    }
  }
}

void TFBuf_comp(
           tapa::ostream<WORD>& tfToBu_0,
           tapa::ostream<WORD>& tfToBu_1,
           WORD TFArr[N/2],
           const VAR_TYPE_16 fwd_log_stageWiseTFs[( (logN+(H_BUG_SIZE-1))/H_BUG_SIZE )],
           const VAR_TYPE_16 inv_log_stageWiseTFs[( (logN+(H_BUG_SIZE-1))/H_BUG_SIZE )],
           const VAR_TYPE_16 fwd_stageWiseOffset[( (logN+(H_BUG_SIZE-1))/H_BUG_SIZE )],
           const VAR_TYPE_16 inv_stageWiseOffset[( (logN+(H_BUG_SIZE-1))/H_BUG_SIZE )],
           const VAR_TYPE_16 fwd_log_num_val_accessed_by_consec_BUs_arr[( (logN+(H_BUG_SIZE-1))/H_BUG_SIZE )],
           const VAR_TYPE_16 inv_log_num_val_accessed_by_consec_BUs_arr[( (logN+(H_BUG_SIZE-1))/H_BUG_SIZE )],
           VAR_TYPE_8 unit_id_0,
           VAR_TYPE_8 unit_id_1,
           VAR_TYPE_16 dataCount,
           VAR_TYPE_8 dataFlowIterLimit,
           bool direction){
  
  #pragma HLS inline off

  TF_DF_ITER:for(VAR_TYPE_8 dataFlowIter = 0; dataFlowIter<dataFlowIterLimit; dataFlowIter++){
      
    VAR_TYPE_16 log_num_TFs;
    VAR_TYPE_16 tf_offset;
    VAR_TYPE_16 log_num_val_accessed_by_consec_BUs;
    VAR_TYPE_16 log_countLimit;

    if(direction){
      log_num_TFs = fwd_log_stageWiseTFs[dataFlowIter];
      tf_offset = fwd_stageWiseOffset[dataFlowIter];
      log_num_val_accessed_by_consec_BUs = fwd_log_num_val_accessed_by_consec_BUs_arr[dataFlowIter];
      log_countLimit = log_num_TFs - log_num_val_accessed_by_consec_BUs;
    }
    else{
      log_num_TFs = inv_log_stageWiseTFs[dataFlowIter];
      tf_offset = inv_stageWiseOffset[dataFlowIter];
      log_num_val_accessed_by_consec_BUs = inv_log_num_val_accessed_by_consec_BUs_arr[dataFlowIter];
      log_countLimit = (logN - log_V_TOTAL_DATA) - (log_num_TFs - log_num_val_accessed_by_consec_BUs);
    }

    TF_COMP:for(VAR_TYPE_16 inpCount=0; inpCount<dataCount; inpCount++){
      #pragma HLS PIPELINE II = 1
      
      VAR_TYPE_16 tf_idx_0;
      VAR_TYPE_16 tf_idx_1;
      if(direction){
        tf_idx_0 = tf_offset + (inpCount & ( (1<<log_countLimit)-1 ) ) + ( (unit_id_0 & ( (1<<log_num_val_accessed_by_consec_BUs) - 1 )) << (log_countLimit) );
        tf_idx_1 = tf_offset + (inpCount & ( (1<<log_countLimit)-1 ) ) + ( (unit_id_1 & ( (1<<log_num_val_accessed_by_consec_BUs) - 1 )) << (log_countLimit) );
      }
      else{
        tf_idx_0 = tf_offset + ( (inpCount >> (log_countLimit) ) << log_num_val_accessed_by_consec_BUs ) + (unit_id_0 & ( (1<<log_num_val_accessed_by_consec_BUs) - 1 ));
        tf_idx_1 = tf_offset + ( (inpCount >> (log_countLimit) ) << log_num_val_accessed_by_consec_BUs ) + (unit_id_1 & ( (1<<log_num_val_accessed_by_consec_BUs) - 1 ));
      }
    
      WORD tfVal_0 = TFArr[tf_idx_0];
      WORD tfVal_1 = TFArr[tf_idx_1];

      tfToBu_0.write(tfVal_0);
      tfToBu_1.write(tfVal_1);

    }

  }
}

void TFBuf_wiFW_0(
           tapa::istream<DWORD>& loadTFStream,
           tapa::ostream<DWORD>& loadTFToNext,
           tapa::ostream<WORD>& tfToBu_0,
           tapa::ostream<WORD>& tfToBu_1,
           bool direction,
           VAR_TYPE_8 BUG_id,
           VAR_TYPE_8 TFGen_offset, 
           VAR_TYPE_8 unit_id,
           VAR_TYPE_16 iter);    

void TFBuf_wiFW_1(
           tapa::istream<DWORD>& loadTFStream,
           tapa::ostream<DWORD>& loadTFToNext,
           tapa::ostream<WORD>& tfToBu_0,
           tapa::ostream<WORD>& tfToBu_1,
           bool direction,
           VAR_TYPE_8 BUG_id,
           VAR_TYPE_8 TFGen_offset, 
           VAR_TYPE_8 unit_id,
           VAR_TYPE_16 iter);    

void TFBuf_wiFW_2(
           tapa::istream<DWORD>& loadTFStream,
           tapa::ostream<DWORD>& loadTFToNext,
           tapa::ostream<WORD>& tfToBu_0,
           tapa::ostream<WORD>& tfToBu_1,
           bool direction,
           VAR_TYPE_8 BUG_id,
           VAR_TYPE_8 TFGen_offset, 
           VAR_TYPE_8 unit_id,
           VAR_TYPE_16 iter);    

void TFBuf_woFW_3(
           tapa::istream<DWORD>& loadTFStream,
           tapa::ostream<WORD>& tfToBu_0,
           tapa::ostream<WORD>& tfToBu_1,
           bool direction,
           VAR_TYPE_8 BUG_id,
           VAR_TYPE_8 TFGen_offset, 
           VAR_TYPE_8 unit_id,
           VAR_TYPE_16 iter);


// /********************************************************************************************
// * Input/Output selector tasks *
// ********************************************************************************************/

void  input_selector_single(
                    tapa::istream<WORD>& fwd_valFromLoad_0,
                    tapa::istream<WORD>& fwd_valFromShuf,
                    tapa::istream<WORD>& inv_valFromBU,
                    tapa::ostream<WORD>& fwd_valToBU,
                    tapa::ostream<WORD>& inv_valToShuf,
                    tapa::ostream<WORD>& inv_valToStore_0,
                    bool direction,
                    VAR_TYPE_16 iter);
void output_slector_single(
                  tapa::istream<WORD>& fwd_valFromBU,
                  tapa::istream<WORD>& inv_valFromLoad_0,
                  tapa::istream<WORD>& inv_valFromShuf,
                  tapa::ostream<WORD>& fwd_valToShuf,
                  tapa::ostream<WORD>& fwd_valToStore_0,
                  tapa::ostream<WORD>& inv_valToBU,
                  bool direction,
                  VAR_TYPE_16 iter);

// /********************************************************************************************
// * Dual interface FIFO tasks *
// ********************************************************************************************/

void dual_interface_FIFO_receive(
                  tapa::istream<WORD>& fwd_inVal,
                  tapa::istream<WORD>& inv_inVal,
                  tapa::ostream<WORD>& common_outVal,
                  bool direction);

void dual_interface_FIFO_send(
                  tapa::istream<WORD>& common_inVal,
                  tapa::ostream<WORD>& fwd_outVal,
                  tapa::ostream<WORD>& inv_outVal,
                  bool direction);

// /********************************************************************************************
// * Top kernel *
// ********************************************************************************************/

void NTT_kernel(
    uint64_t  polyVectorInOut_G0,
    uint64_t  TFArr_G0,
    uint64_t  TFArr_G1,
    uint64_t  polyVectorInOut_G1,
    WORD q0,
    WORD twoInverse0,
    WORD_PLUS3 factor0,
    bool direction,
    VAR_TYPE_16 iter
    )
;
