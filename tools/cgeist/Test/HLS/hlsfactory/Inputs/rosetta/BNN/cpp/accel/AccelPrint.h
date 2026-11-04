#ifndef ACCEL_PRINT_H
#define ACCEL_PRINT_H

#include "Accel.h"
#include <stdio.h>

//------------------------------------------------------------------------
// For an array of ap_int's, sets/gets the bit at bit_idx
// calculated from the start of the array to val
//------------------------------------------------------------------------
template<typename T>
inline void set_bit(T array[], unsigned bit_idx, Bit val) {
  unsigned W = array[0].length();
  Address idx = bit_idx / W;
  Address offset = bit_idx % W;
  array[idx][offset] = val;
}

template<typename T>
inline Bit get_bit(T array[], unsigned bit_idx) {
  unsigned W = array[0].length();
  Address idx = bit_idx / W;
  Address offset = bit_idx % W;
  Bit result = array[idx][offset];
  return result;
}

//------------------------------------------------------------------------
// Printing matrices and bit arrays
//------------------------------------------------------------------------
template<typename T>
void print_mat(T in[], unsigned S, unsigned R, unsigned C) {
  R = (R >= S) ? S : R;
  C = (C >= S) ? S : C;
  for (unsigned r = 0; r < R; ++r) {
    for (unsigned c = 0; c < C; ++c)
      std::cout << std::setw(4) << in[r*S+c] << " ";
    printf ("\n");
  }
}

template<typename T>
void print_mat3d(T in[], unsigned M, unsigned num, unsigned S, unsigned R, unsigned C) {
  for (unsigned m = M; m < M+num; ++m) {
    print_mat(in+m*S*S, S, R, C);
    printf ("--%u--\n", m+1);
  }
}

template<typename T>
void print_bits(T in[], unsigned bit_offset, unsigned S, unsigned R, unsigned C) {
  R = (R >= S) ? S : R;
  C = (C >= S) ? S : C;
  for (unsigned r = 0; r < R; ++r) {
    for (unsigned c = 0; c < C; ++c)
      std::cout << std::setw(2) << get_bit(in, bit_offset+r*S+c) << " ";
    printf ("\n");
  }
}

template<typename T>
void print_bits3d(T in[], unsigned M, unsigned num, unsigned S, unsigned R, unsigned C) {
  for (unsigned m = M; m < M+num; ++m) {
    print_bits(in, m*S*S, S, R, C);
    printf ("--%u--\n", m+1);
  }
}

void print_params3d(Word in[], unsigned M, unsigned num);
void print_wt_word(const Word& in);
void print_wt(const WtType& in);

#endif
