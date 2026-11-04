#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstdint>
#include <stdio.h>
#include <ap_int.h>
#include <hls_vector.h>

#ifdef BIT16
using spint = uint16_t;
using udpint = uint32_t;
using dpint = uint32_t;	

constexpr spint Wordlength = 16;
constexpr spint Radix = 13;

#elif BIT32
using spint = uint32_t;
using udpint = uint64_t;
using dpint = uint64_t;

constexpr spint Wordlength = 32;
constexpr spint Radix = 29;

#else
// using spint = uint64_t;
// using udpint = __uint128_t;
// using dpint = __uint128_t;

constexpr uint32_t Wordlength = 64;
constexpr uint32_t Radix = 13;
// using spint = uint32_t;
// using udpint = uint64_t;
// using dpint = uint64_t;

// constexpr uint32_t Wordlength = 32;
// constexpr uint32_t Radix = 29;

// constexpr uint32_t Wordlength = 16;
// constexpr uint32_t Radix = 13;
#endif

// Only way to get number of bits at compile time in C++14
static constexpr unsigned numberOfBits(unsigned x)
{
    return x < 2 ? x : 1 + numberOfBits(x >> 1);
}

// Only way to use ceiling division at compile time in C++14
template<typename T>
static constexpr T ceil(T num, T den) {
    return (num + den - 1) / den;
}

constexpr uint32_t Nbytes = 32;
constexpr uint32_t Nbits = 255;
constexpr uint32_t Nlimbs = ceil(Nbits, Radix);
constexpr uint32_t OverflowBits = Radix * Nlimbs - Nbits;
constexpr uint32_t FinalLimbRadix = Radix - OverflowBits; 
constexpr bool LastLimbOverflows = Nbits < Radix * Nlimbs;

constexpr uint32_t CarryIndex = Nlimbs < 2 ? 0 : 1;


constexpr uint32_t CongruentFactor = 0x13; // 19 in decimal
constexpr uint32_t ReductionFactor = CongruentFactor * (1 << OverflowBits);

// Calculates the number of bits to avoid overflow 
constexpr uint32_t TermWidth = 2 * Radix + numberOfBits(ReductionFactor) + numberOfBits(Nlimbs - 1) + 1;

using spint = ap_uint<Radix>; 
using dpint = ap_uint<TermWidth>; 
using udpint = ap_uint<TermWidth>; 
using spint_vec = hls::vector<spint, Nlimbs>;
using spint_red_vec = hls::vector<ap_uint<Radix + numberOfBits(ReductionFactor) + 1>, Nlimbs>; 

void mod_mul(spint_vec& x, spint_vec& y, spint_vec& z);
