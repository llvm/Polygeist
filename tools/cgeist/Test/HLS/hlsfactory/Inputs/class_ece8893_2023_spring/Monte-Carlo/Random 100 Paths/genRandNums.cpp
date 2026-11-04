
#include "./mersenneTwister.hpp"
#include <chrono>
#include <ctime>
#include <iostream>


/* initializing the array with a NONZERO seed */
void sgenrand(unsigned long seed)
{
    //Seed Generation
    /* setting initial seeds to mt[N] using         */
    /* the generator Line 25 of Table 1 in          */
    /* [KNUTH 1981, The Art of Computer Programming */
    /*    Vol. 2 (2nd Ed.), pp102]                  */
    mt[0]= seed & 0xffffffff;
    
    for (mti=1; mti<N; mti++)       //can basically be pipelined -- cuz dependance 
        mt[mti] = (69069 * mt[mti-1]) & 0xffffffff;
}

double /* generating reals */
/* unsigned long */ /* for integer generation */
genrand()
{
    unsigned long y;
    static unsigned long mag01[2]={0x0, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        int kk;

        if (mti == N+1)   /* if sgenrand() has not been called, */
        {
            sgenrand(4357); /* a default initial seed is used   */
        }
            

        for (kk=0;kk<N-M;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        for (;kk<N-1;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
        mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1];

        mti = 0;
    }
  
    y = mt[mti++];
    y ^= TEMPERING_SHIFT_U(y);
    y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
    y ^= TEMPERING_SHIFT_L(y);

    return ((double)y / (unsigned long)0xffffffff ); /* reals */
    /* return y; */ /* for integer generation */
}

void genRandNums(int seed[1], double randNums[100][100]) {
    #pragma HLS INTERFACE m_axi depth=1  port=seed  bundle=sd
    #pragma HLS INTERFACE m_axi depth=1  port=randNums  bundle=nums
    double buff[100][100];

    sgenrand(seed[0]);

    for(int i = 0; i < 100;i++) {
        for(int j = 0; j < 100; j++) {
            buff[i][j]  = sqrt(-2.0 * log(genrand())) * cos(2.0 * 3.14159265358979323846 * genrand()); 
        }
    }

    for(int i = 0; i < 100;i++) {
        for(int j = 0; j < 100; j++) {
            randNums[i][j] = buff[i][j];
        }
    }
}