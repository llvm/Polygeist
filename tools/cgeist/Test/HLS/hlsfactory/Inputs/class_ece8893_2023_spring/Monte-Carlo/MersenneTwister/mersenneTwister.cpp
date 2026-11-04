
#include "mersenneTwister.hpp"
#include <chrono>
#include <ctime>
#include <iostream>
// #include <ap_fixed.h>

/* initializing the array with a NONZERO seed */
void sgenrand(unsigned long seed)
{
    //Seed Generation
    /* setting initial seeds to mt[N] using         */
    /* the generator Line 25 of Table 1 in          */
    /* [KNUTH 1981, The Art of Computer Programming */
    /*    Vol. 2 (2nd Ed.), pp102]                  */
    mt[0]= seed & 0xffffffff;
    
    // pragma
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
            

        // pragma
        for (kk=0;kk<N-M;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1];
        }

        // pragma
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



// /* this main() outputs first 1000 generated numbers  */
void randomGen(
   int seed[1],
   double out[NUM]
)
{   
    sgenrand(seed[0]); /* any nonzero integer can be used as a seed */     //to set the initial seed 
    
    //#pragma HLS pipeline
    for (int j=0; j<NUM; j++) {
        double value = genrand();
        out[j] = value;
        // printf("%5f ", value);
        // if (j%8==7) printf("\n");
    }
    
}