/* FP32 contiguous specialization of pinned hardswish_kernel.
 * Scalar operation order and parameter roles are retained. See manifest. */
#include <math.h>
#define PG_MIN(A,B) ((B)<(A)?(B):(A))
#define PG_MAX(A,B) ((A)<(B)?(B):(A))
#define PG_CLAMP(A,L,H) PG_MIN(PG_MAX(A,L),H)
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC required"
#endif
void NETLIB_SPEC(int n,float x[],float output[]){
const float zero=0.0f;
const float three=3.0f;
const float six=6.0f;
for(int i=0;i<n;++i){

        output[i] = x[i] * PG_CLAMP(x[i] + three, zero, six) / six;
      
}
}
