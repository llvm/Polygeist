/* FP32 contiguous specialization of pinned clamp_min_scalar_kernel_impl.
 * Scalar operation order and parameter roles are retained. See manifest. */
#include <math.h>
#define PG_MIN(A,B) ((B)<(A)?(B):(A))
#define PG_MAX(A,B) ((A)<(B)?(B):(A))
#define PG_CLAMP(A,L,H) PG_MIN(PG_MAX(A,L),H)
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC required"
#endif
void NETLIB_SPEC(int n,float a[],float output[],float min){
for(int i=0;i<n;++i){

          output[i] = PG_MAX(a[i], min);
        
}
}
