/* FP32 contiguous specialization of pinned scalar_gelu.
 * Scalar operation order and parameter roles are retained. See manifest. */
#include <math.h>
#define PG_MIN(A,B) ((B)<(A)?(B):(A))
#define PG_MAX(A,B) ((A)<(B)?(B):(A))
#define PG_CLAMP(A,L,H) PG_MIN(PG_MAX(A,L),H)
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC required"
#endif
#define kGeluBeta (M_SQRT2 * M_2_SQRTPI * 0.5)
#define kGeluKappa (0.044715)
void NETLIB_SPEC(int n,float x[],float output[]){
for(int i=0;i<n;++i){

  const float kAlpha = ((float)(M_SQRT1_2));
  output[i] = (x[i]) * ((float)(0.5)) * (((float)(1)) + erff((x[i]) * kAlpha));

}
}
