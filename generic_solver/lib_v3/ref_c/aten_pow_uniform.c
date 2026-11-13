/* Uniform FP32 exponent specialization of the pinned Vectorized<float>
 * broadcast constructor and pow lane method. The constructor fills every
 * exponent lane; the pow method uses std::pow(float,float) in each lane.
 * Lane traversal becomes the same parameterized C loop as aten_pow.c. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC required"
#endif
extern __attribute__((const)) float powf(float,float);
void NETLIB_SPEC(int n,float a[],float exponent,float output[]){
 for(int i=0;i<n;++i)output[i]=powf(a[i],exponent);
}
