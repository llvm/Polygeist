/* FP32-output scalar specialization of pinned logical_not_kernel.
 * Preserve comparison/cast semantics; TensorIterator becomes a contiguous loop. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC required"
#endif
void NETLIB_SPEC(int n,float a[],float output[]){
for(int i=0;i<n;++i){ output[i] = (float)(!a[i]); }
}
