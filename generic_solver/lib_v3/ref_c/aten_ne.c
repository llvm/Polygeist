/* FP32-output scalar specialization of pinned ne_kernel.
 * Preserve comparison/cast semantics; TensorIterator becomes a contiguous loop. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC required"
#endif
void NETLIB_SPEC(int n,float a[],float b[],float output[]){
for(int i=0;i<n;++i){
            output[i] = (float)(a[i] != b[i]);
          }
}
