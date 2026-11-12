/* Source-derived FP32 specialization. See manifest for exact source and adaptation. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC required"
#endif
extern __attribute__((const)) float truncf(float);
void NETLIB_SPEC(int n,float a[],float output[]){
for(int i=0;i<n;++i){
  output[i] = truncf(a[i]);
}
}
