/* Scalar float specialization of pinned ATen sign_kernel. The pinned
 * is_negative<float> helper evaluates a < float(0); its definition is hashed
 * in the manifest. Preserve integer subtraction before the FP32 conversion. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC required"
#endif
void NETLIB_SPEC(int n,float a[],float output[]){
 for(int i=0;i<n;++i)output[i]=(0<a[i])-(a[i]<0.0f);
}
