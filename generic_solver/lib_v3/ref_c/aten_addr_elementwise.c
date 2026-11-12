/* Source-derived FP32 specialization. See manifest for exact source and adaptation. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC required"
#endif
extern __attribute__((const)) float truncf(float);
void NETLIB_SPEC(int n,float self_val[],float vec1_val[],float vec2_val[],float beta_val,float alpha_val,float output[]){
for(int i=0;i<n;++i){if(beta_val == 0.0f){
            output[i] = alpha_val * vec1_val[i] * vec2_val[i];
          }else{
            output[i] = beta_val * self_val[i] + alpha_val * vec1_val[i] * vec2_val[i];
          }}
}
