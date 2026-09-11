/* Fixed-shape ATen max_pool1d. */
#ifndef B
#define B 1
#endif
#ifndef C
#define C 2
#endif
#ifndef I0
#define I0 6
#endif
#define O0 (I0/2)
void aten_max_pool1d_cpu(float input[B*C*I0], float output[B*C*O0], int indices[B*C*O0]){
  for(int n=0;n<B;++n){
    for(int c=0;c<C;++c){
    for(int o0=0;o0<O0;++o0){
      int s0=o0*2;
      int e0=s0+2;
      float value=-3.402823466e38f;int best=0;
      for(int i0=s0;i0<e0;++i0){
        if(input[((((n)*C+c))*I0+i0)]>value){value=input[((((n)*C+c))*I0+i0)];best=i0;}
      }
      output[((((n)*C+c))*O0+o0)]=value;indices[((((n)*C+c))*O0+o0)]=best;
    }
  }
}
}
