/* Fixed-shape ATen adaptive_max_pool2d. */
#ifndef B
#define B 1
#endif
#ifndef C
#define C 2
#endif
#ifndef I0
#define I0 6
#endif
#ifndef O0
#define O0 3
#endif
#ifndef I1
#define I1 7
#endif
#ifndef O1
#define O1 3
#endif
void aten_adaptive_max_pool2d_cpu(float input[B*C*I0*I1], float output[B*C*O0*O1], int indices[B*C*O0*O1]){
  for(int n=0;n<B;++n){
    for(int c=0;c<C;++c){
    for(int o0=0;o0<O0;++o0){
      for(int o1=0;o1<O1;++o1){
        int s0=o0*I0/O0;
        int e0=((o0+1)*I0+O0-1)/O0;
        int s1=o1*I1/O1;
        int e1=((o1+1)*I1+O1-1)/O1;
        float value=-3.402823466e38f;int best=0;
        for(int i0=s0;i0<e0;++i0){
          for(int i1=s1;i1<e1;++i1){
            if(input[((((((n)*C+c))*I0+i0))*I1+i1)]>value){value=input[((((((n)*C+c))*I0+i0))*I1+i1)];best=((i0)*I1+i1);}
          }
        }
        output[((((((n)*C+c))*O0+o0))*O1+o1)]=value;indices[((((((n)*C+c))*O0+o0))*O1+o1)]=best;
      }
    }
  }
}
}
