/* Fixed-shape ATen adaptive_max_pool3d. */
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
#ifndef I2
#define I2 8
#endif
#ifndef O2
#define O2 3
#endif
void aten_adaptive_max_pool3d_cpu(float input[B*C*I0*I1*I2], float output[B*C*O0*O1*O2], int indices[B*C*O0*O1*O2]){
  for(int n=0;n<B;++n){
    for(int c=0;c<C;++c){
    for(int o0=0;o0<O0;++o0){
      for(int o1=0;o1<O1;++o1){
        for(int o2=0;o2<O2;++o2){
          int s0=o0*I0/O0;
          int e0=((o0+1)*I0+O0-1)/O0;
          int s1=o1*I1/O1;
          int e1=((o1+1)*I1+O1-1)/O1;
          int s2=o2*I2/O2;
          int e2=((o2+1)*I2+O2-1)/O2;
          float value=-3.402823466e38f;int best=0;
          for(int i0=s0;i0<e0;++i0){
            for(int i1=s1;i1<e1;++i1){
              for(int i2=s2;i2<e2;++i2){
                if(input[((((((((n)*C+c))*I0+i0))*I1+i1))*I2+i2)]>value){value=input[((((((((n)*C+c))*I0+i0))*I1+i1))*I2+i2)];best=((((i0)*I1+i1))*I2+i2);}
              }
            }
          }
          output[((((((((n)*C+c))*O0+o0))*O1+o1))*O2+o2)]=value;indices[((((((((n)*C+c))*O0+o0))*O1+o1))*O2+o2)]=best;
        }
      }
    }
  }
}
}
