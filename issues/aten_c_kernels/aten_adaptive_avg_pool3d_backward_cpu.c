/* Fixed-shape ATen adaptive_avg_pool3d_backward. */
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
void aten_adaptive_avg_pool3d_backward_cpu(float grad_output[B*C*O0*O1*O2], float grad_input[B*C*I0*I1*I2]){
  for(int p=0;p<B*C*I0*I1*I2;++p)grad_input[p]=0.0f;
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
          for(int i0=s0;i0<e0;++i0){
            for(int i1=s1;i1<e1;++i1){
              for(int i2=s2;i2<e2;++i2){
                grad_input[((((((((n)*C+c))*I0+i0))*I1+i1))*I2+i2)]+=grad_output[((((((((n)*C+c))*O0+o0))*O1+o1))*O2+o2)]/(float)((e0-s0)*(e1-s1)*(e2-s2));
              }
            }
          }

        }
      }
    }
  }
}
}
