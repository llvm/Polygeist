/* Fixed-shape ATen max_pool3d_backward. */
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
#ifndef I1
#define I1 7
#endif
#define O1 (I1/2)
#ifndef I2
#define I2 8
#endif
#define O2 (I2/2)
void aten_max_pool3d_backward_cpu(float grad_output[B*C*O0*O1*O2], int indices[B*C*O0*O1*O2], float grad_input[B*C*I0*I1*I2]){
  for(int p=0;p<B*C*I0*I1*I2;++p)grad_input[p]=0.0f;
  for(int n=0;n<B;++n){
    for(int c=0;c<C;++c){
    for(int o0=0;o0<O0;++o0){
      for(int o1=0;o1<O1;++o1){
        for(int o2=0;o2<O2;++o2){
          int s0=o0*2;
          int e0=s0+2;
          int s1=o1*2;
          int e1=s1+2;
          int s2=o2*2;
          int e2=s2+2;
          int flat=indices[((((((((n)*C+c))*O0+o0))*O1+o1))*O2+o2)];
          grad_input[(n*C+c)*I0*I1*I2+flat]+=grad_output[((((((((n)*C+c))*O0+o0))*O1+o1))*O2+o2)];
        }
      }
    }
  }
}
}
