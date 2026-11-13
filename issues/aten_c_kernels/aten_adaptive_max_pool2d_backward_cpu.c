/* Fixed-shape ATen adaptive_max_pool2d_backward. */
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
void aten_adaptive_max_pool2d_backward_cpu(float grad_output[B*C*O0*O1], int indices[B*C*O0*O1], float grad_input[B*C*I0*I1]){
  for(int p=0;p<B*C*I0*I1;++p)grad_input[p]=0.0f;
  for(int n=0;n<B;++n){
    for(int c=0;c<C;++c){
    for(int o0=0;o0<O0;++o0){
      for(int o1=0;o1<O1;++o1){
        int s0=o0*I0/O0;
        int e0=((o0+1)*I0+O0-1)/O0;
        int s1=o1*I1/O1;
        int e1=((o1+1)*I1+O1-1)/O1;
        int flat=indices[((((((n)*C+c))*O0+o0))*O1+o1)];
        grad_input[(n*C+c)*I0*I1+flat]+=grad_output[((((((n)*C+c))*O0+o0))*O1+o1)];
      }
    }
  }
}
}
