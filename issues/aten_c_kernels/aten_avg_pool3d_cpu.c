/* Fixed-shape ATen avg_pool3d. */
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
void aten_avg_pool3d_cpu(float input[B*C*I0*I1*I2], float output[B*C*O0*O1*O2]){
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
          float value=0.0f;int count=0;
          for(int i0=s0;i0<e0;++i0){
            for(int i1=s1;i1<e1;++i1){
              for(int i2=s2;i2<e2;++i2){
                value+=input[((((((((n)*C+c))*I0+i0))*I1+i1))*I2+i2)];++count;
              }
            }
          }
          output[((((((((n)*C+c))*O0+o0))*O1+o1))*O2+o2)]=value/(float)count;
        }
      }
    }
  }
}
}
