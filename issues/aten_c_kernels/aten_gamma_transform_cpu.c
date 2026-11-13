#define N 1024
extern float sqrtf(float);void aten_gamma_transform_cpu(float alpha[N],float normal[N],float uniform[N],float out[N]){for(int i=0;i<N;++i){float d=alpha[i]-.3333333f,c=1/sqrtf(9*d),v=1+c*normal[i];out[i]=d*v*v*v;}}
