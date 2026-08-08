#define N 255
void aten_ifftshift_cpu(float x[N],float out[N]){for(int i=0;i<N;++i)out[i]=x[(i+(N+1)/2)%N];}
