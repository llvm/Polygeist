#define N 1024
void aten_joint_scaling_cpu(float a[N],float b[N],float out[1]){float ma=0,mb=0;for(int i=0;i<N;++i){float x=a[i]<0?-a[i]:a[i],y=b[i]<0?-b[i]:b[i];if(x>ma)ma=x;if(y>mb)mb=y;}out[0]=ma*mb;}
