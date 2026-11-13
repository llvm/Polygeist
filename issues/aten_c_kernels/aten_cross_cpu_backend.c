#define V 256
void aten_cross_cpu_backend(float a[V][3],float b[V][3],float out[V][3]){for(int i=0;i<V;++i){out[i][0]=a[i][1]*b[i][2]-a[i][2]*b[i][1];out[i][1]=a[i][2]*b[i][0]-a[i][0]*b[i][2];out[i][2]=a[i][0]*b[i][1]-a[i][1]*b[i][0];}}
