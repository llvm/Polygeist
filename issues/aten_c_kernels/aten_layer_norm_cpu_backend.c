#define B 16
#define D 64
extern float sqrtf(float);void aten_layer_norm_cpu_backend(float x[B][D],float weight[D],float bias[D],float eps,float out[B][D],float mean[B],float rstd[B]){for(int b=0;b<B;++b){float s=0;for(int d=0;d<D;++d)s+=x[b][d];mean[b]=s/D;float q=0;for(int d=0;d<D;++d){float z=x[b][d]-mean[b];q+=z*z;}rstd[b]=1.0f/sqrtf(q/D+eps);for(int d=0;d<D;++d)out[b][d]=(x[b][d]-mean[b])*rstd[b]*weight[d]+bias[d];}}
