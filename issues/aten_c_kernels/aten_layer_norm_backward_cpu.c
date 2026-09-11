#define B 16
#define D 64
void aten_layer_norm_backward_cpu(float grad[B][D],float x[B][D],float mean[B],float rstd[B],float weight[D],float dx[B][D],float dw[D],float db[D]){for(int d=0;d<D;++d){dw[d]=0;db[d]=0;}for(int b=0;b<B;++b){float s1=0,s2=0;for(int d=0;d<D;++d){float w=grad[b][d]*weight[d];s1+=w;s2+=w*(x[b][d]-mean[b]);dw[d]+=grad[b][d]*(x[b][d]-mean[b])*rstd[b];db[d]+=grad[b][d];}for(int d=0;d<D;++d){float w=grad[b][d]*weight[d];dx[b][d]=rstd[b]/D*(D*w-s1-(x[b][d]-mean[b])*rstd[b]*rstd[b]*s2);}}}
