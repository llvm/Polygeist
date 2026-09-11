#define B 32
#define D 64
#define E 1024
void aten_embedding_bag_backward_max_cpu(float grad[B][D],int maxidx[B][D],float out[E][D]){for(int p=0;p<E*D;++p)((float*)out)[p]=0;for(int b=0;b<B;++b)for(int d=0;d<D;++d)out[maxidx[b][d]][d]+=grad[b][d];}
