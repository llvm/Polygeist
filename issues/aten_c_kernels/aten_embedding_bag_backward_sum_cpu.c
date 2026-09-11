#define B 32
#define L 16
#define E 1024
#define D 64
void aten_embedding_bag_backward_sum_cpu(float grad[B][D],int index[B][L],float out[E][D]){for(int p=0;p<E*D;++p)((float*)out)[p]=0;for(int b=0;b<B;++b)for(int l=0;l<L;++l)for(int d=0;d<D;++d)out[index[b][l]][d]+=grad[b][d];}
