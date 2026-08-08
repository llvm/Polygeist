#define B 32
#define L 16
#define E 1024
#define D 64
void aten_embedding_bag_per_sample_backward_cpu(float grad[B][D],float table[E][D],int index[B][L],float out[B][L]){for(int b=0;b<B;++b)for(int l=0;l<L;++l){float v=0;for(int d=0;d<D;++d)v+=grad[b][d]*table[index[b][l]][d];out[b][l]=v;}}
