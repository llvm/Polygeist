#define B 32
#define L 16
#define E 1024
#define D 64
void aten_embedding_bag_max_cpu(float table[E][D],int index[B][L],float out[B][D]){for(int b=0;b<B;++b)for(int d=0;d<D;++d){float v=table[index[b][0]][d];for(int l=1;l<L;++l)if(table[index[b][l]][d]>v)v=table[index[b][l]][d];out[b][d]=v;}}
