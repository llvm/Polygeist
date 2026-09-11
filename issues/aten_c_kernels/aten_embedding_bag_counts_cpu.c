#define N 512
#define E 1024
void aten_embedding_bag_counts_cpu(int index[N],int out[E]){for(int e=0;e<E;++e)out[e]=0;for(int i=0;i<N;++i)out[index[i]]++;}
