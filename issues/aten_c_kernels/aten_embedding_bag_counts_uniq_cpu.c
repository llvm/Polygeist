#define N 512
void aten_embedding_bag_counts_uniq_cpu(int index[N],int out[N]){for(int i=0;i<N;++i){int n=0;for(int j=0;j<N;++j)n+=index[j]==index[i];out[i]=n;}}
