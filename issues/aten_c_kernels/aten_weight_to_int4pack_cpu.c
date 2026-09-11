#define M 32
#define K 64
#define N 48
void aten_weight_to_int4pack_cpu(unsigned char weight[N][K],unsigned char packed[N][K/2]){for(int n=0;n<N;++n)for(int k=0;k<K;k+=2)packed[n][k/2]=(weight[n][k]&15)|((weight[n][k+1]&15)<<4);}
