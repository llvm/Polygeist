#define N 512
void aten_coalesce_sparse_cpu(int idx[N],float val[N],int out_idx[N],float out_val[N],int count[1]){int p=0;for(int i=0;i<N;++i){if(i&&idx[i]==out_idx[p-1])out_val[p-1]+=val[i];else{out_idx[p]=idx[i];out_val[p++]=val[i];}}count[0]=p;}
