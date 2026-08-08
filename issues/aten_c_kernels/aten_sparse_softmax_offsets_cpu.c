#define N 512
#define R 64
void aten_sparse_softmax_offsets_cpu(int row[N],int out[R+1]){int p=0;for(int r=0;r<=R;++r){while(p<N&&row[p]<r)++p;out[r]=p;}}
