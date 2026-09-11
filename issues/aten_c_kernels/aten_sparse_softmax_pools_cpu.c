#define R 64
void aten_sparse_softmax_pools_cpu(int off[R+1],int size[R]){for(int r=0;r<R;++r)size[r]=off[r+1]-off[r];}
