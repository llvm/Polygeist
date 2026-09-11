#define R 16
#define C 32
void aten_sparse_full_coo_indices_cpu(int row[R*C],int col[R*C]){int p=0;for(int r=0;r<R;++r)for(int c=0;c<C;++c){row[p]=r;col[p++]=c;}}
