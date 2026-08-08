#define N 512
#define R 64
void aten_convert_csr_to_coo_cpu(int ptr[R+1],int col[N],int row[N]){for(int r=0;r<R;++r)for(int p=ptr[r];p<ptr[r+1];++p)row[p]=r;}
