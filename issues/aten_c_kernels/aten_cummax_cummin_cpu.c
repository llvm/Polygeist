#define R 16
#define N 64
void aten_cummax_cummin_cpu(float x[R][N],int is_max,float out[R][N],int index[R][N]){for(int r=0;r<R;++r){float v=x[r][0];int q=0;for(int i=0;i<N;++i){if((is_max&&x[r][i]>=v)||(!is_max&&x[r][i]<=v)){v=x[r][i];q=i;}out[r][i]=v;index[r][i]=q;}}}
