#ifndef R
#define R 16
#endif
#ifndef K
#define K 64
#endif
#ifndef TOP
#define TOP 8
#endif
void aten_sort_cpu(float input[R][K],float values[R][K],int indices[R][K]){for(int r=0;r<R;++r){for(int k=0;k<K;++k){values[r][k]=input[r][k];indices[r][k]=k;}for(int k=1;k<K;++k){float v=values[r][k];int idx=indices[r][k],j=k-1;while(j>=0&&values[r][j]<v){values[r][j+1]=values[r][j];indices[r][j+1]=indices[r][j];--j;}values[r][j+1]=v;indices[r][j+1]=idx;}}}
