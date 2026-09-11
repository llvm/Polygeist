#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
void aten_and_reduce_cpu(int x[R][K], int out[R]) {
  for(int r=0;r<R;++r){int v=1;for(int k=0;k<K;++k)v=v&&(x[r][k]!=0);out[r]=v;}
}
