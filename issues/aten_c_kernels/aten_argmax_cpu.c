#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
void aten_argmax_cpu(float x[R][K], int out[R]) {
  for(int r=0;r<R;++r){float v=x[r][0];int best=0;for(int k=1;k<K;++k)if(x[r][k]>v){v=x[r][k];best=k;}out[r]=best;}
}
