#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
#ifndef S
#define S 128
#endif
void aten_put_cpu(float out[S],int index[R],float source[R],int accumulate){for(int r=0;r<R;++r){if(accumulate)out[index[r]]+=source[r];else out[index[r]]=source[r];}}
