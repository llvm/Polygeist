#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
#ifndef S
#define S 128
#endif
void aten_take_cpu(float input[S],int index[R],float out[R]){for(int r=0;r<R;++r)out[r]=input[index[r]];}
