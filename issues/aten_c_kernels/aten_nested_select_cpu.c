#define B 8
#define N 64
void aten_nested_select_cpu(float x[B][N],int idx[B],float out[B]){for(int b=0;b<B;++b)out[b]=x[b][idx[b]];}
