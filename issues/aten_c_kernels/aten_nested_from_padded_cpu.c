#define B 8
#define P 80
void aten_nested_from_padded_cpu(float x[B][P],int len[B],float out[B][P]){for(int b=0;b<B;++b)for(int i=0;i<P;++i)out[b][i]=i<len[b]?x[b][i]:0;}
