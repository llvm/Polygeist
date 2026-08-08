#define B 8
#define P 80
void aten_nested_to_padded_cpu(float x[B][P],int len[B],float pad,float out[B][P]){for(int b=0;b<B;++b)for(int i=0;i<P;++i)out[b][i]=i<len[b]?x[b][i]:pad;}
