#define B 8
#define C 32
#define S 16
void aten_multinomial_with_replacement_cpu(float probability[B][C],float uniform[B][S],int out[B][S]){for(int b=0;b<B;++b){float total=0,cdf[C];for(int c=0;c<C;++c){total+=probability[b][c];cdf[c]=total;}for(int s=0;s<S;++s){float u=uniform[b][s]*total;int c=0;while(c<C-1&&cdf[c]<u)++c;out[b][s]=c;}}}
