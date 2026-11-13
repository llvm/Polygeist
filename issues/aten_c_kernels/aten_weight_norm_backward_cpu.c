#define B 4
#define C 8
#define S 32
void aten_weight_norm_backward_cpu(float grad[C][S],float v[C][S],float g[C],float norms[C],float dv[C][S],float dg[C]){for(int c=0;c<C;++c){float dot=0;for(int i=0;i<S;++i)dot+=grad[c][i]*v[c][i];dg[c]=dot/norms[c];for(int i=0;i<S;++i)dv[c][i]=g[c]/norms[c]*(grad[c][i]-v[c][i]*dot/(norms[c]*norms[c]));}}
