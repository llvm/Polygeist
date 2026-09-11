#define B 4
#define C 8
#define S 32
extern float sqrtf(float);void aten_weight_norm_cpu(float v[C][S],float g[C],float out[C][S],float norms[C]){for(int c=0;c<C;++c){float s=0;for(int i=0;i<S;++i)s+=v[c][i]*v[c][i];norms[c]=sqrtf(s);for(int i=0;i<S;++i)out[c][i]=g[c]*v[c][i]/norms[c];}}
