#define B 4
#define G 4
#define CPG 2
#define S 16
extern float sqrtf(float);void aten_group_norm_cpu(float x[B][G][CPG][S],float weight[G][CPG],float bias[G][CPG],float eps,float out[B][G][CPG][S],float mean[B][G],float rstd[B][G]){for(int b=0;b<B;++b)for(int g=0;g<G;++g){float s=0;for(int c=0;c<CPG;++c)for(int i=0;i<S;++i)s+=x[b][g][c][i];mean[b][g]=s/(CPG*S);float q=0;for(int c=0;c<CPG;++c)for(int i=0;i<S;++i){float d=x[b][g][c][i]-mean[b][g];q+=d*d;}rstd[b][g]=1.0f/sqrtf(q/(CPG*S)+eps);for(int c=0;c<CPG;++c)for(int i=0;i<S;++i)out[b][g][c][i]=(x[b][g][c][i]-mean[b][g])*rstd[b][g]*weight[g][c]+bias[g][c];}}
