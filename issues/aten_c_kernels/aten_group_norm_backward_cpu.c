#define B 4
#define G 4
#define CPG 2
#define S 16
void aten_group_norm_backward_cpu(float grad[B][G][CPG][S],float x[B][G][CPG][S],float mean[B][G],float rstd[B][G],float weight[G][CPG],float dx[B][G][CPG][S]){for(int b=0;b<B;++b)for(int g=0;g<G;++g){float s1=0,s2=0;for(int c=0;c<CPG;++c)for(int i=0;i<S;++i){float w=grad[b][g][c][i]*weight[g][c];s1+=w;s2+=w*(x[b][g][c][i]-mean[b][g]);}for(int c=0;c<CPG;++c)for(int i=0;i<S;++i){float w=grad[b][g][c][i]*weight[g][c];dx[b][g][c][i]=rstd[b][g]/(CPG*S)*((CPG*S)*w-s1-(x[b][g][c][i]-mean[b][g])*rstd[b][g]*rstd[b][g]*s2);}}}
