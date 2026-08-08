#define B 2
#define G 4
#define CPG 3
#define S 32
void aten_channel_shuffle_cpu(float x[B][G*CPG][S],float out[B][G*CPG][S]){for(int n=0;n<B;++n)for(int g=0;g<G;++g)for(int c=0;c<CPG;++c)for(int s=0;s<S;++s)out[n][c*G+g][s]=x[n][g*CPG+c][s];}
