#ifndef N
#define N 4096
#endif
#ifndef B0
#define B0 16
#endif
#ifndef B1
#define B1 12
#endif
void aten_histogram_select_outer_bin_edges_cpu(float x[N],float out_min[1],float out_max[1]){float lo=x[0],hi=x[0];for(int i=1;i<N;++i){lo=x[i]<lo?x[i]:lo;hi=x[i]>hi?x[i]:hi;}out_min[0]=lo;out_max[0]=hi;}
