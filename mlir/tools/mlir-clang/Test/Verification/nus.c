
#define N 5500
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

void kernel_nussinov(int n, int table[N])
{
  int j;

#pragma scop
  for (j=1; j<N; j++) {

   if (j-1>=0)
      table[j] = max_score(table[j], table[j-1]);

 }
#pragma endscop

}
