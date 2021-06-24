// RUN: mlir-clang %s -detect-reduction --function=kernel_nussinov | FileCheck %s

#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

void kernel_nussinov(int n, double table[20][60])  {
  int i, j, k;

#pragma scop
    i =0;
//  for (j=i+1; j<_PB_N; j++) {
     j = 0;
   for (k=0; k<j; k++) {
      table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
   }
  //}
#pragma endscop

}