#include "nussinov.h"


#include "ap_fixed.h"
#include "hls_math.h"


void kernel_nussinov( char seq[ 500 + 0],
			   int table[ 500 + 0][500 + 0])
{
  #pragma HLS top name=kernel_nussinov

    const int n = 500;

  int i, j, k;

 for (i = n-1; i >= 0; i--) {
  for (j=i+1; j<n; j++) {

   if (j-1>=0)
      table[i][j] = ((table[i][j] >= table[i][j-1]) ? table[i][j] : table[i][j-1]);
   if (i+1<n)
      table[i][j] = ((table[i][j] >= table[i+1][j]) ? table[i][j] : table[i+1][j]);

   if (j-1>=0 && i+1<n) {

     if (i<j-1)
        table[i][j] = ((table[i][j] >= table[i+1][j-1]+(((seq[i])+(seq[j])) == 3 ? 1 : 0)) ? table[i][j] : table[i+1][j-1]+(((seq[i])+(seq[j])) == 3 ? 1 : 0));
     else
        table[i][j] = ((table[i][j] >= table[i+1][j-1]) ? table[i][j] : table[i+1][j-1]);
   }

   for (k=i+1; k<j; k++) {
      table[i][j] = ((table[i][j] >= table[i][k] + table[k+1][j]) ? table[i][j] : table[i][k] + table[k+1][j]);
   }
  }
 }

}