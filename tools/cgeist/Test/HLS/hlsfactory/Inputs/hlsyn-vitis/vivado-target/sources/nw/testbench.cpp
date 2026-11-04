#include "nw.h"

int main() {
  char SEQA[128];
  char SEQB[128];
  char alignedA[256];
  char alignedB[256];
  int M[16641];
  char ptr[16641];
  needwun(SEQA, SEQB, alignedA, alignedB, M, ptr);
  return 0;
}
