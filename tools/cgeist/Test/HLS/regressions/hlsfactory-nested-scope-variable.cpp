// Regression (HLSFactory class labs): a pragma at the top of a function naming
// a buffer declared only inside a nested block was rejected as undeclared.
// A unique such declaration is used, with a warning; several are ambiguous.
// RUN: cgeist %s --function='*' -S --hls-annotate 2>&1 | FileCheck %s
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK: hlsfactory-nested-scope-variable.cpp:[[@LINE+5]]:38: warning: 'V_local' is declared in a nested scope that does not contain '#pragma HLS array_partition'; using that declaration
// CHECK: hls.array_partition = [{dim = 2 : i32, factor = 2 : i32, kind = "cyclic", variable = "V_local"}], hls.preserve, polygeist.varname = "V_local"
// CHECK-NOT: error
typedef float fixed_t;
void attention(fixed_t V[4][8][16], fixed_t out[4][16]) {
#pragma HLS ARRAY_PARTITION variable=V_local dim=2 factor=2 cyclic
  for (int b = 0; b < 4; ++b) {
    fixed_t V_local[8][16];
    for (int j = 0; j < 8; ++j)
      for (int k = 0; k < 16; ++k)
        V_local[j][k] = V[b][j][k];
    for (int k = 0; k < 16; ++k) {
      fixed_t sum = 0;
      for (int j = 0; j < 8; ++j)
        sum += V_local[j][k];
      out[b][k] = sum;
    }
  }
}

#ifdef ERRORS
void ambiguous(int *a) {
// ERR: hlsfactory-nested-scope-variable.cpp:[[@LINE+1]]:38: error: 'tmp' in '#pragma HLS array_partition' is ambiguous: it is declared in several nested scopes that do not contain the pragma
#pragma HLS array_partition variable=tmp complete
  {
    int tmp[4];
    tmp[0] = a[0];
    a[1] = tmp[0];
  }
  {
    int tmp[4];
    tmp[1] = a[1];
    a[2] = tmp[1];
  }
}
#endif
