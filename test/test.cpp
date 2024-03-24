

void gather(double *a, double *b, int *idx, int n, const int C) {
  for (int i = 0; i < n; i++) {
    // printf("a[%d](%lf) += %d * b[idx[%d](%d)](%lf) =", i, a[i], C, i, idx[i],
    // b[idx[i]]);
    a[i] += C * b[idx[i] * C];
    // printf(" %lf\n", a[i]);
  }
}

void scatter(double *a, double *b, int *idx, int *idx2, int n, const int C) {
  for (int i = 0; i < n; i++) {
    // printf("a[idx[%d](%d)](%lf) += %d * b[%d](%lf) =\n", i, idx[i],
    // a[idx[i]], C, i, b[i]);
    a[idx[idx2[i]]] += C * b[i];
    // printf(" %lf\n", a[idx[i]]);
  }
}

// int main(){
//     double a[10];
//     double b[10];
//     int idx[10];
//     int n = 10;
//     const int C = 2;
//     const int B = 3;
//     gather(a, b, idx, n, C);
//     scatter(a, b, idx, n, C);
// }
