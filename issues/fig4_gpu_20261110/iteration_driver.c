#include <stdio.h>
extern int polygeist_polybench_iteration(int, char **);
int main(void) {
  char *args[44] = {""};
  for (int i = 0; i < 10; ++i) {
    printf("POLYGEIST_BENCHMARK_ITERATION phase=%s index=%d total=10\n",
           i < 5 ? "warmup" : "sample", i < 5 ? i : i - 5);
    fflush(stdout);
    int rc = polygeist_polybench_iteration(i == 9 ? 43 : 0, args);
    if (rc) return rc;
  }
  return 0;
}
