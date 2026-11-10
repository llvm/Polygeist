// Algorithm-neutral in-process repetition driver for PolyBench executables.
//
// Compile the unchanged benchmark translation unit with
//   -Dmain=polygeist_polybench_iteration
// and link it with this file.  Each invocation executes the benchmark's own
// allocation, initialization, timed kernel region, and cleanup.  This keeps
// inputs identical across repetitions while allowing process-local library
// state (for example a cuBLAS handle) to warm up normally.

#include <stdio.h>

#ifndef POLYGEIST_WARMUP_RUNS
#define POLYGEIST_WARMUP_RUNS 5
#endif

#ifndef POLYGEIST_TIMED_RUNS
#define POLYGEIST_TIMED_RUNS 5
#endif

extern int polygeist_polybench_iteration(int argc, char **argv);

int main(int argc, char **argv) {
  const int total = POLYGEIST_WARMUP_RUNS + POLYGEIST_TIMED_RUNS;
  for (int iteration = 0; iteration < total; ++iteration) {
    const char *phase =
        iteration < POLYGEIST_WARMUP_RUNS ? "warmup" : "sample";
    const int phase_index = iteration < POLYGEIST_WARMUP_RUNS
                                ? iteration
                                : iteration - POLYGEIST_WARMUP_RUNS;
    printf("POLYGEIST_BENCHMARK_ITERATION phase=%s index=%d total=%d\n",
           phase, phase_index, total);
    fflush(stdout);
    int status = polygeist_polybench_iteration(argc, argv);
    if (status != 0)
      return status;
  }
  return 0;
}
