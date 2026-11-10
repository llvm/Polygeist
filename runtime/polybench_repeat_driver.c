// Algorithm-neutral in-process repetition driver for PolyBench executables.
//
// Compile the unchanged benchmark translation unit with
//   -Dmain=polygeist_polybench_iteration
// and link it with this file.  Each invocation executes the benchmark's own
// allocation, initialization, timed kernel region, and cleanup.  This keeps
// inputs identical across repetitions while allowing process-local library
// state (for example a cuBLAS handle) to warm up normally.

#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef POLYGEIST_WARMUP_RUNS
#define POLYGEIST_WARMUP_RUNS 5
#endif

#ifndef POLYGEIST_TIMED_RUNS
#define POLYGEIST_TIMED_RUNS 5
#endif

extern int polygeist_polybench_iteration(int argc, char **argv);

int main(int argc, char **argv) {
  // Without POLYBENCH_DUMP_ARRAYS, PolyBench guards its complete output
  // with argc > 42 and an empty argv[0]. Enable that guard only on the last
  // iteration so correctness and timing use the same executable.
  const char *dump_setting = getenv("POLYGEIST_DUMP_LAST");
  const int dump_last = dump_setting && strcmp(dump_setting, "1") == 0;
  char *dump_argv[44] = {""};
  int saved_stderr = dump_last ? dup(STDERR_FILENO) : -1;
  int sink = dump_last ? open("/dev/null", O_WRONLY) : -1;
  if (dump_last && (saved_stderr < 0 || sink < 0)) return 1;
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
    if (dump_last) {
      fflush(stderr);
      if (dup2(iteration == total - 1 ? saved_stderr : sink, STDERR_FILENO) < 0)
        return 1;
    }
    int status = polygeist_polybench_iteration(
        dump_last ? (iteration == total - 1 ? 43 : 0) : argc,
        dump_last ? dump_argv : argv);
    if (status != 0)
      return status;
  }
  if (dump_last) {
    close(saved_stderr);
    close(sink);
  }
  return 0;
}
