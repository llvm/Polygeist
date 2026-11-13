#ifndef POLYGEIST_POLYBENCH_REPEAT_ENTRY_H
#define POLYGEIST_POLYBENCH_REPEAT_ENTRY_H

// Keep a source main renamed by the benchmark driver visible to whole-program
// Cgeist even though no function in the original translation unit calls it.
int polygeist_polybench_iteration(int argc, char **argv)
    __attribute__((used, externally_visible));

#endif
