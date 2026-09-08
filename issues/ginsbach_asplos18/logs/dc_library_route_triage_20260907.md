# NPB DC external-library route triage — 2026-09-07

## Confirmed evidence

- Ginsbach et al. report six scalar-reduction instances for DC in Figure 16.
- The paper does not publish their source files, functions, or line numbers.
- All five recovered DC translation units now pass the Polygeist frontend and
  raising pipeline, producing 24 `linalg.generic` operations and no current
  external-library launch.
- `jobcntl.c` contains the relevant reduction-shaped IR. Most bodies carry
  multiple scalar states through fixed 32- or 64-iteration loops using shifts,
  bitwise OR/AND, comparisons, and conditional increments.

## Inferred source candidates

These are plausible explanations for the paper's six count, not confirmed
occurrence locations:

1. `setLeadingOnes32`, line 52: repeated shift and bitwise OR.
2. `NumOfCombsFromNbyK`, line 78: multiplication/division recurrence.
3. `NumberOfOnes`, line 293: 64-bit population count.
4. `CreateBinTuple`, line 351: indirect bitset construction.
5. `countTupleOnes`, line 408: bounded population count.
6. `MultiFileProcJobs`, line 489: maximum of `nGroupbys[0:nTasks]`.

The authors' reconstructed matcher reports many more candidate loops under a
different recovered pipeline, so it cannot identify these six uniquely.

## Library-route assessment

- The two population-count loops and leading-mask loop have scalar compiler or
  CUDA intrinsic equivalents, but not useful external GPU-library operations.
- The binomial-coefficient loop is an order-dependent integer recurrence.
- Tuple-bitset construction could be reformulated as a bitwise-OR transform
  reduction, but uses at most the number of dimensions and would require a new
  associative memory representation.
- The `MultiFileProcJobs` maximum maps semantically to CUB
  `DeviceReduce::Max`, but its input length is `nTasks`, a small control-plane
  quantity. Launch and transfer overhead would dominate.

## Decision

Do not inflate the external-library count by adding unprofitable scalar GPU
launches. DC remains at zero current computational launches. Its principal
classification is `missing loop or memory representation`, with a secondary
profitability blocker. Exact occurrence recovery from the authors' tool is
still required before mapping any of these candidates to the paper's six.
