# MFEM Section 4.2 campaign design

This directory defines the minimal publication campaign without modifying the
historical artifacts in `../results`, `../match_results`,
`../application_extractions/results`, or `../silicon_results`.

## Publication questions

Section 4.2 needs two separately denominated results:

1. **Coverage:** which of the 20 MFEM semantic kernel pairs contain at least
   one external-library match, and how many static matched regions are emitted?
2. **Performance:** for a declared cohort of complete operator paths, what are
   the resident steady-state times for native C on the Orin CPU, native MFEM
   CUDA, and Polygeist-raised GPU execution under one measurement protocol?

Coverage and performance must not share a numerator unless they actually use
the same rows.  The current 40 fixture files, 20 semantic pairs, 11 application
paths, and 35/98/128 historical/live launch counts are different quantities.

## Denominators and allowed claims

- `semantic_kernel_count = 20`: one row for each original/normalized pair in
  `../manifest.csv`.  This is the coverage denominator.
- `kernel_with_external_match_count`: number of those 20 normalized rows with
  at least one ABI-lowerable launch to a pre-existing external library.  Say
  "contains a match," not "complete kernel replacement."
- `static_external_launch_count`: total static external-library launch sites
  across those 20 normalized rows.  This is a region/site count, not a kernel
  count and not a dynamic execution count.
- `performance_path_count`: number of complete extracted operator paths in the
  explicitly frozen performance cohort.  Initially use only
  `mfem_app_abs_l1_mass_3d` and `mfem_app_abs_l1_diffusion_3d`, because these are
  the two paths for which exact native-MFEM CUDA counterparts are recorded.
  Expand the cohort only after adding exact counterparts.

The 40 rows in `../manifest.csv` are representation variants (20 original and
20 normalized), not 40 independent kernels.  The 11 rows in
`../application_extractions/manifest.csv` are MFEM-derived, manually normalized
hot paths and are neither untouched MFEM applications nor the coverage
denominator.  Component sums, pairwise fallbacks, and unavailable native GPU
baselines are not valid performance rows.

## Required retained layout

Each immutable run directory outside the source checkout has this form:

```
mfem-section42-<UTC timestamp>-<12-char Polygeist SHA>/
  src/                         detached clean Git worktree
  metadata/
    snapshot.txt              Git/submodule/tool/compiler/library versions
    commands.log              commands in execution order
    environment.txt           allowlisted non-secret environment
    cmake-cache.txt
    input.sha256
    tool.sha256
    artifact.sha256
  logs/
    llvm-build.log
    build.log
    extraction_correctness.log
    raise.log
    match.log
    application_raise.log
  coverage/
    raise_summary.csv
    match_summary.csv
    coverage.csv
  correctness/
    extraction_pair_results.txt
    application_results.csv
  performance/
    cohort.csv
    raw/
    summary.csv
```

`coverage.csv` has exactly one row per semantic kernel and these columns:

```
semantic_id,original_id,normalized_id,family,dimension,operation,
original_source,normalized_source,upstream_file,upstream_symbol,
frontend_ok,raise_ok,fully_raised,linalg_ops,residual_loops,
debufferize_ok,matcher_ok,matched_region_count,external_launch_count,
external_symbols,residual_linalg_ops,abi_lower_ok,build_ok,correctness_ok,notes
```

The ledger generator must join rows using the manifest key
`(family, dimension, operation, upstream_file, upstream_symbol)`; the current
manifest has 20 unique such keys and exactly one `original` plus one
`normalized` row for each key.  It must not infer identity from function names
or count prose in Markdown files.
`external_launch_count` counts only launches that lower to an existing vendor
or platform library.  Project-authored generated kernels and analysis-only
candidates are recorded in `notes`, not counted.

## Freeze a clean source snapshot

Do not use the current shared checkout directly: it may contain unrelated work
and the existing sweep scripts write into their source tree.  First commit the
intended audited matcher/compiler state.  Then substitute that full commit hash
for `CAMPAIGN_COMMIT` below.  Do not use a branch name or `HEAD` in a retained
campaign.

```
export CAMPAIGN_COMMIT=<full-40-character-commit>
export CAMPAIGN_PARENT=/home/arjaiswal/mfem-section42-runs
export CAMPAIGN_ID="mfem-section42-$(date -u +%Y%m%dT%H%M%SZ)-$(printf '%s' "$CAMPAIGN_COMMIT" | cut -c1-12)"
export CAMPAIGN_ROOT="$CAMPAIGN_PARENT/$CAMPAIGN_ID"
export CAMPAIGN_SRC="$CAMPAIGN_ROOT/src"
mkdir -p "$CAMPAIGN_ROOT"/{metadata,logs,coverage,correctness,performance/raw}
git -C /home/arjaiswal/Polygeist cat-file -e "$CAMPAIGN_COMMIT^{commit}"
git -C /home/arjaiswal/Polygeist worktree add --detach "$CAMPAIGN_SRC" "$CAMPAIGN_COMMIT"
git -C "$CAMPAIGN_SRC" submodule update --init --recursive
test -z "$(git -C "$CAMPAIGN_SRC" status --porcelain=v1)"
test "$(git -C "$CAMPAIGN_SRC" rev-parse HEAD)" = "$CAMPAIGN_COMMIT"
```

The campaign is invalid if either `test` fails.  The MFEM provenance expected
by the corpus is upstream commit
`951cf8886b9c0c33fb36a2f0ede268c8d6a0d8b5`; retain this literal revision in
`snapshot.txt` and verify any upstream checkout against it.

## Build the pinned tools

Build LLVM/MLIR/Clang and Polygeist from the detached worktree.  Do not reuse
the shared checkout's LLVM build for a publication campaign: at the time this
design was written its LLVM source tree contained local edits, so a binary hash
alone would not identify its source.  The currently expected LLVM submodule revision is
`26eb4285b56edd8c897642078d91f16ff0fd3472` (LLVM 18 development snapshot), but
the run must record rather than assume it.

```
export LLVM_BUILD="$CAMPAIGN_SRC/llvm-project/build"
test -z "$(git -C "$CAMPAIGN_SRC/llvm-project" status --porcelain=v1)"
cmake -S "$CAMPAIGN_SRC/llvm-project/llvm" -B "$LLVM_BUILD" -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLVM_BUILD" --target mlir-opt mlir-translate clang -j 24 \
  2>&1 | tee "$CAMPAIGN_ROOT/logs/llvm-build.log"
cmake -S "$CAMPAIGN_SRC" -B "$CAMPAIGN_SRC/build" -G Ninja \
  -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" \
  -DClang_DIR="$LLVM_BUILD/lib/cmake/clang" \
  -DLLVM_EXTERNAL_LIT="$LLVM_BUILD/bin/llvm-lit" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$CAMPAIGN_SRC/build" --target cgeist polygeist-opt -j 24 \
  2>&1 | tee "$CAMPAIGN_ROOT/logs/build.log"
```

Building LLVM at `$CAMPAIGN_SRC/llvm-project/build` also satisfies the current
sweep scripts' pinned Clang resource-directory lookup without a symlink to an
uncontrolled build.

## Record provenance before running

Record only an allowlist of environment variables; never dump credentials.

```
{
  printf 'polygeist_commit=%s\n' "$(git -C "$CAMPAIGN_SRC" rev-parse HEAD)"
  printf 'polygeist_describe=%s\n' "$(git -C "$CAMPAIGN_SRC" describe --always --dirty)"
  git -C "$CAMPAIGN_SRC" submodule status --recursive
  printf 'mfem_upstream_commit=%s\n' 951cf8886b9c0c33fb36a2f0ede268c8d6a0d8b5
  "$CAMPAIGN_SRC/build/bin/cgeist" --version
  "$CAMPAIGN_SRC/build/bin/polygeist-opt" --version
  clang --version
  cmake --version
  ninja --version
  python3 --version
} > "$CAMPAIGN_ROOT/metadata/snapshot.txt"
cmake -LAH -N "$CAMPAIGN_SRC/build" > "$CAMPAIGN_ROOT/metadata/cmake-cache.txt"
env | LC_ALL=C sort | rg '^(CC|CXX|CFLAGS|CXXFLAGS|CUDA_HOME|PATH|LD_LIBRARY_PATH|OMP_NUM_THREADS|OPENBLAS_NUM_THREADS)=' \
  > "$CAMPAIGN_ROOT/metadata/environment.txt"
find "$CAMPAIGN_SRC/issues/mfem_c_kernels" \
  \( -path '*/results/*' -o -path '*/match_results/*' -o -path '*/silicon_results/*' \) -prune -o \
  -type f \( -name '*.c' -o -name '*.h' -o -name '*.csv' \) -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum > "$CAMPAIGN_ROOT/metadata/input.sha256"
sha256sum \
  "$CAMPAIGN_SRC/build/bin/cgeist" \
  "$CAMPAIGN_SRC/build/bin/polygeist-opt" \
  "$CAMPAIGN_SRC/scripts/correctness/mfem_raise_sweep.py" \
  "$CAMPAIGN_SRC/scripts/correctness/mfem_match_sweep.py" \
  "$CAMPAIGN_SRC/scripts/correctness/mfem_application_raise_sweep.py" \
  "$CAMPAIGN_SRC/scripts/correctness/mfem_validate_extractions.py" \
  "$CAMPAIGN_SRC/scripts/correctness/kernel_match.py" \
  "$CAMPAIGN_SRC/scripts/correctness/kernel_match_rewrite.py" \
  > "$CAMPAIGN_ROOT/metadata/tool.sha256"
```

## Regenerate coverage without touching historical artifacts

The sweep scripts write only inside the detached worktree.  Run them there and
immediately copy the authoritative CSVs and logs to the immutable run root.

```
cd "$CAMPAIGN_SRC"
python3 scripts/correctness/mfem_validate_extractions.py \
  2>&1 | tee "$CAMPAIGN_ROOT/logs/extraction_correctness.log"
python3 scripts/correctness/mfem_raise_sweep.py \
  2>&1 | tee "$CAMPAIGN_ROOT/logs/raise.log"
python3 scripts/correctness/mfem_match_sweep.py \
  2>&1 | tee "$CAMPAIGN_ROOT/logs/match.log"
python3 scripts/correctness/mfem_application_raise_sweep.py \
  2>&1 | tee "$CAMPAIGN_ROOT/logs/application_raise.log"
cp issues/mfem_c_kernels/results/summary.csv \
  "$CAMPAIGN_ROOT/coverage/raise_summary.csv"
cp issues/mfem_c_kernels/match_results/summary.csv \
  "$CAMPAIGN_ROOT/coverage/match_summary.csv"
cp issues/mfem_c_kernels/application_extractions/results/summary.csv \
  "$CAMPAIGN_ROOT/coverage/application_raise_summary.csv"
cp "$CAMPAIGN_ROOT/logs/extraction_correctness.log" \
  "$CAMPAIGN_ROOT/correctness/extraction_pair_results.txt"
```

Before accepting `coverage.csv`, enforce all of these invariants in its
generator:

- exactly 20 original and 20 normalized manifest rows;
- exactly one normalized partner for each semantic kernel;
- exactly 20 coverage-ledger rows;
- every counted external symbol is ABI-lowerable at the pinned revision;
- `kernel_with_external_match_count` counts rows, while
  `static_external_launch_count` sums sites;
- matched IR and ABI-lowered IR are parsed, not inferred from matcher text;
- the two totals are emitted independently;
- original-versus-normalized numerical correctness must pass before a row can
  be publication eligible.

Do not promote the previously observed live value of 128, or the retained 35
and 98 values, until this clean run produces and validates one value.

## Minimal performance cohort and protocol

Create `performance/cohort.csv` with the two initial exact-counterpart rows:

```
path_id,source_function,datatype,NE,D1D,Q1D,native_cpu,native_mfem_cuda,raised_gpu
abs_l1_mass_3d,mfem_app_abs_l1_mass_3d,f64,1024,4,5,required,exact,required
abs_l1_diffusion_3d,mfem_app_abs_l1_diffusion_3d,f64,1024,4,5,required,exact,required
```

For each column, cross-compile on the x86 host and execute the completed AArch64
binary on the same selected Orin.  Use identical generated input bytes and
hash them.  Use resident buffers and the same synchronized host-wall timing
boundary.  Each implementation uses 5 independent processes, 5 warm-ups, and
20 timed operations.  Retain all 100 samples per implementation, the five
process medians, median/IQR/min/max, complete-output `atol`/`rtol` results,
binary/shared-library hashes, clocks, power mode, temperature, and throttling
state.  Rotate CPU/native-GPU/raised-GPU execution order between process sets.

No performance row is publication-ready unless all three implementations pass
correctness on identical inputs.  Do not combine the existing mapped-host
raised timings with resident native-MFEM CUDA timings, or timings obtained on
different boards/sessions.

## Final campaign acceptance

After all host and silicon steps:

```
git -C "$CAMPAIGN_SRC" diff --exit-code -- \
  scripts lib tools runtime issues/mfem_c_kernels/original \
  issues/mfem_c_kernels/normalized \
  issues/mfem_c_kernels/application_extractions/*.c \
  issues/mfem_c_kernels/application_extractions/*.h \
  issues/mfem_c_kernels/manifest.csv \
  issues/mfem_c_kernels/application_extractions/manifest.csv
find "$CAMPAIGN_ROOT" -type f ! -path '*/src/*' -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum \
  > "$CAMPAIGN_ROOT/metadata/artifact.sha256"
```

The final report must state, without combining denominators:

- `20` semantic kernels in the frozen coverage corpus;
- how many of 20 contain at least one validated external match;
- how many validated static external launch sites those rows contain;
- how many of the 20 are fully replaced versus retain residual computation;
- how many declared performance paths have correct, comparable three-column
  measurements; and
- that the evaluated source is manually normalized MFEM-derived C unless and
  until the same result is demonstrated from untouched upstream MFEM.
