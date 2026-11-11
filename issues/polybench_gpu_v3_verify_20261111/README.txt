Current-v3 Figure 4 verification
==============================

This directory retains fresh source builds and board executions of all twelve
FP64 LARGE PolyBench GPU cases. The saved compiler/catalog snapshot is described
by final-provenance.json and source-snapshot-final/. Later changes in the working
tree are not evidence for these binaries. Match proofs and compiler hashes are
checked against that frozen snapshot by audit.py.

Scope and comparisons
---------------------
The application and imported library references use the production raising path,
source-derived v3 catalog, shared matchers and typed ABI lowering. Residual loops
and scalar boundary writes are compiled to GPU work, and eligible buffers remain
resident across the kernel body. This is an explicit ABI test harness around the
compiler-produced kernel, rather than whole-program transformation.

Every canonical output is checked against the original C implementation using
abs(actual-reference) <= 1e-8 + 1e-8*abs(reference). Array names, exact sizes and
finite values are checked. An additional execution of the exact timed executable
records real vendor calls and must report zero staged host/device bytes.

All runs use five warmups and five measured repetitions; the paper statistic is
the minimum CUDA-event device compute interval. That interval includes residual
GPU computation and library work, excluding initial/final transfers and memory
allocation. All samples, transfer times, asynchronous host submission times and
synchronized end-to-end host times are retained in each *-run.json.

Ten native computational bodies are pinned to PolyBenchGPU commit
5584aaa7d0be810ff5eb0b61c49fb64ecc81ba4c. The correlation/covariance comparisons use
the exact locally authored CUDA adaptations used for the paper. They are labelled
paper-baseline, not claimed to be verbatim upstream kernels. No old executable
timings are used. Earlier attempts of this campaign are retained under attempts/.

Board power, clocks, load, thermal logs and fan settings are retained per run.
The fan remains automatic, so the fixed-fan publication protocol is not fully
satisfied. All measured temperatures must stay below 85 C for the audit to pass.

Reproduction commands (from the repository root)
----------------------------------------------
export POLYGEIST_ROOT=/home/arjaiswal/Polygeist
source "$POLYGEIST_ROOT/envsetup.sh"

With the source/compiler snapshot restored and a suitable remote adapter:
python3 issues/polybench_gpu_v3_verify_20261111/reproduce.py \
  --remote-helper /path/to/private/remote.py

The remote adapter supplies run(command, timeout) -> (exit_code, output) and
upload(local_path, remote_path) -> exit_code. Credentials are not part of the
artifact. The board must be reserved before measurement; do not overlap workloads.

To recheck saved evidence without a board or current production-source equality:
python3 issues/polybench_gpu_v3_verify_20261111/audit.py
python3 issues/polybench_gpu_v3_verify_20261111/report.py

Results are in report.txt, results.csv, results.json and the detailed audit.json.
Source/compiler regression evidence is in compiler-tests-final.log and
python-tests-final.log. STATUS.txt records the latest campaign state.
