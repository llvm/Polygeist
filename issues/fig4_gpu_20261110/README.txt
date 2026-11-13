Figure 4 GPU reproduction records

Start with report.txt and results.csv for the measured results and their
qualifications. audit.json records complete-output comparison results and
raw timing samples; provenance.json records the source and tool hashes at
measurement time. Its Git revision and no-commit flag describe that snapshot,
before the subsequent commit of these fixes and records.

The committed files include reproduction scripts, the iteration driver,
paper targets, audit/results, build and run metadata, board provenance, and
final regression/smoke-test logs. Large output dumps, executables, generated
IR, compiler backups, and intermediate attempts remain in the local campaign
directory; they are not part of this Git snapshot. Reproduction regenerates
the executables and complete dumps needed to rerun audit.py.

From the repository root, set POLYGEIST_ROOT, source envsetup.sh, and run:
  python3 issues/fig4_gpu_20261110/reproduce.py --remote-helper PRIVATE_ADAPTER
  python3 issues/fig4_gpu_20261110/report.py

The private adapter must provide run(command, timeout) and
upload(local, destination). Access configuration is intentionally external
to this repository. Builds use the configured AArch64 cross toolchain and
CUDA 12.6 installation. Board measurements must run sequentially.

These are isolated compiler-generated kernel measurements with explicit ABI
harnesses. Correlation/covariance use the actual locally adapted paper
baselines; separate upstream audits are labelled in the report. The fan was
automatic, so fixed-fan publication-protocol validation remains pending.
