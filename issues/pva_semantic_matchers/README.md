# Reproducing the Polygeist PVA experiments

This directory contains the retained evidence for raising ordinary C image
operations, recognizing their semantics, lowering them to NVIDIA PVA Solutions
operator calls, cross-compiling AArch64 executables, and validating complete
outputs on an NVIDIA Jetson.

Polygeist does **not** vendor NVIDIA PVA Solutions, the PVA SDK, cuPVA, or their
shared libraries. Obtain those components directly from NVIDIA under the terms
that apply to your organization. Do not copy an internal clone URL, license
server address, binary, credential, or board address into this repository.

## Supported reproduction levels

There are three increasingly demanding levels:

1. **Compiler-only:** raise the plain-C fixtures and test structural matching,
   numerical-contract policy, and MLIR lowering. This does not require PVA
   hardware or NVIDIA PVA binaries.
2. **AArch64 cross-build:** additionally compile the runtime adapter and link
   completed Jetson executables. This requires PVA Solutions headers, PVA SDK
   headers, compatible AArch64 PVA libraries, and a CUDA AArch64 cross sysroot.
3. **Silicon:** deploy only the completed AArch64 executables and required
   shared libraries to a compatible Jetson and compare complete outputs.

The paper-facing counting unit is one raised-C operation/datatype route. A
direct call to a PVA adapter is useful ABI evidence but is not counted as a
compiler match.

## Obtain the NVIDIA prerequisites

Start with NVIDIA's official documentation:

- [PVA SDK 2.9 installation](https://docs.nvidia.com/pva/sdk/2.9.0/installation.html)
- [PVA SDK 2.9 release notes](https://docs.nvidia.com/pva/sdk/2.9.0/release-notes.html)
- [Deploying PVA applications](https://docs.nvidia.com/pva/sdk/2.9.0/deploy.html)
- [PVA application allowlists](https://docs.nvidia.com/pva/sdk/2.9.0/vpu-allowlist.html)

NVIDIA distributes PVA SDK 2.9 using local Debian repository installers. After
obtaining the installers through your NVIDIA download or support channel, the
official L4T development package is installed in the following form:

```sh
sudo apt install ./<pva-sdk-local-repository-installer>.deb
sudo apt update
sudo apt install pva-sdk-2.9-l4t-dev
```

On a supported Jetson/L4T target, the matching cuPVA runtime package is:

```sh
sudo apt install pva-sdk-2.9-l4t
```

The SDK normally installs under `/opt/nvidia/pva-sdk-2.9`. PVA SDK 2.9 is
documented for Ubuntu 22.04/24.04 x86-64 hosts, JetPack 6.0 or newer targets,
and Orin or Thor PVA architectures. Always use the release notes for the SDK
version actually supplied to you; a newer shared object is not automatically
compatible with an older target BSP or cuPVA runtime.

PVA Solutions is a separate NVIDIA-provided source/package distribution. It is
not a public dependency of Polygeist. Obtain an entitled copy from the NVIDIA
channel associated with your PVA SDK access. Preserve its license and use its
own `README.rst` as the authority for that revision. The inspected revision
supports an L4T build using precompiled device targets, avoiding the optional
Synopsys ASIP Programmer during this reproduction:

```sh
git lfs install
git submodule update --init --recursive
cmake -S <pva-solutions> -B <pva-solutions>/build_l4t \
  -DPVA_BUILD_MODE=L4T \
  -DIMPORT_DEVICE_TARGETS=ON
cmake --build <pva-solutions>/build_l4t --target package -j"$(nproc)"
```

The reference campaign inspected PVA Solutions commit
`86bb7aa3de76785248d4be61773d4ba90aae6909` (described locally as
`v0.5.1-54-g86bb7aa3d`). This identifier records provenance; it is not a public
download location. If NVIDIA supplies a different revision, record it and
rerun the complete correctness matrix rather than assuming equivalence.

If your supplied PVA Solutions revision does not provide
`IMPORT_DEVICE_TARGETS`, follow that revision's documented ASIP Programmer and
license-server workflow. Never publish the license configuration.

The generated L4T packages normally include a runtime package, samples, and
sample assets. They may be installed on the target as documented by NVIDIA, or
their AArch64 libraries may be extracted into a private staging directory for
cross-linking:

```sh
mkdir -p <pva-package-extract>
dpkg-deb -x <pva-solutions-version-l4t.deb> <pva-package-extract>
find <pva-package-extract> \( -type f -o -type l \)
```

Do not commit the extracted files.

## Required build inputs

Set absolute paths before invoking the Polygeist PVA scripts:

```sh
export PVASOL_ROOT=/absolute/path/to/pva-solutions
export CUPVA_SDK_ROOT=/opt/nvidia/pva-sdk-2.9
export PVA_LIB_STAGE=/absolute/path/to/aarch64-pva-libraries
export POLYGEIST_CUDA_CROSS_ROOT=/absolute/path/to/cuda/targets/sbsa-linux
export POLYGEIST_PVA_TARGET_RPATH=/target/pva/lib:/target/cuda/lib
```

`PVA_LIB_STAGE` must make the link-time names for at least these AArch64
libraries available:

```text
libpva_operator.so
libnvcv_types.so
libcupva_host.so
```

Preserve the versioned files and their symlinks. Before building, verify their
architecture and dependency versions:

```sh
file "$PVA_LIB_STAGE"/libpva_operator.so
readelf -d "$PVA_LIB_STAGE"/libpva_operator.so | grep NEEDED
sha256sum "$PVA_LIB_STAGE"/libpva_operator.so
```

The reported file must be AArch64. Its `libcupva_host.so.<major>.<minor>`
dependency must be present on the target and compatible with the target BSP.

## Build Polygeist and run compiler-only tests

Build `cgeist` and `polygeist-opt` using the main project instructions, then
run:

```sh
PYTHON=/usr/bin/python3 \
  bash scripts/correctness/test_pva_structural_match.sh \
  /tmp/pva_structural_match_test
```

This checks that recognition is invariant under source-function renaming,
rejects structurally perturbed negative cases, selects exact routes by default,
and requires explicit budgets for approximate routes.

The lowering-specific tests are:

```sh
build/bin/polygeist-opt \
  test/polygeist-opt/lower-kernel-launch-pva-typed.mlir \
  --lower-kernel-launch-to-pva

# These inputs are expected to fail; the normal test suite checks their errors.
build/bin/polygeist-opt \
  test/polygeist-opt/lower-kernel-launch-pva-contract-reject.mlir \
  --lower-kernel-launch-to-pva
build/bin/polygeist-opt \
  test/polygeist-opt/lower-kernel-launch-pva-legacy-reject.mlir \
  --lower-kernel-launch-to-pva
```

## Cross-build every raised datatype route

With the NVIDIA inputs configured, run:

```sh
export POLYGEIST_SILICON_PROFILE=manual
export POLYGEIST_JETSON_HOST=<ssh-host-alias>
export POLYGEIST_JETSON_USER=<target-user>
export POLYGEIST_JETSON_LD_LIBRARY_PATH=<target-pva-and-cuda-library-paths>

bash scripts/correctness/run_pva_all_raised_dtypes.sh \
  /tmp/pva_all_raised_dtypes
```

For a direct SSH target, `manual` or `direct` mode is sufficient. More complex
bounce-host configuration belongs in the permission-restricted file selected
by `POLYGEIST_JETSON_CREDENTIALS_FILE`; never commit that file.

If the PVA libraries are not installed in a system loader path on the target,
either provide their directory through `POLYGEIST_JETSON_LD_LIBRARY_PATH` or
stage the required files beside the executable with
`POLYGEIST_JETSON_EXTRA_LIBS`. Use `ldd` on the target to confirm that every
dependency resolves before running.

If PVA application authentication is enabled, install the PVA Solutions
allowlist using NVIDIA's documented `nvidia-pva-allow` workflow. Do not disable
authentication merely to make an experiment pass.

## Numerical policy

Exact matches carry:

```text
polygeist.numerical_contract = "exact"
```

Box, bilateral, and histogram equalization differ from the corresponding
raised fixtures because of vendor fixed-point coefficients, support, or
rounding. They remain residual code unless the user explicitly authorizes a
per-operation bound, for example:

```sh
export POLYGEIST_PVA_APPROXIMATION_BUDGETS=box-filter=1
```

The resulting launch records both `polygeist.numerical_contract =
"approximate"` and `polygeist.max_abs_error_budget`. A measured error bound is
evidence for that input campaign, not a proof for every possible image.

MLIR integer types are signless. For histogram output, select the external ABI
explicitly instead of inferring U32/S32 from a source name:

```sh
export POLYGEIST_PVA_HISTOGRAM_OUTPUT_TYPE=u32  # or s32
```

## Expected retained result

The current occurrence-level ledger is
[`raised_c_numerical_contracts.csv`](raised_c_numerical_contracts.csv):

- 18 raised-C datatype routes structurally recognized, lowered, built, and
  executed;
- 12 exact routes across Gaussian, morphology, and histogram; and
- 6 explicit approximate routes across box, bilateral, and histogram
  equalization.

Regenerate the PVA HTML table with:

```sh
python3 scripts/correctness/build_ce_viewer.py --pva-only
```

The generated page is `/tmp/ir_viewer/pva.html` by default.

## Provenance checklist

For a publishable rerun, retain:

- Polygeist commit and dirty-state status;
- PVA Solutions revision or package version;
- PVA SDK, cuPVA runtime, JetPack/L4T, CUDA, compiler, and linker versions;
- SHA-256 hashes for `libpva_operator.so`, `libnvcv_types.so`,
  `libcupva_host.so`, the allowlist, harness, and executable;
- exact build and execution commands;
- full-output comparison results; and
- hardware model plus non-sensitive power/clock configuration.

Sanitize hostnames, addresses, usernames, credentials, and license-server
configuration before committing logs. Correctness campaigns do not establish a
performance result: use the project's publication benchmark methodology before
reporting speedups.
