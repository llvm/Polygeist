# MFEM Section 4.2 refresh — 2026-09-07

The raised `abs_l1_mass_3d` application pipeline was revalidated on
`pva-compiler-orin-1.nvidia.com`, Jetson AGX Orin SM87, CUDA 12.6, MAXN with
CPU, GPU, and EMC clocks locked. The problem is f64, `NE=1024`, `D1D=4`, and
`Q1D=5`. Each process performs warm-up before timing 20 applications.

All five raised runs pass with maximum absolute and relative error
`6.9388939039072284e-18`. Raised times are 544.8576, 542.4544, 546.6336,
544.3200, and 545.2368 us; the median is 544.8576 us. The corresponding
same-process native C reference median is 777.4784 us.

The synchronized resident native MFEM CUDA baseline is 134.9440 us from the
prior normalized MAXN measurement session. Thus the raised/native-GPU runtime
ratio is 4.038x. This is a same-model, same-shape comparison across separate
measurement sessions, not a claim that the two binaries were co-scheduled.

The new compiler-visible repeated-call session gives the same result as the
surviving August composed binary (548.4528 us median when rerun under the same
MAXN configuration). It performs no CUDA calls in source: the compiler owns
the accelerator lifetime and keeps the connected library pipeline resident.

The audit also found and fixed general pipeline failures:

- residual GPU binaries now link CUDA libdevice, so device `sqrt` and related
  math no longer produce unresolved PTX calls;
- the generated-kernel cache now accommodates complete application pipelines
  instead of aborting after 128 distinct kernels.
- dynamic local host allocations that reach generated kernels are registered
  through their complete memref-view chains;
- device-to-host boundary copies use pinned staging, avoiding allocator pages
  already covered by mapped-host registrations;
- the mapped-host cache holds all simultaneously live full-program workspaces
  (Navier has 305), rather than evicting one before its first kernel use.

All 11 larger-application/operator paths now pass at `NE=1024`; the worst
maximum absolute error is `8.3266726846886741e-17`. Minimal surface additionally
required correcting the extracted C fixture: its quadrature loop had processed
only element zero and left the rest of the batch uninitialized. Ex9 graph
replay uses bounded persistent scratch because replay requires stable captured
pointers. The per-path timings other than the mass headline remain
single-process diagnostics: they execute repeated host calls and many small
graph/library stages, so they are not promoted as Section 4.2 performance
claims until each complete connected path has a compiler-visible session.
