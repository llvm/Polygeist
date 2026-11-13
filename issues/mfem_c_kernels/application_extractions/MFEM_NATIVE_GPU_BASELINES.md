# MFEM native GPU baselines

MFEM's optimized GPU implementations are not generally standalone `.cu`
files. They are C++ partial-assembly kernels expressed with
`MFEM_HOST_DEVICE`, `mfem::forall`, `MFEM_FOREACH_THREAD`, and shared-memory
templates. A CUDA-enabled MFEM build compiles these lambdas into native CUDA
kernels.

The native implementations corresponding to the extracted application stages
are:

- Scalar mass: `fem/integ/bilininteg_mass_kernels.hpp`,
  `SmemPAMassApply3D` and `PAMassApply3D`.
- Vector mass: `fem/integ/bilininteg_vecmass_pa.hpp`,
  `SmemPAVectorMassApply3D`.
- Scalar diffusion: `fem/integ/bilininteg_diffusion_pa.cpp` and its shared
  tensor-product kernels.
- Vector diffusion: `fem/integ/bilininteg_vecdiffusion_pa.hpp`,
  `SmemPAVectorDiffusionApply3D`.
- Nonlinear vector convection:
  `fem/integ/nonlininteg_vecconvection_pa.cpp`,
  `SmemPAConvectionNLApply3D`.
- Discrete gradient: `fem/integ/bilininteg_gradient_pa.cpp`,
  `PAGradientApply3D` and `SmemPAGradientApply3D`.
- H(curl) mass and curl-curl:
  `fem/integ/bilininteg_hcurl_kernels.hpp`, including
  `SmemPACurlCurlApply3D`.
- H(div) mass and div-div: `fem/integ/bilininteg_hdiv_kernels.hpp`, including
  `SmemPAHdivMassApply3D` and `PADivDivApply3D`.

MFEM already provides a suitable performance driver in
`tests/benchmarks/bench_assembly_levels.cpp`. Its `BK1` through `BK6` partial
assembly cases exercise scalar/vector mass and scalar/vector diffusion over
roughly 1,000 to 10,000,000 degrees of freedom. The larger applications
`examples/ex35p.cpp`, `miniapps/mtop/mtop_test_iso_elasticity.cpp`,
`miniapps/dfem/dfem-minimal-surface.cpp`, and
`miniapps/fluids/navier/navier_tgv.cpp` provide whole-application native GPU
baselines.

The attached Jetson currently supplies CUDA 12.6 runtime libraries but no
`nvcc`, NVRTC, CUDA development toolkit, or CUDA-enabled MFEM build. Therefore
the native source mapping is complete, but performance execution is blocked
until a CUDA compiler toolkit is installed and MFEM is rebuilt with
`MFEM_USE_CUDA=YES` for the Jetson architecture.
