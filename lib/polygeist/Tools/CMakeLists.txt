add_mlir_dialect_library(MLIRPolygeistTools
  MergeHostDeviceGPUModules.cpp

  DEPENDS
  MLIRIR
)
add_llvm_pass_plugin(PolygeistCUDALaunchFixUp
  CUDALaunchFixUp.cpp
  DEPENDS
  intrinsics_gen
  BUILDTREE_ONLY
)
