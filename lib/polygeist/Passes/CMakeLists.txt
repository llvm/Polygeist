add_mlir_dialect_library(MLIRPolygeistTransforms
  AffineCFG.cpp
  AffineReduction.cpp
  CanonicalizeFor.cpp
  LoopRestructure.cpp
  Mem2Reg.cpp
  ParallelLoopDistribute.cpp
  BarrierRemovalContinuation.cpp
  RaiseToAffine.cpp
  ParallelLower.cpp
  TrivialUse.cpp
  ConvertPolygeistToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRPolygeistOpsIncGen
  MLIRPolygeistPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAffineUtils
  MLIRFunc
  MLIRFuncTransforms
  MLIRGPUOps
  MLIRIR
  MLIRLLVMIR
  MLIRMemRef
  MLIRNVVMIR
  MLIRPass
  MLIRPolygeist
  MLIRSideEffectInterfaces
  MLIRSCFToControlFlow
  MLIRTransformUtils
  )
