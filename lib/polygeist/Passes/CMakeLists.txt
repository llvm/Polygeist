add_mlir_dialect_library(MLIRPolygeistTransforms
  AffineCFG.cpp
  AffineReduction.cpp
  BarrierRemovalContinuation.cpp
  CanonicalizeFor.cpp
  ConvertPolygeistToLLVM.cpp
  InnerSerialization.cpp
  LoopRestructure.cpp
  Mem2Reg.cpp
  OpenMPOpt.cpp
  ParallelLICM.cpp
  ParallelLoopDistribute.cpp
  ParallelLower.cpp
  RaiseToAffine.cpp
  TemporaryStorageElimination.cpp
  TrivialUse.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRPolygeistOpsIncGen
  MLIRPolygeistPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAsync
  MLIRAffineUtils
  MLIRFunc
  MLIRFuncTransforms
  MLIRGPUOps
  MLIRIR
  MLIRLLVMIR
  MLIRMath
  MLIRMathToLLVM
  MLIRMemRef
  MLIRNVVMIR
  MLIRPass
  MLIRPolygeist
  MLIRSideEffectInterfaces
  MLIRSCFToControlFlow
  MLIRTransformUtils
  )
