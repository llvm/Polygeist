add_mlir_dialect_library(MLIRPolygeistTransforms
  AffineCFG.cpp
  AffineReduction.cpp
  CanonicalizeFor.cpp
  LoopRestructure.cpp
  Mem2Reg.cpp
  ParallelLoopDistribute.cpp
  ParallelLICM.cpp
  OpenMPOpt.cpp
  BarrierRemovalContinuation.cpp
  RaiseToAffine.cpp
  ParallelLower.cpp
  TrivialUse.cpp
  ConvertPolygeistToLLVM.cpp
  InnerSerialization.cpp
  ForBreakToWhile.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRPolygeistOpsIncGen
  MLIRPolygeistPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRAffineUtils
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRGPUOps
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMathToLLVM
  MLIRMemRefDialect
  MLIRNVVMDialect
  MLIRPass
  MLIRPolygeist
  MLIRSideEffectInterfaces
  MLIRSCFToControlFlow
  MLIRTransformUtils
  )
