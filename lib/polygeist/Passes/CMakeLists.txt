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
  RemoveDeviceFunctions.cpp
  ConvertParallelToGPU.cpp
  SerializeToCubin.cpp

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
  MLIRTargetLLVMIRImport
  MLIRTransformUtils
  MLIRControlFlowToLLVM
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRArithToLLVM
  MLIROpenMPToLLVM
  )

if(POLYGEIST_ENABLE_CUDA)
  find_package(CUDA)
  enable_language(CUDA)

  # Enable gpu-to-cubin pass.
  target_compile_definitions(obj.MLIRPolygeistTransforms
    PRIVATE
    POLYGEIST_ENABLE_CUDA=1
  )

  # Add CUDA headers includes and the libcuda.so library.
  target_include_directories(obj.MLIRPolygeistTransforms
    PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

  find_library(CUDA_DRIVER_LIBRARY cuda)

  target_link_libraries(MLIRPolygeistTransforms
    PRIVATE
    ${CUDA_DRIVER_LIBRARY}
  )

endif()
