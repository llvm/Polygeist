set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(emit-openscop
  emit-openscop.cc
)
llvm_update_compile_flags(emit-openscop)
target_link_libraries(emit-openscop
  PRIVATE
  MLIRStandardOps
  MLIRLoopAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIREDSC
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR

  POLYMEREmitOpenScop
  )