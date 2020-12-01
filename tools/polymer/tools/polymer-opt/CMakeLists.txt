set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(polymer-opt
  polymer-opt.cc
)
llvm_update_compile_flags(polymer-opt)
target_link_libraries(polymer-opt
  PRIVATE
  MLIRStandardOps
  MLIRLLVMIR
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

  PolymerTransforms
  )
