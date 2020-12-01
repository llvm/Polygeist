set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(polymer-translate
  polymer-translate.cc
)
llvm_update_compile_flags(polymer-translate)
target_link_libraries(polymer-translate
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

  PolymerTargetOpenScop
  )
