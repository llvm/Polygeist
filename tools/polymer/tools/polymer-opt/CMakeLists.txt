set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(polymer-opt
  polymer-opt.cc
)
llvm_update_compile_flags(polymer-opt)
target_link_libraries(polymer-opt
  PRIVATE

  PolymerTransforms
  PolymerTargetOpenScop

  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIRMathDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIRAffineTransforms
  MLIRTranslateLib
  ${translation_libs}
  )
