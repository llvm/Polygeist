set(LLVM_LINK_COMPONENTS
  Support
)

#get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_tool(polymer-translate
  polymer-translate.cc
)
llvm_update_compile_flags(polymer-translate)
target_link_libraries(polymer-translate
  PRIVATE

  PolymerTargetOpenScop

  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIRTranslateLib
  MLIRToLLVMIRTranslationRegistration
  )
