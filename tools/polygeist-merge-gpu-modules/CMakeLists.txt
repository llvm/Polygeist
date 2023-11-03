get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

if(MLIR_INCLUDE_TESTS)
  set(test_libs
    MLIRTestDialect
    )
endif()

set(LIBS
  ${conversion_libs}
  ${dialect_libs}
  ${test_libs}
  MLIRDialect
  MLIRIR
  MLIRPass
  )

add_mlir_tool(polygeist-merge-gpu-modules
  polygeist-merge-gpu-modules.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(polygeist-merge-gpu-modules PRIVATE ${LIBS})
llvm_update_compile_flags(polygeist-merge-gpu-modules)

mlir_check_all_link_libraries(polygeist-merge-gpu-modules)

add_subdirectory(Test)
