set(MLIR_CLANG_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_CLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CLANG_HEADER_DIR ${LLVM_MAIN_SRC_DIR}/../clang/lib/Headers)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
)

add_lit_testsuite(check-mlir-clang "Running the clang-to-mlir regression tests"
  ${CMAKE_CURRENT_BINARY_DIR} 
  DEPENDS ${PET_TO_MLIR_DEPENDS}
  )
set_target_properties(check-mlir-clang PROPERTIES FOLDER "clang-to-mlir tests")
add_dependencies(check-mlir-clang
  FileCheck mlir-clang
  llvm-config count not)
