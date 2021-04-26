set(MLIR_CLANG_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_CLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CLANG_HEADER_DIR ${LLVM_MAIN_SRC_DIR}/../clang/lib/Headers)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  PATHS
  "LLVM_SOURCE_DIR"
  "LLVM_BINARY_DIR"
  "LLVM_TOOLS_DIR"
)

list(APPEND MLIR_CLANG_TEST_DEPS
  llvm-config 
  FileCheck count not
  mlir-clang
  split-file
  clang
  )

add_lit_testsuite(check-mlir-clang "Running the clang-to-mlir regression tests"
  ${CMAKE_CURRENT_BINARY_DIR} 
  DEPENDS ${MLIR_CLANG_TEST_DEPS}
  )

add_lit_testsuite(check-mlir-clang-single "Running the clang-to-mlir regression tests"
  ${CMAKE_CURRENT_BINARY_DIR} 
  DEPENDS ${MLIR_CLANG_TEST_DEPS}
  ARGS -j 1
  )

set_target_properties(check-mlir-clang PROPERTIES FOLDER "clang-to-mlir tests")
