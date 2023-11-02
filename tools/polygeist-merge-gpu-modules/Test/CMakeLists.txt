set(MLIR_CLANG_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_CLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg
)

list(APPEND MLIR_CLANG_TEST_DEPS
  llvm-config
  FileCheck count not
  split-file
  mlir-translate
  polygeist-merge-gpu-modules
  clang
  clang++
  )

add_lit_testsuite(check-polygeist-merge-gpu-modules "Running the gpu module merge tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_CLANG_TEST_DEPS}
  )

set_target_properties(check-polygeist-merge-gpu-modules PROPERTIES FOLDER "gpu module merge tests")
