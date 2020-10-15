set(PET_TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(PET_TO_MLIR_DEPENDS
  FileCheck count not 
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
)

add_lit_testsuite(check-mlir-clang "Running the clang-to-mlir regression tests"
  ${CMAKE_CURRENT_BINARY_DIR} 
  DEPENDS ${PET_TO_MLIR_DEPENDS}
  )
set_target_properties(check-mlir-clang PROPERTIES FOLDER "clang-to-mlir tests")


