# PET library.

include_directories("pet")
include_directories("pet/include")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

configure_file("config.h.cmake" "${CMAKE_CURRENT_BINARY_DIR}/config.h")

# Write configure result
# configure_file(... COPYONLY) avoids that the time stamp changes if the file is identical
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/gitversion.h.tmp"
  "#define GIT_HEAD_ID \"abcdef\"")
configure_file("${CMAKE_CURRENT_BINARY_DIR}/gitversion.h.tmp"
  "${CMAKE_CURRENT_BINARY_DIR}/gitversion.h" COPYONLY)

set(CLANG_PREFIX ${LLVM_BINARY_DIR})

set(PET_FILES
  pet/aff.c
  pet/array.c
  pet/clang.cc
  pet/context.c
  pet/dummy.cc
  #  pet/emit.c
  pet/expr_arg.c
  pet/expr.c
  pet/expr_plus.cc
  pet/filter.c
  pet/id.cc
  pet/inlined_calls.cc
  pet/inliner.cc
  pet/isl_id_to_pet_expr.c
  pet/killed_locals.cc
  pet/loc.c
  # pet/main.c
  pet/nest.c
  pet/options.c
  #  pet/parse.c
  pet/patch.c
  pet/pet.cc
  # pet/pet_check_code.c
  # pet/pet_codegen.c
  pet/pet_expr_to_isl_pw_aff.c
  # pet/pet_scop_cmp.c
  pet/print.c
  pet/scan.cc
  pet/scop.c
  pet/scop_plus.cc
  pet/skip.c
  pet/substituter.cc
  pet/summary.c
  pet/tree2scop.c
  pet/tree.c
  pet/value_bounds.c
  pet/version.cc
  )

add_library(PET STATIC ${PET_FILES})
target_include_directories(PET PUBLIC "../isl/isl/include")
target_include_directories(PET PUBLIC "${LLVM_BINARY_DIR}/tools/isl/include")
target_include_directories(PET PRIVATE "../clang/include")
target_include_directories(PET PRIVATE "${LLVM_BINARY_DIR}/tools/clang/include")
target_link_libraries(PET PUBLIC clangFrontend clangParse clangSema clangAST clangLex clangBasic clangDriver clangSerialization ISL)
target_compile_options(PET PRIVATE "-w")
target_compile_options(PET PRIVATE "-fno-rtti")
