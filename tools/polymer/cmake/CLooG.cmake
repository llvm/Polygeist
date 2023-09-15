# Build CLooG from the source tree.

include(ExternalProject)

ExternalProject_Add(
  cloog
  GIT_REPOSITORY https://github.com/kumasento/cloog.git
  GIT_TAG 43CFB85ED1E1BA1C2F27B450498522B35467ACE7
  CONFIGURE_COMMAND "./autogen.sh" && "./configure" --prefix=${CMAKE_CURRENT_BINARY_DIR}/cloog --with-osl-prefix="${CMAKE_CURRENT_BINARY_DIR}/openscop"
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/cloog
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  BUILD_IN_SOURCE 1
  BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/cloog/lib/libcloog-isl.a" "${CMAKE_CURRENT_BINARY_DIR}/cloog/lib/libisl.a"
)

set(CLOOG_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/cloog/include")
set(CLOOG_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/cloog/lib")

add_library(libcloog SHARED IMPORTED)
set_target_properties(libcloog PROPERTIES IMPORTED_LOCATION "${CLOOG_LIB_DIR}/libcloog-isl.a")
add_library(libisl SHARED IMPORTED)
set_target_properties(libisl PROPERTIES IMPORTED_LOCATION "${CLOOG_LIB_DIR}/libisl.a")

add_dependencies(libcloog cloog)
add_dependencies(libisl cloog)
