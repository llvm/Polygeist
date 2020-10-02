# Build CLooG from the source tree.

include(ExternalProject)

ExternalProject_Add(
  cloog
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/cloog"
  CONFIGURE_COMMAND "${CMAKE_SOURCE_DIR}/cloog/autogen.sh" && "${CMAKE_SOURCE_DIR}/cloog/configure" --prefix=${CMAKE_BINARY_DIR}/cloog --with-osl-prefix="${CMAKE_BINARY_DIR}/openscop"
  PREFIX ${CMAKE_BINARY_DIR}/cloog
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  BUILD_IN_SOURCE 1
  BUILD_BYPRODUCTS "${CMAKE_BINARY_DIR}/cloog/lib/libcloog-isl.a" "${CMAKE_BINARY_DIR}/cloog/lib/libisl.a"
)

set(CLOOG_INCLUDE_DIR "${CMAKE_BINARY_DIR}/cloog/include")
set(CLOOG_LIB_DIR "${CMAKE_BINARY_DIR}/cloog/lib")

add_library(libcloog SHARED IMPORTED)
set_target_properties(libcloog PROPERTIES IMPORTED_LOCATION "${CLOOG_LIB_DIR}/libcloog-isl.a")
add_library(libisl SHARED IMPORTED)
set_target_properties(libisl PROPERTIES IMPORTED_LOCATION "${CLOOG_LIB_DIR}/libisl.a")

add_dependencies(libcloog cloog)
add_dependencies(libisl cloog)
