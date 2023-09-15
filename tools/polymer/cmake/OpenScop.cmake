# Install OpenScop as an external project.

include(ExternalProject)

ExternalProject_Add(
  osl
  GIT_REPOSITORY https://github.com/periscop/openscop.git
  GIT_TAG 37805d8fef38c2d1b8aa8f5c26b40f79100322e7
  CONFIGURE_COMMAND "./autogen.sh" && "./configure" --prefix=${CMAKE_CURRENT_BINARY_DIR}/openscop
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/openscop
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  BUILD_IN_SOURCE 1
  BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/openscop/lib/libosl.a"
)

set(OSL_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/openscop/include")
set(OSL_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/openscop/lib")

add_library(libosl SHARED IMPORTED)
set_target_properties(libosl PROPERTIES IMPORTED_LOCATION "${OSL_LIB_DIR}/libosl.a")
add_dependencies(libosl osl)
