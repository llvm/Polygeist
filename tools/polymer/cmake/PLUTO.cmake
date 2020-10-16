# Install PLUTO as an external project.

include(ExternalProject)

ExternalProject_Add(
  pluto 
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/pluto"
  CONFIGURE_COMMAND "${CMAKE_SOURCE_DIR}/pluto/autogen.sh" && "${CMAKE_SOURCE_DIR}/pluto/configure" --prefix=${CMAKE_BINARY_DIR}/pluto
  PREFIX ${CMAKE_BINARY_DIR}/pluto
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  BUILD_IN_SOURCE 1
  BUILD_BYPRODUCTS "${CMAKE_BINARY_DIR}/pluto/lib/libpluto.a"
)

set(PLUTO_INCLUDE_DIR "${CMAKE_BINARY_DIR}/pluto/include")
set(PLUTO_LIB_DIR "${CMAKE_BINARY_DIR}/pluto/lib")

add_library(libpluto SHARED IMPORTED)
set_target_properties(libpluto PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libpluto.a")
add_dependencies(libpluto pluto)

