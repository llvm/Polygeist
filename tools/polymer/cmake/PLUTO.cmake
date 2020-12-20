# Install PLUTO as an external project.

include(ExternalProject)

set(PLUTO_INCLUDE_DIR "${CMAKE_BINARY_DIR}/pluto/include")
set(PLUTO_LIB_DIR "${CMAKE_BINARY_DIR}/pluto/lib")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  execute_process(
    COMMAND bash -c "which ${CMAKE_CXX_COMPILER}"  
    OUTPUT_VARIABLE CLANG_ABSPATH
  )
  get_filename_component(CLANG_BINDIR ${CLANG_ABSPATH} DIRECTORY)
  get_filename_component(CLANG_PREFIX ${CLANG_BINDIR} DIRECTORY)
  set(CLANG_PREFIX_CONFIG "--with-clang-prefix=${CLANG_PREFIX}")
else()
  set(CLANG_PREFIX_CONFIG "")
endif()

ExternalProject_Add(
  pluto 
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/pluto"
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} ${CMAKE_SOURCE_DIR}/pluto/autogen.sh && ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} ${CMAKE_SOURCE_DIR}/pluto/configure --prefix=${CMAKE_BINARY_DIR}/pluto ${CLANG_PREFIX_CONFIG}
  PREFIX ${CMAKE_BINARY_DIR}/pluto
  BUILD_COMMAND make 
  INSTALL_COMMAND make install
  BUILD_IN_SOURCE 1
  BUILD_BYPRODUCTS 
   "${PLUTO_LIB_DIR}/libpluto.so"
   "${PLUTO_LIB_DIR}/libisl.so"
   "${PLUTO_LIB_DIR}/libosl.so"
   "${PLUTO_LIB_DIR}/libcloog-isl.so"
   "${PLUTO_LIB_DIR}/libpiplib_dp.so"
   "${PLUTO_LIB_DIR}/libpolylib64.so"
   "${PLUTO_LIB_DIR}/libcandl.so"
)

add_library(libpluto SHARED IMPORTED)
set_target_properties(libpluto PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libpluto.so")
add_library(libplutoosl SHARED IMPORTED)
set_target_properties(libplutoosl PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libosl.so")
add_library(libplutoisl SHARED IMPORTED)
set_target_properties(libplutoisl PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libisl.so")
add_library(libplutopip SHARED IMPORTED)
set_target_properties(libplutopip PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libpiplib_dp.so")
add_library(libplutopolylib SHARED IMPORTED)
set_target_properties(libplutopolylib PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libpolylib64.so")
add_library(libplutocloog SHARED IMPORTED)
set_target_properties(libplutocloog PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libcloog-isl.so")
add_library(libplutocandl STATIC IMPORTED)
set_target_properties(libplutocandl PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libcandl.so")

add_dependencies(libpluto pluto)
add_dependencies(libplutoisl pluto)
add_dependencies(libplutoosl pluto)
add_dependencies(libplutopip pluto)
add_dependencies(libplutopolylib pluto)
add_dependencies(libplutocloog pluto)
add_dependencies(libplutocandl pluto)

