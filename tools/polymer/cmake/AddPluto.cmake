# Install PLUTO as an external project.

include(ExternalProject)

set(PLUTO_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/pluto/include")
set(PLUTO_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/pluto/lib")
set(PLUTO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pluto")
set(PLUTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/pluto")

set(PLUTO_LIBCLANG_PREFIX "" CACHE STRING
    "The prefix to libclang used by Pluto (version < 10 required).")

if (NOT PLUTO_LIBCLANG_PREFIX)
  set(PLUTO_LIBCLANG_PREFIX_CONFIG "")
endif()

# Pluto configuration shell script.
string(CONCAT PLUTO_CONFIGURE_SHELL_SCRIPT
       "#!/usr/bin/env bash\n"
       "${PLUTO_SOURCE_DIR}/autogen.sh\n"
       "${PLUTO_SOURCE_DIR}/configure --prefix=${PLUTO_BINARY_DIR} ${PLUTO_LIBCLANG_PREFIX_CONFIG}\n")
set(PLUTO_CONFIGURE_COMMAND "${CMAKE_CURRENT_BINARY_DIR}/configure-pluto.sh")
file(GENERATE OUTPUT ${PLUTO_CONFIGURE_COMMAND}
     CONTENT ${PLUTO_CONFIGURE_SHELL_SCRIPT})

set(PLUTO_LLVM_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/llvm10")
ExternalProject_Add(
  llvm10
  GIT_REPOSITORY https://github.com/llvm/llvm-project.git
  GIT_TAG release/10.x
  CMAKE_ARGS -DLLVM_ENABLE_PROJECTS=clang -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_INSTALL_PREFIX=${PLUTO_LLVM_PREFIX}/llvm10-install
  PREFIX llvm10
  SOURCE_SUBDIR llvm
  PATCH_COMMAND sed -i.bak -e "/\\#include \\\"llvm\\/Support\\/Signals.h\\\"/i #include <stdint.h>" llvm/lib/Support/Signals.cpp &&
    sed -i.bak -e "/\\#include <vector>/i #include <limits>" llvm/utils/benchmark/src/benchmark_register.h
)

ExternalProject_Add(
  pluto
  GIT_REPOSITORY https://github.com/kumasento/pluto
  GIT_TAG 5603283fb3e74fb33c380bb52874972b440d51a2
  PREFIX pluto
  # SOURCE_DIR ${PLUTO_SOURCE_DIR}
  # CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env bash ${PLUTO_CONFIGURE_COMMAND}
  CONFIGURE_COMMAND ./autogen.sh &&
    ./configure --prefix=${CMAKE_CURRENT_BINARY_DIR}/pluto/install --with-clang-prefix=${PLUTO_LLVM_PREFIX}/llvm10-install
  INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/pluto
  BUILD_COMMAND make -j LDFLAGS="-Wl,--copy-dt-needed-entries"
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

add_dependencies(pluto llvm10)

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

