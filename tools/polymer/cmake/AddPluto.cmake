# Install PLUTO as an external project.

include(ExternalProject)

set(PLUTO_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/pluto/include")
set(PLUTO_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/pluto/lib")
set(PLUTO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pluto")
set(PLUTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/pluto")

set(PLUTO_LIBCLANG_PREFIX "" CACHE STRING
    "The prefix to libclang used by Pluto (version < 10 required).")

# If PLUTO_LIBCLANG_PREFIX is not set, we try to find a working version.
# Note that if you set this prefix to a invalid path, then that path will be cached and 
# the following code won't remedy that.
if (NOT PLUTO_LIBCLANG_PREFIX)
  message(STATUS "PLUTO_LIBCLANG_PREFIX not provided")

  # If the provided CMAKE_CXX_COMPILER is clang, we will check its version and use its prefix if version is matched.
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 10)
      execute_process(
        COMMAND bash -c "which ${CMAKE_CXX_COMPILER}"  
        OUTPUT_VARIABLE CLANG_ABSPATH
      )
      get_filename_component(CLANG_BINARY_DIR ${CLANG_ABSPATH} DIRECTORY)
      get_filename_component(CLANG_PREFIX_DIR ${CLANG_BINARY_DIR} DIRECTORY)

      message (STATUS "Provided CMAKE_CXX_COMPILER is clang of version less than 10 (${CMAKE_CXX_COMPILER_VERSION})") 
      message (STATUS "Use its prefix for PLUTO_LIBCLANG_PREFIX: ${CLANG_PREFIX_DIR}")

      set(PLUTO_LIBCLANG_PREFIX ${CLANG_PREFIX_DIR})
    endif()
  endif()

endif()

if (NOT PLUTO_LIBCLANG_PREFIX)
  set(PLUTO_LIBCLANG_PREFIX_CONFIG "")
else()
  # If a valid libclang is still not found, we try to search it on the system.
  message(STATUS "PLUTO_LIBCLANG_PREFIX: ${PLUTO_LIBCLANG_PREFIX}")
  set(PLUTO_LIBCLANG_PREFIX_CONFIG "--with-clang-prefix=${PLUTO_LIBCLANG_PREFIX}")
endif()

# Bootstrap Pluto
if (NOT EXISTS "${PLUTO_SOURCE_DIR}/.git")
  message(STATUS "Pluto not found at ${PLUTO_SOURCE_DIR}, downloading ...")
  execute_process(COMMAND     ${POLYMER_SOURCE_DIR}/scripts/update-pluto.sh
                  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/update-pluto.log)
endif()

# Get Pluto git commit.
execute_process(COMMAND git rev-parse HEAD
                OUTPUT_VARIABLE PLUTO_GIT_HASH
                OUTPUT_STRIP_TRAILING_WHITESPACE
                WORKING_DIRECTORY ${PLUTO_SOURCE_DIR})
message(STATUS "Pluto git hash: ${PLUTO_GIT_HASH}")

# Get all Pluto submodule git status 
execute_process(COMMAND git submodule status --recursive
                OUTPUT_VARIABLE PLUTO_SUBMODULE_GIT_STATUS
                OUTPUT_STRIP_TRAILING_WHITESPACE
                WORKING_DIRECTORY ${PLUTO_SOURCE_DIR})
STRING(REGEX REPLACE "\n" ";" PLUTO_SUBMODULE_GIT_STATUS ${PLUTO_SUBMODULE_GIT_STATUS})
foreach (submodule ${PLUTO_SUBMODULE_GIT_STATUS})
  STRING(STRIP ${submodule} submodule)
  message(STATUS "${submodule}")
endforeach()

# Pluto configuration shell script.
string(CONCAT PLUTO_CONFIGURE_SHELL_SCRIPT
       "#!/usr/bin/env bash\n"
       "${PLUTO_SOURCE_DIR}/autogen.sh\n"
       "PATH=${PLUTO_LIBCLANG_PREFIX}/bin:${PATH} ${PLUTO_SOURCE_DIR}/configure --prefix=${PLUTO_BINARY_DIR} ${PLUTO_LIBCLANG_PREFIX_CONFIG}\n")
set(PLUTO_CONFIGURE_COMMAND "${CMAKE_CURRENT_BINARY_DIR}/configure-pluto.sh")
file(GENERATE OUTPUT ${PLUTO_CONFIGURE_COMMAND}
     CONTENT ${PLUTO_CONFIGURE_SHELL_SCRIPT})

ExternalProject_Add(
  pluto 
  PREFIX ${PLUTO_BINARY_DIR}
  SOURCE_DIR ${PLUTO_SOURCE_DIR}
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env bash ${PLUTO_CONFIGURE_COMMAND}
  BUILD_COMMAND make -j 4
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

