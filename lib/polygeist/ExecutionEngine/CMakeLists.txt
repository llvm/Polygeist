# TODO we do not support cross compilation currently

if(POLYGEIST_ENABLE_CUDA)
  find_package(CUDA)
  enable_language(CUDA)

  find_library(CUDA_RUNTIME_LIBRARY cuda)

  # Bitcode lib wrapper
  find_program(CLANG_TOOL clang PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)

  set(source_directory ${CMAKE_CURRENT_SOURCE_DIR})
  set(src_files
    ${source_directory}/CudaRuntimeWrappers.cpp
    )

  set(bc_flags -c -emit-llvm -std=c++17 -fvisibility=hidden
    -O3 -nocudalib -Xclang -no-opaque-pointers
    )

  foreach(src ${src_files})
    get_filename_component(infile ${src} ABSOLUTE)
    get_filename_component(filename ${src} NAME)
    set(inc_outfile "${filename}.bin.h")
    set(bc_outfile "${filename}.bc")

    # TODO I think CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES is supposed to be a
    # list but seems to contain only one dir so this usage currently works - fix
    add_custom_command(OUTPUT ${bc_outfile}
      COMMAND ${CLANG_TOOL}
      ${bc_flags}
      ${infile} -o ${bc_outfile}
      -I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
      -DPOLYGEIST_PGO_DEFAULT_DATA_DIR="${POLYGEIST_PGO_DEFAULT_DATA_DIR}"
      -DPOLYGEIST_PGO_ALTERNATIVE_ENV_VAR="${POLYGEIST_PGO_ALTERNATIVE_ENV_VAR}"
      -DPOLYGEIST_PGO_DATA_DIR_ENV_VAR="${POLYGEIST_PGO_DATA_DIR_ENV_VAR}"
      DEPENDS ${infile}
      COMMENT "Building LLVM bitcode ${bc_outfile}"
      VERBATIM
    )
    add_custom_target(${bc_outfile}_target DEPENDS ${bc_outfile})
    add_custom_command(OUTPUT ${inc_outfile}
      COMMAND ${XXD_BIN} -i ${bc_outfile} ${inc_outfile}
      DEPENDS ${bc_outfile}
      COMMENT "Generating C header ${inc_outfile}"
      VERBATIM
    )
    add_custom_target(execution_engine_cuda_wrapper_binary_include DEPENDS ${inc_outfile})
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${bc_outfile})
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${inc_outfile})

  endforeach()


endif()
if(POLYGEIST_ENABLE_ROCM)

  # Bitcode lib wrapper
  find_program(CLANG_TOOL clang PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)

  set(source_directory ${CMAKE_CURRENT_SOURCE_DIR})
  set(src_files
    ${source_directory}/RocmRuntimeWrappers.cpp
    )

  set(bc_flags -c -emit-llvm -std=c++17 -fvisibility=hidden
    -O3
    #-nocudalib
    -Xclang -no-opaque-pointers
    )

  foreach(src ${src_files})
    get_filename_component(infile ${src} ABSOLUTE)
    get_filename_component(filename ${src} NAME)
    set(inc_outfile "${filename}.bin.h")
    set(bc_outfile "${filename}.bc")

    add_custom_command(OUTPUT ${bc_outfile}
      COMMAND ${CLANG_TOOL}
      ${bc_flags}
      ${infile} -o ${bc_outfile}
      #-I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
      -DPOLYGEIST_PGO_DEFAULT_DATA_DIR="${POLYGEIST_PGO_DEFAULT_DATA_DIR}"
      -DPOLYGEIST_PGO_ALTERNATIVE_ENV_VAR="${POLYGEIST_PGO_ALTERNATIVE_ENV_VAR}"
      -DPOLYGEIST_PGO_DATA_DIR_ENV_VAR="${POLYGEIST_PGO_DATA_DIR_ENV_VAR}"
      DEPENDS ${infile}
      COMMENT "Building LLVM bitcode ${bc_outfile}"
      VERBATIM
    )
    add_custom_target(${bc_outfile}_target DEPENDS ${bc_outfile})
    add_custom_command(OUTPUT ${inc_outfile}
      COMMAND ${XXD_BIN} -i ${bc_outfile} ${inc_outfile}
      DEPENDS ${bc_outfile}
      COMMENT "Generating C header ${inc_outfile}"
      VERBATIM
    )
    add_custom_target(execution_engine_rocm_wrapper_binary_include DEPENDS ${inc_outfile})
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${bc_outfile})
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${inc_outfile})

  endforeach()
endif()
