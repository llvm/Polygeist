
if(POLYGEIST_ENABLE_CUDA)
  find_package(CUDA)
  enable_language(CUDA)

  add_mlir_library(polygeist_cuda_runtime
    SHARED
    CudaRuntimeWrappers.cpp

    EXCLUDE_FROM_LIBMLIR
  )

  find_library(CUDA_RUNTIME_LIBRARY cuda)

  set_property(TARGET polygeist_cuda_runtime PROPERTY CXX_STANDARD 14)
  target_include_directories(polygeist_cuda_runtime
    PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
  target_link_libraries(polygeist_cuda_runtime
    PRIVATE
    ${CUDA_RUNTIME_LIBRARY}
  )

endif()
