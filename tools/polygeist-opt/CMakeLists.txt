get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
		MLIRPolygeist
		MLIRPolygeistTransforms
        )
add_llvm_executable(polygeist-opt polygeist-opt.cpp)

llvm_update_compile_flags(polygeist-opt)
target_link_libraries(polygeist-opt PRIVATE ${LIBS})
