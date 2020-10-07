# otherwise those tools added by add_llvm_tool won't be installed.
set(LLVM_BUILD_TOOLS ON)

add_subdirectory(polymer-translate)
add_subdirectory(polymer-opt)
