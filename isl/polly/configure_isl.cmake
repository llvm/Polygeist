# ISL configuration borrowed from Polly.

# Determine compiler characteristics
include(CheckCSourceCompiles)

# Like check_c_source_compiles, but sets the result to either
# 0 (error while compiling) or 1 (compiled successfully)
# Required for compatibility with autotool's AC_CHECK_DECLS
function (check_c_source_compiles_numeric _prog _var)
  check_c_source_compiles("${_prog}" "${_var}")
  if ("${${_var}}")
    set("${_var}" 1 PARENT_SCOPE)
  else ()
    set("${_var}" 0 PARENT_SCOPE)
  endif ()
endfunction ()

# Check for the existance of a type
function (check_c_type_exists _type _files _variable)
  set(_includes "")
  foreach (file_name ${_files})
    set(_includes "${_includes}#include<${file_name}>\n")
  endforeach()
  check_c_source_compiles("
  ${_includes}
  ${_type} typeVar;
  int main() {
  return 0;
  }
  " ${_variable})
endfunction ()


check_c_source_compiles("
int func(void) __attribute__((__warn_unused_result__));
int main() { return 0; }
" HAS_ATTRIBUTE_WARN_UNUSED_RESULT)
set(GCC_WARN_UNUSED_RESULT)
if (HAS_ATTRIBUTE_WARN_UNUSED_RESULT)
  set(GCC_WARN_UNUSED_RESULT "__attribute__((__warn_unused_result__))")
endif ()

check_c_source_compiles("
__attribute__ ((unused)) static void foo(void);
int main() { return 0; }
" HAVE___ATTRIBUTE__)


check_c_source_compiles_numeric("
#include <strings.h>
int main() { (void)ffs(0); return 0; }
" HAVE_DECL_FFS)

check_c_source_compiles_numeric("
int main() { (void)__builtin_ffs(0); return 0; }
" HAVE_DECL___BUILTIN_FFS)

check_c_source_compiles_numeric("
#include <intrin.h>
int main() { (void)_BitScanForward(NULL, 0); return 0; }
" HAVE_DECL__BITSCANFORWARD)

if (NOT HAVE_DECL_FFS AND
    NOT HAVE_DECL___BUILTIN_FFS AND
    NOT HAVE_DECL__BITSCANFORWARD)
  message(FATAL_ERROR "No ffs implementation found")
endif ()


check_c_source_compiles_numeric("
#include <strings.h>
int main() { (void)strcasecmp(\"\", \"\"); return 0; }
" HAVE_DECL_STRCASECMP)

check_c_source_compiles_numeric("
#include <string.h>
int main() { (void)_stricmp(\"\", \"\"); return 0; }
" HAVE_DECL__STRICMP)

if (NOT HAVE_DECL_STRCASECMP AND NOT HAVE_DECL__STRICMP)
  message(FATAL_ERROR "No strcasecmp implementation found")
endif ()


check_c_source_compiles_numeric("
#include <strings.h>
int main() { (void)strncasecmp(\"\", \"\", 0); return 0; }
" HAVE_DECL_STRNCASECMP)

check_c_source_compiles_numeric("
#include <string.h>
int main() { (void)_strnicmp(\"\", \"\", 0); return 0; }
" HAVE_DECL__STRNICMP)

if (NOT HAVE_DECL_STRNCASECMP AND NOT HAVE_DECL__STRNICMP)
  message(FATAL_ERROR "No strncasecmp implementation found")
endif ()


check_c_source_compiles_numeric("
#include <stdio.h>
int main() { snprintf((void*)0, 0, \" \"); return 0; }
" HAVE_DECL_SNPRINTF)

check_c_source_compiles_numeric("
#include <stdio.h>
int main() { _snprintf((void*)0, 0, \" \"); return 0; }
" HAVE_DECL__SNPRINTF)

if (NOT HAVE_DECL_SNPRINTF AND NOT HAVE_DECL__SNPRINTF)
  message(FATAL_ERROR "No snprintf implementation found")
endif ()


check_c_type_exists(uint8_t "" HAVE_UINT8T)
check_c_type_exists(uint8_t "stdint.h" HAVE_STDINT_H)
check_c_type_exists(uint8_t "inttypes.h" HAVE_INTTYPES_H)
check_c_type_exists(uint8_t "sys/types.h" HAVE_SYS_INTTYPES_H)
if (HAVE_UINT8T)
  set(INCLUDE_STDINT_H "")
elseif (HAVE_STDINT_H)
  set(INCLUDE_STDINT_H "#include <stdint.h>")
elseif (HAVE_INTTYPES_H)
  set(INCLUDE_STDINT_H "#include <inttypes.h>")
elseif (HAVE_SYS_INTTYPES_H)
  set(INCLUDE_STDINT_H "#include <sys/inttypes.h>")
else ()
  message(FATAL_ERROR "No stdint.h or compatible found")
endif ()

# Write configure result
# configure_file(... COPYONLY) avoids that the time stamp changes if the file is identical
file(WRITE "${ISL_BINARY_DIR}/gitversion.h.tmp"
  "#define GIT_HEAD_ID \"${ISL_GIT_HEAD_ID}\"")
configure_file("${ISL_BINARY_DIR}/gitversion.h.tmp"
  "${ISL_BINARY_DIR}/gitversion.h" COPYONLY)

file(WRITE "${ISL_BINARY_DIR}/include/isl/stdint.h.tmp"
  "${INCLUDE_STDINT_H}\n")
configure_file("${ISL_BINARY_DIR}/include/isl/stdint.h.tmp"
  "${ISL_BINARY_DIR}/include/isl/stdint.h" COPYONLY)

configure_file("polly/isl_config.h.cmake" "${ISL_BINARY_DIR}/isl_config.h")
#configure_file("isl_srcdir.c.cmake" "${ISL_BINARY_DIR}/isl_srcdir.c")
