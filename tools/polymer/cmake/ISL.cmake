
set(ISL_INCLUDE_DIR "${POLYGEIST_PLUTO_DIR}/isl/install/include")
set(ISL_LIB_DIR "${POLYGEIST_PLUTO_DIR}/isl/install/lib")

add_library(libisl SHARED IMPORTED)
set_target_properties(libisl PROPERTIES IMPORTED_LOCATION "${ISL_LIB_DIR}/libisl.a")
