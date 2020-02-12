#
#    Locates isl
#
#    Output:
#
#    ISL_FOUND - TRUE/FALSE - Whether the library has been found.
#        If FALSE, all other output variables are not defined.
#
#    ISL_INCLUDE_PATH/ISL_LIBRARY
#
set(ISL_FOUND FALSE)

message(STATUS "Looking for ISL")
find_path(ISL_PATH_ISLROOT isl-noexceptions.h PATHS $ENV{HOME}/isl/install/include/isl NO_DEFAULT_PATH)

if(ISL_PATH_ISLROOT)
    get_filename_component(ISL_PATH ${ISL_PATH_ISLROOT}/.. ABSOLUTE)
    get_filename_component(ISL_PATH_PREV ${ISL_PATH}../.. ABSOLUTE)
    set(ISL_FOUND TRUE)
endif(ISL_PATH_ISLROOT)

if(ISL_FOUND)
  set(ISL_INCLUDE_PATH ${ISL_PATH_PREV}/include)
  find_library(ISL_LIBRARY isl PATHS ${ISL_PATH_PREV}/lib)
else(ISL_FOUND)
  message(FATAL_ERROR "Could not find isl")
endif(ISL_FOUND)

