#
#    Locates pet
#
#    Output:
#
#    PET_FOUND - TRUE/FALSE - Whether the library has been found.
#        If FALSE, all other output variables are not defined.
#
#    PET_INCLUDE_PATH/PET_LIBRARY
#
set(PET_FOUND FALSE)

message(STATUS "Looking for PET")
find_path(PET_PATH_PETROOT pet.h PATHS $ENV{HOME}/pet/install/include NO_DEFAULT_PATH)

if(PET_PATH_PETROOT)
    get_filename_component(PET_PATH ${PET_PATH_PETROOT}/.. ABSOLUTE)
    set(PET_FOUND TRUE)
endif(PET_PATH_PETROOT)

if(PET_FOUND)
  set(PET_INCLUDE_PATH ${PET_PATH}/include)
  find_library(PET_LIBRARY pet PATHS ${PET_PATH}/lib)
else(PET_FOUND)
  message(FATAL_ERROR "Could not find pet")
endif(PET_FOUND)
