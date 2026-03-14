#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Ceres::ceres" for configuration "Debug"
set_property(TARGET Ceres::ceres APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(Ceres::ceres PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/debug/lib/ceres-debug.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "metis"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/bin/ceres-debug.dll"
  )

list(APPEND _cmake_import_check_targets Ceres::ceres )
list(APPEND _cmake_import_check_files_for_Ceres::ceres "${_IMPORT_PREFIX}/debug/lib/ceres-debug.lib" "${_IMPORT_PREFIX}/debug/bin/ceres-debug.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
