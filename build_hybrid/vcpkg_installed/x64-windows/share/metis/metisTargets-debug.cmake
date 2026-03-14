#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "metis" for configuration "Debug"
set_property(TARGET metis APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(metis PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/debug/lib/metis.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/bin/metis.dll"
  )

list(APPEND _cmake_import_check_targets metis )
list(APPEND _cmake_import_check_files_for_metis "${_IMPORT_PREFIX}/debug/lib/metis.lib" "${_IMPORT_PREFIX}/debug/bin/metis.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
