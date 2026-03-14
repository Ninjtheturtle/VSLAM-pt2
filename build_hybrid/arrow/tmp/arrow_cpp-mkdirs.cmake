# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp")
  file(MAKE_DIRECTORY "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp")
endif()
file(MAKE_DIRECTORY
  "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp-build"
  "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow"
  "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/tmp"
  "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp-stamp"
  "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src"
  "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp-stamp${cfgdir}") # cfgdir has leading slash
endif()
