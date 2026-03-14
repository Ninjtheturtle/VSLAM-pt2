# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp-build/src/xsimd_ep")
  file(MAKE_DIRECTORY "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp-build/src/xsimd_ep")
endif()
file(MAKE_DIRECTORY
  "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp-build/src/xsimd_ep-build"
  "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp-build"
  "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp-build/tmp"
  "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp-build/src/xsimd_ep-stamp"
  "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp-build/src"
  "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp-build/src/xsimd_ep-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp-build/src/xsimd_ep-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/nengj/OneDrive/Desktop/VSLAM/build_hybrid/arrow/src/arrow_cpp-build/src/xsimd_ep-stamp${cfgdir}") # cfgdir has leading slash
endif()
