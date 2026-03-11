cmake_minimum_required(VERSION 3.20)
project(VSLAM LANGUAGES CXX CUDA)

# ─── Language Standards ──────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─── Workaround: CMake Tools passes -DCMAKE_CXX_COMPILER=... explicitly,     ─
# which sets CMAKE_CXX_COMPILER_FORCED=1 and skips compiler feature detection, ─
# leaving CMAKE_CXX_COMPILE_FEATURES empty. Manually populate for MSVC 19.x.  ─
if(MSVC AND NOT CMAKE_CXX_COMPILE_FEATURES)
  set(CMAKE_CXX_COMPILE_FEATURES
      "cxx_std_98;cxx_std_11;cxx_std_14;cxx_std_17;cxx_std_20;cxx_std_23"
      CACHE INTERNAL "" FORCE)
  set(CMAKE_CXX98_COMPILE_FEATURES  "cxx_std_98"  CACHE INTERNAL "" FORCE)
  set(CMAKE_CXX11_COMPILE_FEATURES  "cxx_std_11"  CACHE INTERNAL "" FORCE)
  set(CMAKE_CXX14_COMPILE_FEATURES  "cxx_std_14"  CACHE INTERNAL "" FORCE)
  set(CMAKE_CXX17_COMPILE_FEATURES  "cxx_std_17"  CACHE INTERNAL "" FORCE)
  set(CMAKE_CXX20_COMPILE_FEATURES  "cxx_std_20"  CACHE INTERNAL "" FORCE)
  set(CMAKE_CXX23_COMPILE_FEATURES  "cxx_std_23"  CACHE INTERNAL "" FORCE)
endif()

# ─── CUDA Architecture (RTX 3050 Laptop = Ampere SM86) ───────────────────────
set(CMAKE_CUDA_ARCHITECTURES 86)

# ─── Build Type Defaults ─────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ─── vcpkg Dependencies ──────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core features2d calib3d highgui)
find_package(Ceres  REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

# ─── Rerun SDK via FetchContent ──────────────────────────────────────────────
# Arrow patch: Rerun passes CMAKE_TOOLCHAIN_FILE to Arrow's ExternalProject,
# causing vcpkg to try to install Arrow's full vcpkg.json (aws-sdk-cpp, grpc,
# protobuf, …) which are NOT needed for Rerun's minimal build and fail on
# Windows. We patch download_and_build_arrow.cmake to strip that line and add
# -DVCPKG_MANIFEST_MODE=OFF -DARROW_S3=OFF in its place.
include(FetchContent)
FetchContent_Declare(rerun_sdk
  URL https://github.com/rerun-io/rerun/releases/download/0.22.1/rerun_cpp_sdk.zip
  DOWNLOAD_EXTRACT_TIMESTAMP NEW
)
FetchContent_GetProperties(rerun_sdk)
if(NOT rerun_sdk_POPULATED)
  FetchContent_Populate(rerun_sdk)
endif()

# Apply patch unconditionally (safe to re-apply; guarded by content check).
set(_arrow_cmake "${rerun_sdk_SOURCE_DIR}/download_and_build_arrow.cmake")
if(EXISTS "${_arrow_cmake}")
  file(READ "${_arrow_cmake}" _content)
  if(_content MATCHES "DCMAKE_TOOLCHAIN_FILE=")
    string(REGEX REPLACE
      "-DCMAKE_TOOLCHAIN_FILE=[^\r\n]*"
      "-DVCPKG_MANIFEST_MODE=OFF\n        -DARROW_S3=OFF"
      _content "${_content}")
    file(WRITE "${_arrow_cmake}" "${_content}")
    message(STATUS "Arrow patch applied: disabled vcpkg toolchain for Arrow sub-cmake")
  endif()
  unset(_content)
endif()
unset(_arrow_cmake)

add_subdirectory("${rerun_sdk_SOURCE_DIR}" "${rerun_sdk_BINARY_DIR}")

# ─── CUDA Hamming Matcher (static library) ───────────────────────────────────
add_library(hamming_matcher STATIC
  cuda/hamming_matcher.cu
)
target_include_directories(hamming_matcher PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
set_target_properties(hamming_matcher PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
# Suppress MSVC warnings in CUDA device code
target_compile_options(hamming_matcher PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

# ─── Main VSLAM Executable ───────────────────────────────────────────────────
add_executable(vslam
  src/main.cpp
  src/camera.cpp
  src/frame.cpp
  src/map_point.cpp
  src/map.cpp
  src/tracker.cpp
  src/local_ba.cpp
  src/pose_graph.cpp
  src/visualizer.cpp
)
target_include_directories(vslam PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(vslam PRIVATE
  hamming_matcher
  ${OpenCV_LIBS}
  Ceres::ceres
  Eigen3::Eigen
  rerun_sdk
)

# ─── MSVC-specific flags ─────────────────────────────────────────────────────
if(MSVC)
  target_compile_options(vslam PRIVATE
    /W3           # Warning level 3
    /MP           # Multi-processor compilation
    /EHsc         # Exception handling model
    /bigobj       # Large object file (needed for Ceres templates)
    /wd4244       # Suppress conversion warnings from Eigen
    /wd4305       # Suppress truncation warnings from Eigen
  )
  # Ensure CUDA runtime is available on PATH at runtime
  set_target_properties(vslam PROPERTIES
    VS_DEBUGGER_ENVIRONMENT "PATH=${CUDA_TOOLKIT_ROOT_DIR}/bin;$ENV{PATH}"
  )
endif()

# ─── Installation ────────────────────────────────────────────────────────────
install(TARGETS vslam DESTINATION bin)

# ─── Print Configuration Summary ─────────────────────────────────────────────
message(STATUS "=== VSLAM Build Configuration ===")
message(STATUS "  Build type          : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA architectures  : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  OpenCV version      : ${OpenCV_VERSION}")
message(STATUS "  Ceres version       : ${CERES_VERSION}")
message(STATUS "  Eigen3 version      : ${Eigen3_VERSION}")
message(STATUS "=================================")
