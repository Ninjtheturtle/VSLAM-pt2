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

# ─── Optional: Deep Frontend (libtorch + TorchScript XFeat/LighterGlue/TTT) ──
# XFeat and LighterGlue are loaded as TorchScript .pt models via libtorch.
# TensorRT is NOT required for inference — only libtorch is needed.
option(ENABLE_DEEP_FRONTEND "Build hybrid XFeat/LighterGlue/TTT pipeline" OFF)

if(ENABLE_DEEP_FRONTEND)
  # --- libtorch (required) ---
  # Download from https://pytorch.org/ -> C++ / CUDA 12.4 / Release, unzip to C:/libtorch
  # Then set Torch_DIR or pass -DTorch_DIR=C:/Users/nengj/AppData/Local/Programs/Python/Python310/lib/site-packages/torch/share/cmake/Torch
  if(NOT DEFINED Torch_DIR)
    set(Torch_DIR "C:/Users/nengj/AppData/Local/Programs/Python/Python310/lib/site-packages/torch/share/cmake/Torch" CACHE PATH "libtorch cmake dir")
  endif()

  # CUDA 12.x removed the standalone nvToolsExt library (replaced by header-only NVTX3).
  # PyTorch cmake built with CUDA 11.x requires CUDA::nvToolsExt — provide a shim.
  find_package(CUDAToolkit QUIET)
  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED GLOBAL)
    if(CUDAToolkit_INCLUDE_DIRS)
      target_include_directories(CUDA::nvToolsExt INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
    endif()
    # nvToolsExt64_1 is bundled inside torch itself.
    # On MSVC, link the import .lib (not the .dll directly).
    find_file(_nvtools_lib "nvToolsExt64_1.lib"
      HINTS "${Torch_DIR}/../../../lib"
            "${CMAKE_PREFIX_PATH}/lib"
      NO_DEFAULT_PATH)
    if(_nvtools_lib)
      target_link_libraries(CUDA::nvToolsExt INTERFACE "${_nvtools_lib}")
    endif()
  endif()

  find_package(Torch REQUIRED)
  message(STATUS "Torch libraries   : ${TORCH_LIBRARIES}")
  message(STATUS "Torch install dir : ${TORCH_INSTALL_PREFIX}")

  # PyTorch cu118 cmake bakes in NvToolsExt paths that don't exist on CUDA 12.x.
  # (1) Strip non-existent NvToolsExt include dirs from torch imported targets.
  # (2) Redirect NvToolsExt lib to the copy bundled inside torch itself.
  set(_nvtools_lib_fallback
      "${TORCH_INSTALL_PREFIX}/lib/nvToolsExt64_1.lib")
  set(_nvtools_bad_path
      "C:/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64/nvToolsExt64_1.lib")
  foreach(_t torch torch_library c10 c10_cuda caffe2::cudart-stub)
    if(TARGET ${_t})
      # Fix include dirs
      get_target_property(_incs ${_t} INTERFACE_INCLUDE_DIRECTORIES)
      if(_incs)
        list(FILTER _incs EXCLUDE REGEX "NvToolsExt")
        set_target_properties(${_t} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_incs}")
      endif()
      # Fix link libraries: replace the missing NvToolsExt lib path
      get_target_property(_libs ${_t} INTERFACE_LINK_LIBRARIES)
      if(_libs)
        list(TRANSFORM _libs REPLACE
          "${_nvtools_bad_path}"
          "${_nvtools_lib_fallback}")
        set_target_properties(${_t} PROPERTIES INTERFACE_LINK_LIBRARIES "${_libs}")
      endif()
    endif()
  endforeach()
  # Also patch the TORCH_LIBRARIES variable itself (used directly in target_link_libraries)
  list(TRANSFORM TORCH_LIBRARIES REPLACE
    "${_nvtools_bad_path}"
    "${_nvtools_lib_fallback}")
  # Remove nvToolsExt64_1.dll from TORCH_LIBRARIES — cannot link .dll directly
  list(FILTER TORCH_LIBRARIES EXCLUDE REGEX "nvToolsExt64_1\\.dll")
  # Also remove .dll from imported target INTERFACE_LINK_LIBRARIES
  foreach(_t torch torch_library c10 c10_cuda caffe2::cudart-stub)
    if(TARGET ${_t})
      get_target_property(_libs2 ${_t} INTERFACE_LINK_LIBRARIES)
      if(_libs2)
        list(FILTER _libs2 EXCLUDE REGEX "nvToolsExt64_1\\.dll")
        set_target_properties(${_t} PROPERTIES INTERFACE_LINK_LIBRARIES "${_libs2}")
      endif()
    endif()
  endforeach()

  # --- TensorRT (optional, only for CUDA ANMS/L2 kernels) ---
  # Not required for XFeat/LighterGlue inference (those use TorchScript).
  if(NOT DEFINED TRT_ROOT)
    set(TRT_ROOT "C:/TensorRT" CACHE PATH "TensorRT root (optional, for headers only)")
  endif()
  find_path(TRT_INCLUDE_DIR NvInfer.h HINTS "${TRT_ROOT}/include")
  if(TRT_INCLUDE_DIR)
    message(STATUS "TensorRT headers  : ${TRT_INCLUDE_DIR}")
  else()
    message(STATUS "TensorRT headers not found (ANMS/L2 kernels use CUDA only)")
    set(TRT_INCLUDE_DIR "")
  endif()
endif()

# ─── CUDA Kernel Libraries ────────────────────────────────────────────────────
# Always built: Hamming matcher (ORB/stereo epipolar)
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
target_compile_options(hamming_matcher PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

# Deep frontend CUDA kernels (L2 FP16 matcher + ANMS) — only when enabled
if(ENABLE_DEEP_FRONTEND)
  add_library(deep_kernels STATIC
    cuda/l2_matcher.cu
    cuda/anms_kernel.cu
  )
  target_include_directories(deep_kernels PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${TRT_INCLUDE_DIR}
  )
  set_target_properties(deep_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
  target_compile_options(deep_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4996>
  )
endif()

# ─── Main VSLAM Executable ───────────────────────────────────────────────────
set(VSLAM_SOURCES
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

if(ENABLE_DEEP_FRONTEND)
  list(APPEND VSLAM_SOURCES
    src/xfeat_extractor.cu
    src/lighterglue_async.cpp
    src/ttt_autoencoder.cpp
    src/semi_dense_disparity.cpp
  )
endif()

add_executable(vslam ${VSLAM_SOURCES})

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

if(ENABLE_DEEP_FRONTEND)
  target_compile_definitions(vslam PRIVATE ENABLE_DEEP_FRONTEND)
  # TRT headers only needed if found (not required for TorchScript inference)
  if(TRT_INCLUDE_DIR)
    target_include_directories(vslam PRIVATE ${TRT_INCLUDE_DIR})
  endif()
  target_link_libraries(vslam PRIVATE
    deep_kernels
    ${TORCH_LIBRARIES}
    CUDA::cudart
  )
  # Make torch DLLs available next to the exe.
  # torch_cpu.dll (~250 MB) and torch_cuda.dll (~1.2 GB) are too large to copy on
  # space-constrained disks; create hard links instead (zero extra disk usage).
  # Smaller DLLs are copied normally.
  if(MSVC)
    set(_torch_lib_dir "${TORCH_INSTALL_PREFIX}/lib")
    file(GLOB _all_torch_dlls "${_torch_lib_dir}/*.dll")
    set(_large_dlls "torch_cpu.dll" "torch_cuda.dll")
    foreach(_dll ${_all_torch_dlls})
      get_filename_component(_dll_name "${_dll}" NAME)
      set(_dst "$<TARGET_FILE_DIR:vslam>/${_dll_name}")
      if(_dll_name IN_LIST _large_dlls)
        # Hard-link the large DLLs (requires same filesystem; no disk copy)
        add_custom_command(TARGET vslam POST_BUILD
          COMMAND if not exist "${_dst}" mklink /H "${_dst}" "${_dll}"
          VERBATIM
        )
      else()
        add_custom_command(TARGET vslam POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "${_dst}"
        )
      endif()
    endforeach()
  endif()
endif()

# ─── MSVC-specific flags ─────────────────────────────────────────────────────
if(MSVC)
  # Use $<COMPILE_LANGUAGE:CXX> guard so MSVC flags are NOT passed to nvcc
  # when compiling .cu files (nvcc doesn't understand /MP, /W3, etc.)
  target_compile_options(vslam PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3>           # Warning level 3
    $<$<COMPILE_LANGUAGE:CXX>:/MP>           # Multi-processor compilation
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc>         # Exception handling model
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj>       # Large object file (needed for Ceres templates)
    $<$<COMPILE_LANGUAGE:CXX>:/wd4244>       # Suppress conversion warnings from Eigen
    $<$<COMPILE_LANGUAGE:CXX>:/wd4305>       # Suppress truncation warnings from Eigen
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
if(ENABLE_DEEP_FRONTEND)
  message(STATUS "  Deep frontend       : ENABLED (libtorch TorchScript)")
  message(STATUS "    Torch_DIR         : ${Torch_DIR}")
else()
  message(STATUS "  Deep frontend       : DISABLED (ORB pipeline)")
endif()
message(STATUS "=================================")
