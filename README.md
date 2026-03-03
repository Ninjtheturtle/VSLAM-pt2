# Monocular SLAM

Ground-up monocular Visual SLAM in C++17/CUDA targeting real-time performance on the KITTI odometry benchmark.

## Features

- ORB features + GPU-accelerated Hamming matching (custom CUDA kernel)
- Monocular initialization via Essential matrix + triangulation
- Constant-velocity motion model + PnP-RANSAC pose estimation (SQPNP)
- Sliding-window local bundle adjustment (Ceres, 10 KFs, Huber loss, analytical Jacobians)
- Relocalization against the global map on tracking loss
- Real-time 3D visualization via [Rerun](https://rerun.io/)

## Requirements

| Dependency | Version |
|---|---|
| MSVC | 19.x (VS 2022) |
| CUDA Toolkit | 12.x (Compute 8.6 — RTX 30xx/40xx) |
| CMake | 3.20+ |
| vcpkg | latest |
| OpenCV | 4.x |
| Ceres Solver | 2.x (eigensparse + schur) |
| Eigen3 | 3.4+ |
| Rerun SDK | 0.22.1 (auto-fetched via CMake) |

> For other GPU architectures, change `CMAKE_CUDA_ARCHITECTURES` in [CMakeLists.txt](CMakeLists.txt) (e.g. `75` for RTX 20xx, `89` for RTX 40xx).

## Building

```bat
:: 1. Install vcpkg dependencies
cd C:\Users\<you>\vcpkg
vcpkg install opencv4[core,features2d,calib3d,highgui] --triplet x64-windows
vcpkg install ceres[eigensparse,schur] --triplet x64-windows
vcpkg install eigen3 --triplet x64-windows
vcpkg integrate install

:: 2. Configure and build (VS Code)
Ctrl+Shift+P -> CMake: Configure
Ctrl+Shift+P -> CMake: Build

:: Or from a Developer Command Prompt
cmake -B build -DCMAKE_TOOLCHAIN_FILE=C:/Users/<you>/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Rerun SDK is downloaded automatically on first configure (~2–5 min).

## Dataset

Download the [KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) and place sequences under:

```
VSLAM/data/dataset/
  poses/00.txt          <- ground-truth (optional)
  sequences/00/
    calib.txt
    times.txt
    image_0/000000.png ...
```

## Usage

```bat
cd C:\...\VSLAM
build\Release\vslam.exe --sequence data/dataset/sequences/00
```

| Flag | Default | Description |
|---|---|---|
| `--sequence <path>` | required | KITTI sequence directory |
| `--start <N>` | 0 | First frame index |
| `--end <N>` | last | Last frame index |
| `--no-viz` | off | Disable Rerun |

Launch `rerun` before or after starting SLAM. The viewer connects to `127.0.0.1:9876` and shows:

- **Blue line** — estimated trajectory
- **Orange line** — KITTI ground-truth (if pose file found)
- **White cloud** — active map points
- **Purple dots** — current-frame keypoints (2D panel)

## Project Structure

```
VSLAM/
├── CMakeLists.txt
├── vcpkg.json
├── include/slam/       # camera, frame, map_point, map, tracker, local_ba, visualizer
├── include/cuda/       # hamming_matcher.cuh
├── src/                # .cpp implementations + main.cpp (KITTI loader)
├── cuda/               # hamming_matcher.cu (GPU kernel)
└── .vscode/            # tasks, launch, IntelliSense config
```

## Notes

- Target: >60 FPS on a laptop RTX 3050 (KITTI seq 00, 1241×376)
- BA runs asynchronously; the front-end tracker is never blocked
- No loop closure — long sequences will drift; DBoW is the natural next step