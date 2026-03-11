#pragma once

#include <cstdint>
#include <cstdio>

// ─── CUDA Error Checking ──────────────────────────────────────────────────────
//
// Usage:  CUDA_CHECK(cudaMalloc(&ptr, size));
//
#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "[CUDA ERROR] %s:%d — %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── ORB Descriptor Constants ─────────────────────────────────────────────────
//
// ORB descriptors are 256-bit = 32 bytes = 8 × uint32_t
//
static constexpr int kDescBytes  = 32;
static constexpr int kDescUint32 = 8;   // 32 bytes / 4 bytes per uint32_t
static constexpr int kMaxHamming = 256; // maximum possible Hamming distance

// ─── Public API ───────────────────────────────────────────────────────────────

/// @brief  GPU Hamming-distance nearest-neighbour matcher for ORB descriptors.
///
/// Uploads query and train descriptors to the GPU (if not already there),
/// runs the CUDA kernel, and writes the best match index + distance per query.
///
/// All pointers are *host* pointers — device allocation is handled internally.
///
/// @param h_query    Host array of query descriptors (N_q × 32 bytes, row-major)
/// @param h_train    Host array of train descriptors (N_t × 32 bytes, row-major)
/// @param N_q        Number of query descriptors
/// @param N_t        Number of train descriptors
/// @param h_best_idx Output: best match index for each query (length N_q)
/// @param h_best_dist Output: Hamming distance to best match (length N_q)
void cuda_match_hamming(
    const uint8_t* h_query,
    const uint8_t* h_train,
    int            N_q,
    int            N_t,
    int*           h_best_idx,
    int*           h_best_dist
);

/// @brief  Ratio-test filter on raw GPU matches.
///
/// Runs second-best matching in addition to best; retains match i only when
///   best_dist[i] / second_dist[i] < ratio_threshold.
///
/// @param h_query        Host array of query descriptors (N_q × 32 bytes)
/// @param h_train        Host array of train descriptors (N_t × 32 bytes)
/// @param N_q            Number of query descriptors
/// @param N_t            Number of train descriptors
/// @param ratio          Lowe ratio threshold (0.75 is standard)
/// @param h_best_idx     Output: best match index (-1 if rejected)
/// @param h_best_dist    Output: Hamming distance to best match
void cuda_match_hamming_ratio(
    const uint8_t* h_query,
    const uint8_t* h_train,
    int            N_q,
    int            N_t,
    float          ratio,
    int*           h_best_idx,
    int*           h_best_dist
);

/// @brief  Stereo epipolar matching with GPU Hamming distance + Lowe ratio test.
///
/// For rectified stereo pairs, epipolar lines are horizontal.  This matcher
/// restricts each query (left) descriptor to only consider train (right)
/// descriptors whose keypoint satisfies:
///   |y_query[q] - y_train[t]| <= epi_tol        (epipolar band)
///   d_min <= x_query[q] - x_train[t] <= d_max   (valid disparity range)
///
/// @param h_query    Host left  descriptors (N_q x 32 bytes)
/// @param h_train    Host right descriptors (N_t x 32 bytes)
/// @param N_q        Number of left  descriptors
/// @param N_t        Number of right descriptors
/// @param h_y_query  Left  keypoint y-coords [N_q]
/// @param h_y_train  Right keypoint y-coords [N_t]
/// @param h_x_query  Left  keypoint x-coords [N_q]
/// @param h_x_train  Right keypoint x-coords [N_t]
/// @param epi_tol    Max row difference in pixels (typically 2.0)
/// @param d_min      Minimum disparity in pixels (e.g. 5.0)
/// @param d_max      Maximum disparity in pixels (e.g. 300.0)
/// @param ratio      Lowe ratio threshold (0.75 is standard)
/// @param h_best_idx  Output: matched right index (-1 if rejected)
/// @param h_best_dist Output: Hamming distance to best match
void cuda_match_stereo_epipolar(
    const uint8_t* h_query,
    const uint8_t* h_train,
    int            N_q,
    int            N_t,
    const float*   h_y_query,
    const float*   h_y_train,
    const float*   h_x_query,
    const float*   h_x_train,
    float          epi_tol,
    float          d_min,
    float          d_max,
    float          ratio,
    int*           h_best_idx,
    int*           h_best_dist
);
