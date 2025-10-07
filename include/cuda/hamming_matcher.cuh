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
