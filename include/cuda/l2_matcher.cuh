#pragma once
// l2_matcher.cuh — FP16 L2-distance nearest-neighbour matching with Lowe ratio test.
//
// Kernel design:
//   Grid  : (N_q, 1, 1) — one block per query descriptor
//   Block : 256 threads
//   Each thread strides over the train set, maintaining local best/second-best L2².
//   Warp-level reduction via __shfl_down_sync() propagates the minimum to lane 0.
//   half2 vectorization halves the memory-bandwidth pressure for 64-dim FP16 descriptors.
//
// Pseudo-confidence output:
//   w = clamp(1.0f - best_dist / (ratio * second_dist), 0.1f, 1.0f)
//   Used as confidence weight in ConfidenceWeightedReprojectionCost when LighterGlue
//   is not active (temporal tracking phase).

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// cuda_match_l2_fp16
//
// Inputs (device pointers):
//   d_query   — [N_q × D] FP16, row-major
//   d_train   — [N_t × D] FP16, row-major
//   N_q, N_t  — descriptor counts
//   D         — descriptor dimension (must be even for half2; typically 64)
//   ratio     — Lowe ratio threshold (e.g. 0.9f)
//   stream    — CUDA stream for async dispatch (default: stream 0)
//
// Outputs (device pointers, pre-allocated by caller):
//   d_best_idx    — [N_q] int:   best train index per query (-1 if ratio test failed)
//   d_best_dist   — [N_q] float: best L2 distance (before rejection)
//   d_pseudo_conf — [N_q] float: pseudo-confidence weight ∈ [0.1, 1.0]
//
// All outputs are written asynchronously on `stream`.
// Call cudaStreamSynchronize(stream) or cudaDeviceSynchronize() before reading.
// ---------------------------------------------------------------------------
void cuda_match_l2_fp16(
    const __half* d_query,
    const __half* d_train,
    int N_q, int N_t, int D,
    float ratio,
    int*   d_best_idx,
    float* d_best_dist,
    float* d_pseudo_conf,
    cudaStream_t stream = 0
);
