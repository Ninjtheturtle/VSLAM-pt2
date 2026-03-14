#pragma once
// anms_kernel.cuh — GPU Adaptive Non-Maximal Suppression on a VRAM heatmap.
//
// Operates entirely in device memory; no host round-trip between XFeat inference
// and keypoint selection.
//
// Algorithm (two-pass):
//   Pass 1 — non-maximum suppression: for each pixel, mark as candidate if it is
//             the local maximum in a (2*nms_radius+1)² neighbourhood AND response
//             exceeds min_response.  Implemented as a 2D sliding-window max filter
//             using shared memory tiles.
//   Pass 2 — top-K selection: stream-compact candidate list, then partial-sort by
//             response to extract the top max_kps entries.
//             Uses thrust::sort on device or a custom bitonic sort for portability.
//
// The result (d_out_x, d_out_y, d_out_scores) is written to pre-allocated device
// arrays.  The host receives the actual count via a device→host 4-byte copy of
// d_out_count (async on stream; synchronize before reading count).

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// cuda_anms
//
// Inputs:
//   d_heatmap          — [H × W] float32 device pointer (row-major)
//   H, W               — heatmap spatial dimensions
//   min_response_thresh— floor below which candidates are suppressed (e.g. 0.005)
//   max_kps            — output cap (e.g. 2000)
//   nms_radius         — local-max window radius in pixels (e.g. 4)
//   stream             — CUDA stream
//
// Outputs (pre-allocated device arrays, each [max_kps]):
//   d_out_x, d_out_y   — float32 pixel coordinates (column, row order)
//   d_out_scores       — float32 heatmap response values
//
// Returns:
//   Number of keypoints actually written (≤ max_kps).
//   The return value is obtained via a synchronous device→host copy internally
//   (cudaMemcpy of a single int).  If asynchronous count is needed, pass a
//   device int* and call the async variant below.
// ---------------------------------------------------------------------------
int cuda_anms(
    const float* d_heatmap,
    int H, int W,
    float min_response_thresh,
    int max_kps,
    int nms_radius,
    float* d_out_x,
    float* d_out_y,
    float* d_out_scores,
    cudaStream_t stream = 0
);
