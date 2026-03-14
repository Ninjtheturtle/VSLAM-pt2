// anms_kernel.cu — GPU Adaptive Non-Maximal Suppression on XFeat heatmap.
//
// Two-pass design:
//   Pass 1: 2D NMS via shared-memory tile max-filter — marks local maxima above threshold.
//           Each block processes a TILE_W × TILE_H region, loading a (TILE_W + 2*R) ×
//           (TILE_H + 2*R) halo into shared memory.
//   Pass 2: Stream-compact candidate (x, y, score) triples into a flat array using
//           atomic counter, then a bitonic sort on score to keep the top max_kps.

#include "../include/cuda/anms_kernel.cuh"
#include <cuda_runtime.h>
#include <float.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

// ---------------------------------------------------------------------------
// Pass 1: NMS tile kernel
// Marks pixels that are local maxima within a (2*R+1)² neighbourhood.
// Output: d_candidate_mask[y*W + x] = 1 if candidate, 0 otherwise.
// ---------------------------------------------------------------------------
static constexpr int TILE_W = 16;
static constexpr int TILE_H = 16;

__global__ void nms_tile_kernel(
    const float* __restrict__ d_heatmap,
    int H, int W,
    float min_response,
    int R,                  // NMS radius
    uint8_t* __restrict__ d_mask)
{
    const int halo = R;
    const int shw  = TILE_W + 2 * halo;
    const int shh  = TILE_H + 2 * halo;

    extern __shared__ float sh[];  // [shh × shw]

    // Global pixel this thread is responsible for (centre of tile)
    int gx = blockIdx.x * TILE_W + threadIdx.x - halo;
    int gy = blockIdx.y * TILE_H + threadIdx.y - halo;

    // Load halo region into shared memory
    int lid = threadIdx.y * blockDim.x + threadIdx.x;
    int sh_total = shw * shh;
    for (int i = lid; i < sh_total; i += blockDim.x * blockDim.y) {
        int sy = i / shw;
        int sx = i % shw;
        int gy2 = blockIdx.y * TILE_H + sy - halo;
        int gx2 = blockIdx.x * TILE_W + sx - halo;
        float val = 0.0f;
        if (gy2 >= 0 && gy2 < H && gx2 >= 0 && gx2 < W)
            val = d_heatmap[gy2 * W + gx2];
        sh[i] = val;
    }
    __syncthreads();

    // Only valid (non-halo) threads do the NMS check
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (tx < halo || tx >= TILE_W + halo) return;
    if (ty < halo || ty >= TILE_H + halo) return;

    int px = gx + halo;  // pixel column in original image
    int py = gy + halo;  // pixel row
    if (px < 0 || px >= W || py < 0 || py >= H) return;

    float centre = sh[ty * shw + tx];
    if (centre < min_response) { d_mask[py * W + px] = 0; return; }

    // Check all neighbours in [ty-R, ty+R] x [tx-R, tx+R]
    bool is_max = true;
    for (int dy = -R; dy <= R && is_max; ++dy) {
        for (int dx = -R; dx <= R && is_max; ++dx) {
            if (dx == 0 && dy == 0) continue;
            if (sh[(ty + dy) * shw + (tx + dx)] >= centre)
                is_max = false;
        }
    }
    d_mask[py * W + px] = is_max ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Pass 2a: Stream-compact candidates using atomic counter
// ---------------------------------------------------------------------------
__global__ void compact_kernel(
    const float*  __restrict__ d_heatmap,
    const uint8_t* __restrict__ d_mask,
    int H, int W,
    float* __restrict__ d_cand_x,
    float* __restrict__ d_cand_y,
    float* __restrict__ d_cand_score,
    int*  __restrict__ d_count,
    int   max_kps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * W) return;
    if (!d_mask[idx]) return;

    int y = idx / W;
    int x = idx % W;
    int slot = atomicAdd(d_count, 1);
    if (slot >= max_kps) return;  // hard cap (very rare)

    d_cand_x[slot]     = (float)x;
    d_cand_y[slot]     = (float)y;
    d_cand_score[slot] = d_heatmap[idx];
}

// ---------------------------------------------------------------------------
// cuda_anms — host wrapper
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
    cudaStream_t stream)
{
    // Allocate temporaries
    uint8_t* d_mask      = nullptr;
    float*   d_cand_x    = nullptr;
    float*   d_cand_y    = nullptr;
    float*   d_cand_sc   = nullptr;
    int*     d_count     = nullptr;

    cudaMalloc(&d_mask,    H * W * sizeof(uint8_t));
    cudaMalloc(&d_cand_x,  max_kps * sizeof(float));
    cudaMalloc(&d_cand_y,  max_kps * sizeof(float));
    cudaMalloc(&d_cand_sc, max_kps * sizeof(float));
    cudaMalloc(&d_count,   sizeof(int));
    cudaMemsetAsync(d_count, 0, sizeof(int), stream);

    // Pass 1 — NMS
    int halo = nms_radius;
    int sh_bytes = (TILE_W + 2*halo) * (TILE_H + 2*halo) * sizeof(float);
    dim3 tile_block(TILE_W + 2*halo, TILE_H + 2*halo, 1);
    dim3 tile_grid((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H, 1);
    nms_tile_kernel<<<tile_grid, tile_block, sh_bytes, stream>>>(
        d_heatmap, H, W, min_response_thresh, nms_radius, d_mask);

    // Pass 2a — compact
    int n_pixels = H * W;
    int compact_threads = 256;
    int compact_blocks  = (n_pixels + compact_threads - 1) / compact_threads;
    compact_kernel<<<compact_blocks, compact_threads, 0, stream>>>(
        d_heatmap, d_mask, H, W,
        d_cand_x, d_cand_y, d_cand_sc,
        d_count, max_kps);

    // Synchronize before reading count and sorting
    cudaStreamSynchronize(stream);

    int n_cands = 0;
    cudaMemcpy(&n_cands, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    n_cands = (n_cands < max_kps) ? n_cands : max_kps;

    // Pass 2b — sort by score descending, keep top max_kps
    if (n_cands > 0) {
        // Build index array and sort by score (thrust on default stream after sync)
        thrust::device_ptr<float> sc_ptr(d_cand_sc);
        thrust::device_ptr<float> x_ptr(d_cand_x);
        thrust::device_ptr<float> y_ptr(d_cand_y);

        // Sort all three arrays by score descending using zip iterator
        thrust::sort_by_key(
            thrust::device,
            sc_ptr, sc_ptr + n_cands,
            thrust::make_zip_iterator(thrust::make_tuple(x_ptr, y_ptr)),
            thrust::greater<float>()
        );

        int n_out = (n_cands < max_kps) ? n_cands : max_kps;
        cudaMemcpy(d_out_x,      d_cand_x,  n_out * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_out_y,      d_cand_y,  n_out * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_out_scores, d_cand_sc, n_out * sizeof(float), cudaMemcpyDeviceToDevice);
        n_cands = n_out;
    }

    cudaFree(d_mask);
    cudaFree(d_cand_x);
    cudaFree(d_cand_y);
    cudaFree(d_cand_sc);
    cudaFree(d_count);

    return n_cands;
}
