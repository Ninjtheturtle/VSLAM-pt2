// hamming_matcher.cu
//
// CUDA kernel for brute-force Hamming-distance matching of ORB descriptors.
//
// Algorithm overview
// ──────────────────
// Each CUDA block handles exactly ONE query descriptor.  All threads in the
// block collaborate to compute Hamming distances from that query to every
// train descriptor, then perform a parallel reduction to find the closest match.
//
// Shared-memory layout (per block)
// ─────────────────────────────────
//   shm[0..7]   — 8 × uint32_t = 32 bytes = one query descriptor (256 bits)
//   shm[8+tid]  — per-thread minimum distance accumulator (for reduction)
//
// Performance choices
// ───────────────────
//   • 256 threads/block — fills 4 warps; gives ~100% occupancy on SM86
//   • Each thread owns one train descriptor per loop iteration
//   • Inner loop: 8 × __popc(q[k] ^ t[k]) — fully pipelined on Ampere
//   • Warp-level reduction using __shfl_down_sync before shared-memory stage

#include "cuda/hamming_matcher.cuh"
#include <cstring>
#include <limits>
#include <vector>

// ─── Kernel: Best-Match (single nearest neighbour) ───────────────────────────

static constexpr int BLOCK_SIZE = 256; // threads per block

/// shared memory layout per block:
///   [0 .. DESC_U32-1]          : query descriptor (8 uint32s)
///   [DESC_U32 .. DESC_U32+BLOCK_SIZE-1] : per-thread (dist, idx) packed as uint64
///
/// We pack (distance << 32 | index) so that a single uint64 min-reduction gives
/// the correct result (lower distance wins; ties broken by lower index).

__global__ void hamming_match_kernel(
    const uint32_t* __restrict__ d_query,  // (N_q × 8) uint32 row-major
    const uint32_t* __restrict__ d_train,  // (N_t × 8) uint32 row-major
    int              N_t,
    int*  __restrict__ d_best_idx,
    int*  __restrict__ d_best_dist
)
{
    // ── Load query descriptor into shared memory ──────────────────────────────
    extern __shared__ uint32_t shm[];
    // shm[0..7] = query descriptor
    // shm[8..8+BLOCK_SIZE-1] = reduction scratch (reused as uint32 distances)

    const int qid = blockIdx.x;            // one block per query
    const int tid = threadIdx.x;

    // First 8 threads copy one uint32 each of the query descriptor
    if (tid < kDescUint32) {
        shm[tid] = d_query[qid * kDescUint32 + tid];
    }
    __syncthreads();

    // ── Each thread scans train descriptors in strides of BLOCK_SIZE ──────────
    int   local_min_dist = kMaxHamming + 1;
    int   local_min_idx  = -1;

    for (int t = tid; t < N_t; t += BLOCK_SIZE) {
        const uint32_t* tptr = d_train + t * kDescUint32;

        // Compute Hamming distance: sum of popcount(q XOR t) over 8 words
        int dist = 0;
        #pragma unroll
        for (int k = 0; k < kDescUint32; ++k) {
            dist += __popc(shm[k] ^ tptr[k]);
        }

        if (dist < local_min_dist) {
            local_min_dist = dist;
            local_min_idx  = t;
        }
    }

    // ── Warp-level reduction (find min across 32 threads) ────────────────────
    // Pack (dist, idx) into uint64 so a single min() works correctly
    // (distance in high 32 bits, index in low 32 bits)
    uint64_t val = ((uint64_t)(uint32_t)local_min_dist << 32)
                 | ((uint64_t)(uint32_t)local_min_idx);

    // Butterfly reduction within warp
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        uint64_t other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (other < val) val = other;
    }

    // Lane 0 of each warp writes to shared memory
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    __shared__ uint64_t warp_vals[BLOCK_SIZE / 32];  // 8 warps max

    if (lane_id == 0) {
        warp_vals[warp_id] = val;
    }
    __syncthreads();

    // ── Final reduction across warps (thread 0 only) ─────────────────────────
    if (tid == 0) {
        const int n_warps = BLOCK_SIZE / 32;
        uint64_t best = warp_vals[0];
        for (int w = 1; w < n_warps; ++w) {
            if (warp_vals[w] < best) best = warp_vals[w];
        }
        d_best_dist[qid] = (int)(best >> 32);
        d_best_idx [qid] = (int)(best & 0xFFFFFFFFu);
    }
}

// ─── Kernel: Two-Nearest-Neighbours (for Lowe ratio test) ────────────────────

__global__ void hamming_match_ratio_kernel(
    const uint32_t* __restrict__ d_query,
    const uint32_t* __restrict__ d_train,
    int              N_t,
    float            ratio,
    int*  __restrict__ d_best_idx,
    int*  __restrict__ d_best_dist
)
{
    extern __shared__ uint32_t shm[];

    const int qid = blockIdx.x;
    const int tid = threadIdx.x;

    if (tid < kDescUint32) {
        shm[tid] = d_query[qid * kDescUint32 + tid];
    }
    __syncthreads();

    int local_d1 = kMaxHamming + 1, local_i1 = -1;  // best
    int local_d2 = kMaxHamming + 1;                  // second best

    for (int t = tid; t < N_t; t += BLOCK_SIZE) {
        const uint32_t* tptr = d_train + t * kDescUint32;
        int dist = 0;
        #pragma unroll
        for (int k = 0; k < kDescUint32; ++k) {
            dist += __popc(shm[k] ^ tptr[k]);
        }
        if (dist < local_d1) {
            local_d2 = local_d1;
            local_d1 = dist; local_i1 = t;
        } else if (dist < local_d2) {
            local_d2 = dist;
        }
    }

    // Warp-reduction for best and second-best simultaneously
    // We track (d1, i1, d2) per thread; reduce to per-warp minimum
    __shared__ int  warp_d1[BLOCK_SIZE / 32];
    __shared__ int  warp_i1[BLOCK_SIZE / 32];
    __shared__ int  warp_d2[BLOCK_SIZE / 32];

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Warp shuffle reduction for best
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        int od1 = __shfl_down_sync(0xFFFFFFFF, local_d1, offset);
        int oi1 = __shfl_down_sync(0xFFFFFFFF, local_i1, offset);
        int od2 = __shfl_down_sync(0xFFFFFFFF, local_d2, offset);
        if (od1 < local_d1) {
            // incoming best is better — demote our best to second
            if (local_d1 < local_d2) local_d2 = local_d1;
            local_d1 = od1; local_i1 = oi1;
        } else {
            // our best survives — check if incoming helps second
            if (od1 < local_d2) local_d2 = od1;
        }
        if (od2 < local_d2) local_d2 = od2;
    }

    if (lane_id == 0) {
        warp_d1[warp_id] = local_d1;
        warp_i1[warp_id] = local_i1;
        warp_d2[warp_id] = local_d2;
    }
    __syncthreads();

    if (tid == 0) {
        const int n_warps = BLOCK_SIZE / 32;
        int best_d1 = warp_d1[0], best_i1 = warp_i1[0], best_d2 = warp_d2[0];
        for (int w = 1; w < n_warps; ++w) {
            int wd1 = warp_d1[w], wi1 = warp_i1[w], wd2 = warp_d2[w];
            if (wd1 < best_d1) {
                if (best_d1 < best_d2) best_d2 = best_d1;
                best_d1 = wd1; best_i1 = wi1;
            } else {
                if (wd1 < best_d2) best_d2 = wd1;
            }
            if (wd2 < best_d2) best_d2 = wd2;
        }

        // Apply Lowe ratio test
        bool accepted = (best_d1 < ratio * best_d2) && (best_i1 >= 0);
        d_best_idx [qid] = accepted ? best_i1 : -1;
        d_best_dist[qid] = accepted ? best_d1 : kMaxHamming;
    }
}

// ─── Host Wrapper: cuda_match_hamming ─────────────────────────────────────────

void cuda_match_hamming(
    const uint8_t* h_query,
    const uint8_t* h_train,
    int            N_q,
    int            N_t,
    int*           h_best_idx,
    int*           h_best_dist
)
{
    if (N_q == 0 || N_t == 0) return;

    const size_t q_bytes = (size_t)N_q * kDescBytes;
    const size_t t_bytes = (size_t)N_t * kDescBytes;

    // Allocate device memory
    uint32_t *d_query = nullptr, *d_train = nullptr;
    int      *d_idx   = nullptr, *d_dist  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_query, q_bytes));
    CUDA_CHECK(cudaMalloc(&d_train, t_bytes));
    CUDA_CHECK(cudaMalloc(&d_idx,   sizeof(int) * N_q));
    CUDA_CHECK(cudaMalloc(&d_dist,  sizeof(int) * N_q));

    // Copy descriptors to device
    CUDA_CHECK(cudaMemcpy(d_query, h_query, q_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_train, h_train, t_bytes, cudaMemcpyHostToDevice));

    // Shared memory: 8 uint32 for query desc + warp_vals array (8 uint64)
    // Total: 32 + 64 = 96 bytes per block (well within limits)
    const size_t shm_bytes = kDescUint32 * sizeof(uint32_t);

    // Launch: one block per query descriptor
    dim3 grid(N_q, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    hamming_match_kernel<<<grid, block, shm_bytes>>>(
        d_query, d_train, N_t, d_idx, d_dist
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_best_idx,  d_idx,  sizeof(int) * N_q, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_best_dist, d_dist, sizeof(int) * N_q, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_train));
    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaFree(d_dist));
}

// ─── Host Wrapper: cuda_match_hamming_ratio ───────────────────────────────────

void cuda_match_hamming_ratio(
    const uint8_t* h_query,
    const uint8_t* h_train,
    int            N_q,
    int            N_t,
    float          ratio,
    int*           h_best_idx,
    int*           h_best_dist
)
{
    if (N_q == 0 || N_t == 0) return;

    const size_t q_bytes = (size_t)N_q * kDescBytes;
    const size_t t_bytes = (size_t)N_t * kDescBytes;

    uint32_t *d_query = nullptr, *d_train = nullptr;
    int      *d_idx   = nullptr, *d_dist  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_query, q_bytes));
    CUDA_CHECK(cudaMalloc(&d_train, t_bytes));
    CUDA_CHECK(cudaMalloc(&d_idx,   sizeof(int) * N_q));
    CUDA_CHECK(cudaMalloc(&d_dist,  sizeof(int) * N_q));

    CUDA_CHECK(cudaMemcpy(d_query, h_query, q_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_train, h_train, t_bytes, cudaMemcpyHostToDevice));

    const size_t shm_bytes = kDescUint32 * sizeof(uint32_t);
    dim3 grid(N_q, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    hamming_match_ratio_kernel<<<grid, block, shm_bytes>>>(
        d_query, d_train, N_t, ratio, d_idx, d_dist
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_best_idx,  d_idx,  sizeof(int) * N_q, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_best_dist, d_dist, sizeof(int) * N_q, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_train));
    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaFree(d_dist));
}
