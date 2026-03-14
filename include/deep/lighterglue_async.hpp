#pragma once
// LighterGlueAsync: background TensorRT FP16 keypoint matcher for relocalization.
//
// Thread model:
//   - submit_job()     called from MAIN THREAD — non-blocking (returns false if busy)
//   - try_get_result() called from MAIN THREAD — non-blocking (returns nullopt if not ready)
//   - thread_worker()  runs in background; owns its own cudaStream_t (lg_stream_)
//
// CUDA stream policy:
//   - Main thread: stream 0 (default)
//   - LighterGlue thread: lg_stream_ created inside thread_worker()
//   - No cross-stream memory races: descriptor data is deep-copied into RelocJob
//     before submit, so the background thread never accesses main-thread pinned buffers.
//
// Drop policy: only one job at a time. If is_idle()==false, submit_job() returns false
// and the caller must try again next frame. This keeps the main loop non-blocking even
// if relocalization takes multiple frames.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <memory>
#include <vector>
#include <optional>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <string>

namespace deep {

// ---------------------------------------------------------------------------
// RelocJob — submitted by main thread, consumed by LighterGlue thread
// ---------------------------------------------------------------------------
struct RelocJob {
    long job_id         = 0;
    long query_frame_id = 0;
    long candidate_kf_id = 0;

    // Deep copies of FP16 descriptor arrays (NOT borrowed pinned pointers).
    // Layout: N × kXFeatDescDim, row-major.
    std::vector<__half> query_descs;      // [N_q × 64]
    std::vector<float>  query_kp_x;       // [N_q]
    std::vector<float>  query_kp_y;       // [N_q]

    std::vector<__half> candidate_descs;  // [N_t × 64]
    std::vector<float>  candidate_kp_x;   // [N_t]
    std::vector<float>  candidate_kp_y;   // [N_t]
};

// ---------------------------------------------------------------------------
// RelocMatch — one correspondence in a RelocResult
// ---------------------------------------------------------------------------
struct RelocMatch {
    int   query_idx;    // index into query frame keypoints
    int   train_idx;    // index into candidate KF keypoints
    float confidence;   // LighterGlue output probability ∈ [0, 1]
};

// ---------------------------------------------------------------------------
// RelocResult — produced by LighterGlue thread, polled by main thread
// ---------------------------------------------------------------------------
struct RelocResult {
    long job_id          = 0;
    long query_frame_id  = 0;
    long candidate_kf_id = 0;
    std::vector<RelocMatch> matches;
    bool success = false;  // true iff |matches| >= min_matches
};

// ---------------------------------------------------------------------------
// LighterGlueAsync
// ---------------------------------------------------------------------------
class LighterGlueAsync {
public:
    struct Config {
        std::string engine_path;      // path to TorchScript .pt model file
        int   min_matches     = 30;   // threshold to declare relocalization success
        float min_confidence  = 0.5f; // per-match confidence cutoff
    };

    static std::unique_ptr<LighterGlueAsync> create(const Config& cfg);

    // Destructor signals shutdown_ and joins the background thread.
    ~LighterGlueAsync();

    // NON-BLOCKING. Moves job into the pending slot.
    // Returns false if a job is already in-flight (caller should try next frame).
    bool submit_job(RelocJob job);

    // NON-BLOCKING. Moves the latest finished result out (consumes it).
    // Returns std::nullopt if no result is ready yet.
    std::optional<RelocResult> try_get_result();

    // True when no job is currently being processed.
    bool is_idle() const { return !job_pending_.load(std::memory_order_acquire); }

private:
    void thread_worker();

    Config cfg_;
    std::thread thread_;
    std::atomic<bool> shutdown_{false};
    std::atomic<bool> job_pending_{false};

    // --- Input slot (capacity 1) ---
    std::mutex             input_mtx_;
    std::condition_variable input_cv_;
    std::optional<RelocJob> pending_job_;   // guarded by input_mtx_

    // --- Output slot (capacity 1) ---
    std::mutex             output_mtx_;
    std::optional<RelocResult> latest_result_; // guarded by output_mtx_
};

} // namespace deep
