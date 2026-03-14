#pragma once
// TTTLoopDetector: online Test-Time Training auto-encoder for loop closure detection.
//
// Architecture:
//   Encoder: Linear(64,32) → ReLU → Linear(32,32)
//   Decoder: Linear(32,32) → ReLU → Linear(32,64)
//   Loss:    MSE reconstruction on 64-dim XFeat descriptors
//
// Thread model (two mutex-protected shared objects):
//
//   model_         — fast weights, ONLY touched by background thread_
//   model_snapshot_ — read-only clone for main-thread encoding, protected by snapshot_mtx_
//
// Every gradient update step the background thread clones model_ → model_snapshot_.
// Main thread acquires snapshot_mtx_ briefly for a forward pass (encode only, no grad).
//
// Experience replay (catastrophic-forgetting mitigation):
//   - Fixed-size FIFO deque of spatially diverse KF descriptor batches
//   - Spatial diversity: new entry accepted only if kf_position is > kSpatialDiversityThreshM
//     from all existing replay entries (meters in world frame)
//   - Each training step mixes cfg_.replay_mix_ratio fraction from replay buffer

#include <torch/torch.h>
#include <Eigen/Core>
#include <memory>
#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <string>

namespace deep {

static constexpr int   kTTTDescDim            = 64;
static constexpr int   kTTTLatentDim          = 32;
static constexpr int   kReplayCapacity        = 256;
static constexpr float kSpatialDiversityThreshM = 5.0f; // meters

// ---------------------------------------------------------------------------
// TTTAutoEncoderImpl — libtorch module (forward = reconstruct; encode = latent)
// TTTAutoEncoder     — ModuleHolder wrapper created by TORCH_MODULE macro
// ---------------------------------------------------------------------------
struct TTTAutoEncoderImpl : torch::nn::Module {
    torch::nn::Linear enc1{nullptr}, enc2{nullptr};
    torch::nn::Linear dec1{nullptr}, dec2{nullptr};

    TTTAutoEncoderImpl();

    // [N, 64] → [N, 32]  (no grad guard; caller must use torch::NoGradGuard if needed)
    torch::Tensor encode(torch::Tensor x);

    // [N, 64] → [N, 64]  full auto-encode (used during training)
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(TTTAutoEncoder);

// ---------------------------------------------------------------------------
// TTTUpdateJob — pushed by main thread per new keyframe
// ---------------------------------------------------------------------------
struct TTTUpdateJob {
    long kf_id;
    Eigen::Vector3d kf_position;              // world-frame (for spatial diversity)
    std::vector<std::vector<float>> descs;    // [N × 64] float32 descriptor batch
};

// ---------------------------------------------------------------------------
// SceneEmbedding — mean latent per KF, used for loop candidate ranking
// ---------------------------------------------------------------------------
struct SceneEmbedding {
    long kf_id;
    Eigen::Vector3d position;
    std::vector<float> embedding;  // [kTTTLatentDim]
};

// ---------------------------------------------------------------------------
// TTTLoopDetector
// ---------------------------------------------------------------------------
class TTTLoopDetector {
public:
    struct Config {
        float lr                 = 1e-3f;
        int   steps_per_update   = 3;     // gradient steps per KF batch
        float replay_mix_ratio   = 0.5f;  // fraction drawn from replay buffer
        int   top_k_candidates   = 5;     // candidates returned by query_loop_candidates
        float loop_sim_threshold = 0.75f; // cosine similarity floor to be a candidate
    };

    static std::unique_ptr<TTTLoopDetector> create(const Config& cfg);

    // Destructor signals shutdown_ and joins thread_.
    ~TTTLoopDetector();

    // NON-BLOCKING. Moves job into the input queue for async gradient update.
    void push_keyframe(TTTUpdateJob job);

    // THREAD-SAFE. Encodes query_descs using model_snapshot_ and returns KF ids
    // sorted by cosine similarity (descending). Acquires snapshot_mtx_ briefly.
    std::vector<long> query_loop_candidates(
        const std::vector<std::vector<float>>& query_descs,
        int top_k = -1   // -1 → use cfg_.top_k_candidates
    ) const;

    // NON-BLOCKING. Updates the world position of an existing scene embedding
    // (called after PGO corrects KF positions). Posts to input queue as a metadata op.
    void update_embedding_position(long kf_id, const Eigen::Vector3d& new_pos);

private:
    void thread_worker();

    // Returns true if pos is spatially diverse vs all replay buffer entries.
    // Must be called with replay_mtx_ held.
    bool is_spatially_diverse(const Eigen::Vector3d& pos) const;

    // Samples n rows from replay buffer (uniform random over entries).
    // Must be called with replay_mtx_ held.
    torch::Tensor sample_replay_batch(int n);

    // Converts descriptor list to [N, 64] float32 tensor.
    static torch::Tensor descs_to_tensor(const std::vector<std::vector<float>>& descs);

    Config cfg_;
    std::thread thread_;
    std::atomic<bool> shutdown_{false};

    // Fast-weights model — owned exclusively by thread_
    TTTAutoEncoder          model_{nullptr};
    std::unique_ptr<torch::optim::Adam> optimizer_;

    // Inference snapshot — cloned from model_ after every update cycle
    mutable TTTAutoEncoder  model_snapshot_{nullptr};
    mutable std::mutex      snapshot_mtx_;

    // Input queue
    std::mutex              input_mtx_;
    std::condition_variable input_cv_;
    std::deque<TTTUpdateJob> input_queue_;  // guarded by input_mtx_

    // Experience replay buffer: (world_position, descriptor_tensor [N,64])
    mutable std::mutex      replay_mtx_;
    std::deque<std::pair<Eigen::Vector3d, torch::Tensor>> replay_buffer_;

    // Scene embedding index — written by thread_, read by main via query_loop_candidates
    mutable std::mutex      embed_mtx_;
    std::vector<SceneEmbedding> scene_embeddings_;  // guarded by embed_mtx_
};

} // namespace deep
