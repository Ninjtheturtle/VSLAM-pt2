// ttt_autoencoder.cpp — Online TTT auto-encoder with experience replay.
//
// Thread model:
//   model_         — owned by background thread_; never touched by main thread
//   model_snapshot_ — cloned after each update; main thread reads under snapshot_mtx_
//   scene_embeddings_ — written by background thread, read by main (embed_mtx_)
//   replay_buffer_    — written & read by background thread (replay_mtx_)
//
// Catastrophic-forgetting mitigation (Experience Replay):
//   Each training step mixes cfg_.replay_mix_ratio fraction from the replay buffer.
//   Entries admitted only when spatially distant from all existing entries
//   (prevents the buffer being dominated by one loop of the trajectory).

#include "../include/deep/ttt_autoencoder.hpp"
#include <torch/torch.h>
#include <algorithm>
#include <random>
#include <cmath>

namespace deep {

// ---------------------------------------------------------------------------
// TTTAutoEncoder module
// ---------------------------------------------------------------------------
TTTAutoEncoderImpl::TTTAutoEncoderImpl() {
    enc1 = register_module("enc1", torch::nn::Linear(kTTTDescDim,  kTTTLatentDim));
    enc2 = register_module("enc2", torch::nn::Linear(kTTTLatentDim, kTTTLatentDim));
    dec1 = register_module("dec1", torch::nn::Linear(kTTTLatentDim, kTTTLatentDim));
    dec2 = register_module("dec2", torch::nn::Linear(kTTTLatentDim, kTTTDescDim));
}

torch::Tensor TTTAutoEncoderImpl::encode(torch::Tensor x) {
    x = torch::relu(enc1->forward(x));
    x = enc2->forward(x);   // no activation on bottleneck — unit-norm it externally
    return x;
}

torch::Tensor TTTAutoEncoderImpl::forward(torch::Tensor x) {
    auto z = encode(x);
    auto r = torch::relu(dec1->forward(z));
    return dec2->forward(r);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
torch::Tensor TTTLoopDetector::descs_to_tensor(
    const std::vector<std::vector<float>>& descs)
{
    if (descs.empty()) return {};
    int N = (int)descs.size();
    int D = (int)descs[0].size();
    auto t = torch::zeros({N, D});
    auto acc = t.accessor<float, 2>();
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j)
            acc[i][j] = descs[i][j];
    return t;
}

bool TTTLoopDetector::is_spatially_diverse(const Eigen::Vector3d& pos) const {
    // Called with replay_mtx_ held
    for (auto& [rpos, _] : replay_buffer_) {
        if ((pos - rpos).norm() < kSpatialDiversityThreshM) return false;
    }
    return true;
}

torch::Tensor TTTLoopDetector::sample_replay_batch(int n) {
    // Called with replay_mtx_ held
    if (replay_buffer_.empty()) return {};
    std::vector<torch::Tensor> parts;
    int total = 0;
    // Randomly sample entries until we have ≥ n rows
    std::vector<int> idxs;
    for (int i = 0; i < (int)replay_buffer_.size(); ++i) idxs.push_back(i);
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::shuffle(idxs.begin(), idxs.end(), rng);
    for (int idx : idxs) {
        parts.push_back(replay_buffer_[idx].second);
        total += (int)replay_buffer_[idx].second.size(0);
        if (total >= n) break;
    }
    if (parts.empty()) return {};
    auto batch = torch::cat(parts, 0);
    // Subsample to exactly n rows if we have more
    if (batch.size(0) > n) {
        auto perm = torch::randperm(batch.size(0)).slice(0, 0, n);
        batch = batch.index_select(0, perm);
    }
    return batch;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
std::unique_ptr<TTTLoopDetector> TTTLoopDetector::create(const Config& cfg) {
    auto det = std::unique_ptr<TTTLoopDetector>(new TTTLoopDetector());
    det->cfg_ = cfg;
    det->model_         = TTTAutoEncoder();
    det->model_snapshot_ = TTTAutoEncoder();

    // Copy initial weights to snapshot
    {
        torch::NoGradGuard ng;
        auto src_params = det->model_->parameters();
        auto dst_params = det->model_snapshot_->parameters();
        for (size_t i = 0; i < src_params.size(); ++i)
            dst_params[i].copy_(src_params[i]);
    }

    det->optimizer_ = std::make_unique<torch::optim::Adam>(
        det->model_->parameters(),
        torch::optim::AdamOptions(cfg.lr));

    det->thread_ = std::thread(&TTTLoopDetector::thread_worker, det.get());
    return det;
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
TTTLoopDetector::~TTTLoopDetector() {
    shutdown_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(input_mtx_);
        input_queue_.clear();
    }
    input_cv_.notify_one();
    if (thread_.joinable()) thread_.join();
}

// ---------------------------------------------------------------------------
// push_keyframe (main thread, non-blocking)
// ---------------------------------------------------------------------------
void TTTLoopDetector::push_keyframe(TTTUpdateJob job) {
    {
        std::lock_guard<std::mutex> lk(input_mtx_);
        // Cap input queue to 8 pending KFs to avoid unbounded memory growth
        if (input_queue_.size() >= 8) input_queue_.pop_front();
        input_queue_.push_back(std::move(job));
    }
    input_cv_.notify_one();
}

// ---------------------------------------------------------------------------
// update_embedding_position (main thread, non-blocking)
// ---------------------------------------------------------------------------
void TTTLoopDetector::update_embedding_position(long kf_id, const Eigen::Vector3d& new_pos) {
    std::lock_guard<std::mutex> lk(embed_mtx_);
    for (auto& e : scene_embeddings_) {
        if (e.kf_id == kf_id) { e.position = new_pos; break; }
    }
}

// ---------------------------------------------------------------------------
// query_loop_candidates (main thread, thread-safe read)
// ---------------------------------------------------------------------------
std::vector<long> TTTLoopDetector::query_loop_candidates(
    const std::vector<std::vector<float>>& query_descs,
    int top_k) const
{
    if (top_k < 0) top_k = cfg_.top_k_candidates;

    torch::Tensor q_emb;
    {
        torch::NoGradGuard ng;
        std::lock_guard<std::mutex> lk(snapshot_mtx_);
        auto t = descs_to_tensor(query_descs);
        if (!t.defined() || t.size(0) == 0) return {};
        q_emb = model_snapshot_->encode(t).mean(0);  // [32]
    }
    // Normalize
    q_emb = torch::nn::functional::normalize(q_emb.unsqueeze(0), torch::nn::functional::NormalizeFuncOptions().dim(1)).squeeze(0);

    // Cosine similarity against all scene embeddings
    std::vector<std::pair<float, long>> sims;
    {
        std::lock_guard<std::mutex> lk(embed_mtx_);
        for (auto& se : scene_embeddings_) {
            auto e = torch::from_blob(
                const_cast<float*>(se.embedding.data()),
                {kTTTLatentDim}, torch::kFloat32).clone();
            e = torch::nn::functional::normalize(e.unsqueeze(0), torch::nn::functional::NormalizeFuncOptions().dim(1)).squeeze(0);
            float sim = torch::dot(q_emb, e).item<float>();
            if (sim >= cfg_.loop_sim_threshold)
                sims.push_back({sim, se.kf_id});
        }
    }

    std::sort(sims.begin(), sims.end(),
              [](auto& a, auto& b){ return a.first > b.first; });

    std::vector<long> result;
    for (int i = 0; i < (int)sims.size() && i < top_k; ++i)
        result.push_back(sims[i].second);
    return result;
}

// ---------------------------------------------------------------------------
// thread_worker — background gradient updates
// ---------------------------------------------------------------------------
void TTTLoopDetector::thread_worker() {
    fprintf(stderr, "[TTT] Background thread started\n");
    try {
    while (!shutdown_.load(std::memory_order_acquire)) {
        TTTUpdateJob job;
        {
            std::unique_lock<std::mutex> lk(input_mtx_);
            input_cv_.wait(lk, [this]{
                return !input_queue_.empty() || shutdown_.load();
            });
            if (shutdown_.load()) break;
            job = std::move(input_queue_.front());
            input_queue_.pop_front();
        }

        // Build new-feature tensor
        torch::Tensor new_batch = descs_to_tensor(job.descs);  // [N, 64]
        if (!new_batch.defined() || new_batch.size(0) == 0) continue;

        // Build mixed training batch
        torch::Tensor train_batch;
        {
            std::lock_guard<std::mutex> lk(replay_mtx_);
            int replay_n = (int)(new_batch.size(0) * cfg_.replay_mix_ratio
                                  / (1.0f - cfg_.replay_mix_ratio + 1e-6f));
            torch::Tensor rb = sample_replay_batch(replay_n);
            if (rb.defined() && rb.size(0) > 0)
                train_batch = torch::cat({new_batch, rb}, 0);
            else
                train_batch = new_batch;
        }

        // TTT gradient steps (MSE reconstruction)
        model_->train();
        for (int step = 0; step < cfg_.steps_per_update; ++step) {
            optimizer_->zero_grad();
            auto recon = model_->forward(train_batch);
            auto loss  = torch::mse_loss(recon, train_batch.detach());
            loss.backward();
            optimizer_->step();
        }
        model_->eval();

        // Update experience replay buffer
        {
            std::lock_guard<std::mutex> lk(replay_mtx_);
            if (is_spatially_diverse(job.kf_position)) {
                replay_buffer_.push_back({job.kf_position, new_batch.detach().clone()});
                if ((int)replay_buffer_.size() > kReplayCapacity)
                    replay_buffer_.pop_front();
            }
        }

        // Update scene embedding for this KF
        {
            torch::NoGradGuard ng;
            auto emb = model_->encode(new_batch).mean(0);  // [32]
            std::vector<float> emb_vec(emb.data_ptr<float>(),
                                       emb.data_ptr<float>() + kTTTLatentDim);
            SceneEmbedding se;
            se.kf_id     = job.kf_id;
            se.position  = job.kf_position;
            se.embedding = std::move(emb_vec);
            {
                std::lock_guard<std::mutex> lk(embed_mtx_);
                auto it = std::find_if(scene_embeddings_.begin(), scene_embeddings_.end(),
                    [&](const SceneEmbedding& e){ return e.kf_id == job.kf_id; });
                if (it != scene_embeddings_.end()) *it = se;
                else scene_embeddings_.push_back(se);
            }
        }

        // Clone fast weights to inference snapshot
        {
            torch::NoGradGuard ng;
            std::lock_guard<std::mutex> lk(snapshot_mtx_);
            auto src = model_->parameters();
            auto dst = model_snapshot_->parameters();
            for (size_t i = 0; i < src.size(); ++i)
                dst[i].copy_(src[i]);
        }
    }
    } catch (const std::exception& e) {
        fprintf(stderr, "[TTT] Exception in thread_worker: %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "[TTT] Unknown exception in thread_worker\n");
    }
    fprintf(stderr, "[TTT] Background thread exiting\n");
}

} // namespace deep
