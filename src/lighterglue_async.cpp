// lighterglue_async.cpp — background libtorch TorchScript LighterGlue matcher.
//
// Thread synchronization:
//   Main thread: submit_job() / try_get_result() — never blocks
//   Background thread: waits on input_cv_, runs torch inference, writes to latest_result_

#include "../include/deep/lighterglue_async.hpp"

#include <torch/script.h>
#include <torch/cuda.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <vector>

namespace deep {

std::unique_ptr<LighterGlueAsync> LighterGlueAsync::create(const Config& cfg) {
    auto lg = std::unique_ptr<LighterGlueAsync>(new LighterGlueAsync());
    lg->cfg_ = cfg;
    lg->thread_ = std::thread(&LighterGlueAsync::thread_worker, lg.get());
    return lg;
}

LighterGlueAsync::~LighterGlueAsync() {
    shutdown_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(input_mtx_);
        pending_job_.reset();
    }
    input_cv_.notify_one();
    if (thread_.joinable()) thread_.join();
}

bool LighterGlueAsync::submit_job(RelocJob job) {
    if (job_pending_.load(std::memory_order_acquire)) return false;
    {
        std::lock_guard<std::mutex> lk(input_mtx_);
        pending_job_ = std::move(job);
    }
    job_pending_.store(true, std::memory_order_release);
    input_cv_.notify_one();
    return true;
}

std::optional<RelocResult> LighterGlueAsync::try_get_result() {
    std::lock_guard<std::mutex> lk(output_mtx_);
    auto r = std::move(latest_result_);
    latest_result_.reset();
    return r;
}

void LighterGlueAsync::thread_worker() {
    // Load TorchScript model on this thread's CUDA device
    torch::jit::script::Module model;
    try {
        fprintf(stderr, "[LG] Loading model: %s\n", cfg_.engine_path.c_str());
        model = torch::jit::load(cfg_.engine_path, torch::kCUDA);
        model.eval();
        fprintf(stderr, "[LG] Model loaded OK\n");
    } catch (const c10::Error& e) {
        fprintf(stderr, "[LG] c10::Error loading model %s: %s\n", cfg_.engine_path.c_str(), e.what());
        return;
    } catch (const std::exception& e) {
        fprintf(stderr, "[LG] std::exception loading model %s: %s\n", cfg_.engine_path.c_str(), e.what());
        return;
    } catch (...) {
        fprintf(stderr, "[LG] Unknown exception loading model %s\n", cfg_.engine_path.c_str());
        return;
    }

    while (!shutdown_.load(std::memory_order_acquire)) {
        std::optional<RelocJob> job;
        {
            std::unique_lock<std::mutex> lk(input_mtx_);
            input_cv_.wait(lk, [this]{
                return pending_job_.has_value() || shutdown_.load();
            });
            if (shutdown_.load()) break;
            job = std::move(pending_job_);
            pending_job_.reset();
        }
        if (!job.has_value()) continue;

        const RelocJob& J = *job;
        int N_q = (int)J.query_kp_x.size();
        int N_t = (int)J.candidate_kp_x.size();

        if (N_q == 0 || N_t == 0) {
            job_pending_.store(false, std::memory_order_release);
            continue;
        }

        // Build interleaved keypoint tensors [1, N, 2] float32
        std::vector<float> kp0(N_q * 2), kp1(N_t * 2);
        for (int i = 0; i < N_q; ++i) { kp0[i*2] = J.query_kp_x[i]; kp0[i*2+1] = J.query_kp_y[i]; }
        for (int i = 0; i < N_t; ++i) { kp1[i*2] = J.candidate_kp_x[i]; kp1[i*2+1] = J.candidate_kp_y[i]; }

        // Build FP16 -> FP32 descriptor tensors [1, N, 64] float32
        std::vector<float> d0(N_q * 64), d1(N_t * 64);
        for (int i = 0; i < N_q * 64; ++i) d0[i] = __half2float(J.query_descs[i]);
        for (int i = 0; i < N_t * 64; ++i) d1[i] = __half2float(J.candidate_descs[i]);

        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        auto t_kp0  = torch::from_blob(kp0.data(), {1, N_q, 2}, opts).to(torch::kCUDA);
        auto t_d0   = torch::from_blob(d0.data(),  {1, N_q, 64}, opts).to(torch::kCUDA);
        auto t_kp1  = torch::from_blob(kp1.data(), {1, N_t, 2}, opts).to(torch::kCUDA);
        auto t_d1   = torch::from_blob(d1.data(),  {1, N_t, 64}, opts).to(torch::kCUDA);

        torch::NoGradGuard ng;
        auto out     = model.forward({t_kp0, t_d0, t_kp1, t_d1}).toTuple();
        auto matches = out->elements()[0].toTensor().contiguous().cpu();  // (1,N_q) int32
        auto scores  = out->elements()[1].toTensor().contiguous().cpu();  // (1,N_q) float32

        const int*   m_ptr = matches.data_ptr<int>();
        const float* s_ptr = scores.data_ptr<float>();

        RelocResult result;
        result.job_id          = J.job_id;
        result.query_frame_id  = J.query_frame_id;
        result.candidate_kf_id = J.candidate_kf_id;

        for (int i = 0; i < N_q; ++i) {
            if (m_ptr[i] >= 0 && s_ptr[i] >= cfg_.min_confidence) {
                result.matches.push_back({i, m_ptr[i], s_ptr[i]});
            }
        }
        result.success = ((int)result.matches.size() >= cfg_.min_matches);

        {
            std::lock_guard<std::mutex> lk(output_mtx_);
            latest_result_ = std::move(result);
        }
        job_pending_.store(false, std::memory_order_release);
    }
}

} // namespace deep
