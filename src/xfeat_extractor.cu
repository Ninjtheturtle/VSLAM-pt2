// xfeat_extractor.cu — libtorch XFeat backbone inference + C++ NMS + grid_sample.
//
// Model file: models/xfeat.pt  (exported by setup/03b_export_torchscript.py)
// Model I/O:
//   Input : (1, 1, H_32, W_32) float32 CUDA, normalized [0, 1]
//            where H_32 = (H//32)*32, W_32 = (W//32)*32
//   Output: tuple(M1, K1h, H1)
//            M1:  (1, 64, H_32/8, W_32/8)  L2-normalised descriptor maps
//            K1h: (1,  1, H_32,   W_32)    keypoint heatmap
//            H1:  (1,  1, H_32/8, W_32/8)  reliability map
//
// NMS and descriptor sampling are done here in C++/libtorch to avoid
// the torch.jit.trace dynamic-shape issues in XFeat's Python NMS.

#include "../include/deep/xfeat_extractor.hpp"

#include <torch/script.h>
#include <torch/cuda.h>
#include <torch/nn/functional.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <opencv2/imgproc.hpp>

#include <stdexcept>
#include <algorithm>
#include <cassert>

namespace deep {

// FP32 -> FP16 conversion kernel
__global__ static void fp32_to_fp16_kernel(
    const float* __restrict__ src,
    __half*      __restrict__ dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

struct XFeatExtractor::Impl {
    Config cfg;
    torch::jit::script::Module model;

    // Precomputed backbone input dimensions (multiples of 32)
    int H32 = 0, W32 = 0;

    struct PinnedSlot {
        __half*     ptr   = nullptr;
        cudaEvent_t ready = nullptr;
    };
    static constexpr int kNumSlots = 3;
    PinnedSlot slots[kNumSlots];
    int write_slot = 0;
    __half* d_descs_fp16 = nullptr;

    void alloc_buffers() {
        int n = cfg.max_keypoints * kXFeatDescDim;
        cudaMalloc(&d_descs_fp16, n * sizeof(__half));
        for (int i = 0; i < kNumSlots; ++i) {
            cudaMallocHost(&slots[i].ptr, n * sizeof(__half));
            cudaEventCreateWithFlags(&slots[i].ready, cudaEventDisableTiming);
        }
    }
    void free_buffers() {
        cudaFree(d_descs_fp16);
        for (int i = 0; i < kNumSlots; ++i) {
            cudaFreeHost(slots[i].ptr);
            cudaEventDestroy(slots[i].ready);
        }
    }
};

std::unique_ptr<XFeatExtractor> XFeatExtractor::create(const Config& cfg) {
    auto ext = std::unique_ptr<XFeatExtractor>(new XFeatExtractor());
    ext->impl_ = std::make_unique<Impl>();
    Impl& m = *ext->impl_;
    m.cfg = cfg;

    // Compute backbone input dimensions (same as XFeat.preprocess_tensor)
    m.H32 = (cfg.img_height / 32) * 32;
    m.W32 = (cfg.img_width  / 32) * 32;

    try {
        m.model = torch::jit::load(cfg.engine_path, torch::kCUDA);
        m.model.eval();
    } catch (const c10::Error& e) {
        throw std::runtime_error(std::string("XFeatExtractor: cannot load ") + cfg.engine_path + ": " + e.what());
    }
    m.alloc_buffers();
    return ext;
}

XFeatExtractor::~XFeatExtractor() {
    if (impl_) impl_->free_buffers();
}

const XFeatExtractor::Config& XFeatExtractor::config() const { return impl_->cfg; }

XFeatResult XFeatExtractor::extract(const cv::Mat& gray_img) {
    Impl& m = *impl_;
    assert(gray_img.type() == CV_8UC1);

    const int orig_H = gray_img.rows;
    const int orig_W = gray_img.cols;

    // Resize to (H//32)*32 x (W//32)*32 (match what preprocess_tensor does)
    const int H32 = (orig_H / 32) * 32;
    const int W32 = (orig_W / 32) * 32;

    cv::Mat resized;
    if (H32 != orig_H || W32 != orig_W) {
        cv::resize(gray_img, resized, cv::Size(W32, H32), 0, 0, cv::INTER_AREA);
    } else {
        resized = gray_img;
    }

    cv::Mat f32;
    resized.convertTo(f32, CV_32F, 1.0 / 255.0);

    // Build input tensor [1, 1, H32, W32]
    auto opts  = torch::TensorOptions().dtype(torch::kFloat32);
    auto input = torch::from_blob(f32.data, {1, 1, H32, W32}, opts).to(torch::kCUDA);

    // Run backbone: returns (M1, K1h, H1)
    torch::NoGradGuard ng;
    auto out  = m.model.forward({input}).toTuple();
    auto M1   = out->elements()[0].toTensor().contiguous();  // [1, 64, H32/8, W32/8]
    auto K1h  = out->elements()[1].toTensor().contiguous();  // [1,  1, H32,   W32]
    auto H1   = out->elements()[2].toTensor().contiguous();  // [1,  1, H32/8, W32/8]

    // ── NMS on K1h ──────────────────────────────────────────────────────────
    // Local maximum suppression with 5x5 window
    constexpr int kKernelSize = 5;
    constexpr int kPad = kKernelSize / 2;
    auto local_max = torch::nn::functional::max_pool2d(
        K1h,
        torch::nn::functional::MaxPool2dFuncOptions(kKernelSize)
            .stride(1).padding(kPad));
    auto nms_mask = (K1h == local_max) & (K1h > 0.05f);  // [1, 1, H32, W32]

    // Compute per-pixel scores = K1h * H1_upsampled
    auto H1_up = torch::nn::functional::interpolate(
        H1,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{H32, W32})
            .mode(torch::kBilinear)
            .align_corners(false));
    auto score_map = (K1h * H1_up * nms_mask.to(torch::kFloat32)).squeeze(0).squeeze(0); // [H32, W32]

    // Flatten and select top-k
    auto scores_flat = score_map.flatten();      // [H32*W32]
    int total_pixels = H32 * W32;
    int top_k = std::min(m.cfg.max_keypoints, total_pixels);

    torch::Tensor topk_vals, topk_idxs;
    std::tie(topk_vals, topk_idxs) = scores_flat.topk(top_k, /*dim=*/0, /*largest=*/true, /*sorted=*/true);

    // Filter by positive score
    auto valid_mask = topk_vals > 0.0f;
    auto valid_vals = topk_vals.masked_select(valid_mask);
    auto valid_idxs = topk_idxs.masked_select(valid_mask);
    int N = (int)valid_vals.size(0);

    XFeatResult res;
    res.N = N;
    if (N == 0) {
        res.descriptors_pinned = nullptr;
        res.feat_map_device    = nullptr;
        res.feat_map_h = res.feat_map_w = 0;
        return res;
    }

    // Convert flat indices to (x, y) in H32 x W32 space
    auto kp_y_32 = (valid_idxs / W32).to(torch::kFloat32);   // row → y
    auto kp_x_32 = (valid_idxs % W32).to(torch::kFloat32);   // col → x

    // Scale back to original image coordinates
    float scale_x = (float)orig_W / (float)W32;
    float scale_y = (float)orig_H / (float)H32;
    auto kp_x = kp_x_32 * scale_x;
    auto kp_y = kp_y_32 * scale_y;

    // ── Descriptor sampling via grid_sample ─────────────────────────────────
    // M1: [1, 64, H32/8, W32/8]
    // Grid coords in [-1, 1]: gx = 2*kp_x_32/(W32-1) - 1, gy = 2*kp_y_32/(H32-1) - 1
    auto gx = (kp_x_32 / (float)(W32 - 1)) * 2.0f - 1.0f;  // [N]
    auto gy = (kp_y_32 / (float)(H32 - 1)) * 2.0f - 1.0f;  // [N]
    // grid_sample expects [1, 1, N, 2] (batch=1, H_out=1, W_out=N, xy)
    auto grid = torch::stack({gx, gy}, /*dim=*/1)           // [N, 2]
                    .unsqueeze(0).unsqueeze(0);              // [1, 1, N, 2]

    namespace F = torch::nn::functional;
    auto descs_sampled = F::grid_sample(
        M1, grid,
        F::GridSampleFuncOptions()
            .mode(torch::kBilinear)
            .padding_mode(torch::kZeros)
            .align_corners(false));
    // descs_sampled: [1, 64, 1, N] → [N, 64]
    auto descs = descs_sampled.squeeze(0).squeeze(1).t().contiguous();  // [N, 64]
    // L2-normalise
    descs = F::normalize(descs, F::NormalizeFuncOptions().p(2).dim(1));

    // Copy keypoints and scores to CPU
    auto kp_x_cpu = kp_x.cpu();
    auto kp_y_cpu = kp_y.cpu();
    auto scores_cpu = valid_vals.cpu();

    res.kp_x.resize(N); res.kp_y.resize(N); res.scores.resize(N);
    std::copy(kp_x_cpu.data_ptr<float>(),    kp_x_cpu.data_ptr<float>()    + N, res.kp_x.data());
    std::copy(kp_y_cpu.data_ptr<float>(),    kp_y_cpu.data_ptr<float>()    + N, res.kp_y.data());
    std::copy(scores_cpu.data_ptr<float>(),  scores_cpu.data_ptr<float>()  + N, res.scores.data());

    // ── FP32 -> FP16 + DMA to pinned triple-buffer ──────────────────────────
    int ne = N * kXFeatDescDim;
    int bl = (ne + 255) / 256;
    fp32_to_fp16_kernel<<<bl, 256>>>(descs.data_ptr<float>(), m.d_descs_fp16, ne);

    int slot = m.write_slot;
    cudaMemcpyAsync(m.slots[slot].ptr, m.d_descs_fp16,
                    ne * sizeof(__half), cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(m.slots[slot].ready, 0);
    cudaStreamSynchronize(0);

    res.descriptors_pinned = m.slots[slot].ptr;
    res.feat_map_device    = nullptr;
    res.feat_map_h = res.feat_map_w = 0;
    m.write_slot = (slot + 1) % Impl::kNumSlots;
    return res;
}

} // namespace deep
