#pragma once
// XFeatExtractor: libtorch TorchScript inference for XFeat keypoint/descriptor extraction.
//
// All inference is synchronous on the main CUDA stream.
// Pinned memory triple-buffers provide FP16 descriptors for the CUDA L2 matcher.
//
// Ownership contract for XFeatResult::descriptors_pinned:
//   - Pointer is valid until the NEXT call to extract() that cycles back to the same slot
//     (i.e. valid for at least 2 subsequent extract() calls with triple-buffering).
//   - LighterGlueAsync::submit_job() performs a DEEP COPY before enqueuing, so no
//     cross-thread lifetime issues arise from borrowed pointers.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace deep {

static constexpr int kXFeatDescDim  = 64;   // descriptor dimension (FP16 elements)
static constexpr int kXFeatMaxKps   = 2000; // ANMS output cap
static constexpr int kXFeatFeatMapC = 64;   // feature-map channels at 1/8 resolution

// ---------------------------------------------------------------------------
// XFeatResult
// Returned by XFeatExtractor::extract(). All device pointers remain valid
// only until the next extract() call. Host vectors are always owned copies.
// ---------------------------------------------------------------------------
struct XFeatResult {
    // Keypoint pixel coordinates (left-image frame, full resolution)
    std::vector<float> kp_x;    // [N]
    std::vector<float> kp_y;    // [N]
    std::vector<float> scores;  // [N] heatmap response ∈ (0, 1]
    int N = 0;

    // BORROWED pointer into XFeatExtractor's pinned triple-buffer.
    // Layout: N × 64 FP16, row-major.
    // Copy before next extract() if lifetime extension is needed.
    const __half* descriptors_pinned = nullptr;

    // DEVICE pointer (stream 0) to feature map at 1/8 resolution.
    // Layout: kXFeatFeatMapC × (H/8) × (W/8), CHW float32.
    // Copy with cudaMemcpy before next extract() if needed.
    // nullptr if the engine was built without a feature-map output binding.
    const float* feat_map_device = nullptr;
    int feat_map_h = 0;  // H/8
    int feat_map_w = 0;  // W/8
};

// ---------------------------------------------------------------------------
// XFeatExtractor
// ---------------------------------------------------------------------------
class XFeatExtractor {
public:
    struct Config {
        std::string engine_path;      // path to TorchScript .pt model file
        int img_width  = 1242;        // expected input width  (KITTI left image)
        int img_height = 376;         // expected input height
        int max_keypoints = kXFeatMaxKps;
        float anms_min_response = 0.005f;  // unused (model runs internal ANMS)
        float anms_min_dist_px  = 8.0f;    // unused (model runs internal ANMS)
    };

    static std::unique_ptr<XFeatExtractor> create(const Config& cfg);
    ~XFeatExtractor();

    // Synchronous inference on CUDA stream 0.
    // gray_img must be CV_8UC1, size (cfg.img_height × cfg.img_width).
    // Result is valid until the next extract() call.
    XFeatResult extract(const cv::Mat& gray_img);

    const Config& config() const;

private:
    // Pimpl hides NvInfer headers from downstream translation units
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace deep
