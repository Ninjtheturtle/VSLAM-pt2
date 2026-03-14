#pragma once
// SemiDenseDisparity: 1D cross-correlation on XFeat 1/8-resolution feature maps
// to generate sub-pixel stereo disparities and a semi-dense point cloud.
//
// ISOLATION CONTRACT:
//   Output points go EXCLUSIVELY to Rerun (world/map/semi_dense).
//   Nothing from this class is inserted into Map or the Ceres tracking problem.
//   Injecting 1/8-res depth estimates would corrupt the sparse PnP map.
//
// Algorithm:
//   For each spatial position (r, c) in the left feature map:
//     Compute 1D normalized cross-correlation along the epipolar line (same row)
//     with a search window of ±search_width pixels (at 1/8 scale).
//     Peak location gives sub-pixel disparity d_8 (in 1/8-res pixels).
//     Full-resolution disparity: d = d_8 * 8
//     Depth: Z = fx * baseline / d
//     Sharpness filter: ratio of peak to second-highest response >= min_peak_ratio
//   3D point unprojected via T_wc into world frame.

#include <opencv2/core.hpp>
#include <Eigen/Geometry>
#include <vector>

namespace deep {

struct SemiDensePoint3D {
    float x, y, z;       // world-frame position
    float confidence;    // cross-correlation peak sharpness ∈ [0, 1]
};

class SemiDenseDisparity {
public:
    struct Config {
        float baseline;          // metric stereo baseline (m)
        float fx;                // focal length at FULL resolution (pixels)
        float cx, cy;            // principal point at full resolution
        float d_min_full = 3.0f;   // minimum valid disparity at full resolution
        float d_max_full = 300.0f; // maximum valid disparity
        int   search_width = 64;   // ±search_width in 1/8-resolution pixels
        float min_peak_ratio = 1.5f; // sharpness filter: peak / second_peak
        float min_depth = 0.5f;    // meters, reject points closer than this
        float max_depth = 100.0f;  // meters, reject far points (low-confidence)
    };

    explicit SemiDenseDisparity(const Config& cfg);

    // Compute semi-dense point cloud from XFeat feature maps.
    //
    // feat_left, feat_right: CHW float32 host Mat.
    //   Shape: [kXFeatFeatMapC × feat_h × feat_w] stored as a single contiguous Mat
    //   with dims [kXFeatFeatMapC, feat_h, feat_w].  (rows = C*feat_h, cols = feat_w)
    //
    // T_wc: camera-to-world SE3 transform (Isometry3d, consistent with tracker convention)
    //
    // Returns triangulated world-frame points for Rerun visualization ONLY.
    std::vector<SemiDensePoint3D> compute(
        const cv::Mat& feat_left,    // CHW float32
        const cv::Mat& feat_right,   // CHW float32
        const Eigen::Isometry3d& T_wc
    ) const;

private:
    Config cfg_;
};

} // namespace deep
