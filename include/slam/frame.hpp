#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <memory>
#include <vector>
#include <atomic>
#include <cuda_fp16.h>  // __half — for XFeat descriptor pointer type

namespace slam {

class MapPoint;

/// a single processed image frame.
/// pose convention: T_cw transforms X_w (world) to camera frame: X_c = R_cw * X_w + t_cw
/// stored as Eigen::Isometry3d (4×4 SE3 matrix).
class Frame {
public:
    using Ptr = std::shared_ptr<Frame>;

    // factory
    static Ptr create(const cv::Mat& image, double timestamp, long id);

    // data
    long   id;
    double timestamp;           // seconds
    cv::Mat image_gray;         // grayscale, 8U

    // Features — left image
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat                   descriptors;  // N × 32, CV_8U (ORB) — kept for PnP geometry;
                                            // replaced by xfeat_descriptors in hybrid mode

    // Features — right image (stereo; empty if monocular)
    cv::Mat                   image_right;
    std::vector<cv::KeyPoint> keypoints_right;
    cv::Mat                   descriptors_right;

    // right x-coordinate per left keypoint (-1.0f = no stereo match)
    std::vector<float> uR;

    // map associations (one per keypoint; nullptr = unmatched)
    std::vector<std::shared_ptr<MapPoint>> map_points;

    // Pose: world → camera (T_cw)
    Eigen::Isometry3d T_cw = Eigen::Isometry3d::Identity();

    // Whether this frame is promoted to a keyframe
    bool is_keyframe = false;

    // -----------------------------------------------------------------------
    // Deep-frontend fields (hybrid XFeat mode)
    // -----------------------------------------------------------------------

    // XFeat FP32 descriptors (promoted from FP16 for Ceres compatibility).
    // Layout: N × 64, CV_32F.  Populated by HybridTracker::extract_features().
    cv::Mat xfeat_descriptors;

    // Per-keypoint confidence weight ∈ [0.1, 1.0].
    // Source: L2 ratio pseudo-confidence (temporal) or LighterGlue probability (reloc).
    // Consumed by ConfidenceWeightedReprojectionCost in local_ba.cpp.
    std::vector<float> match_confidence;

    // XFeat feature maps at 1/8 resolution (CHW float32, host memory).
    // Populated on keyframes only for semi-dense disparity; released after use.
    // Shape stored as [kXFeatFeatMapC * (H/8) rows × (W/8) cols].
    cv::Mat feat_map_left;
    cv::Mat feat_map_right;

    // helpers
    /// camera-to-world transform (inverse of T_cw)
    Eigen::Isometry3d T_wc() const { return T_cw.inverse(); }

    /// camera centre in world coordinates
    Eigen::Vector3d camera_center() const { return T_wc().translation(); }

    /// descriptor row as a byte pointer (for GPU upload)
    const uint8_t* desc_ptr() const {
        return descriptors.data;
    }

    int num_features() const { return static_cast<int>(keypoints.size()); }

    /// count keypoints with valid map point associations
    int num_tracked() const;

private:
    Frame() = default;
};

}  // namespace slam
