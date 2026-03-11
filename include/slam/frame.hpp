#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <memory>
#include <vector>
#include <atomic>

namespace slam {

class MapPoint;

/// Represents a single processed image frame.
///
/// Pose convention:  T_cw  — transforms a world-frame point X_w into camera
/// frame:  X_c = R_cw * X_w + t_cw
///
/// The pose is stored as a 4×4 SE3 matrix (Eigen::Isometry3d).
class Frame {
public:
    using Ptr = std::shared_ptr<Frame>;

    // ── Factory ──────────────────────────────────────────────────────────────
    static Ptr create(const cv::Mat& image, double timestamp, long id);

    // ── Data ─────────────────────────────────────────────────────────────────
    long   id;
    double timestamp;           // seconds
    cv::Mat image_gray;         // grayscale, 8U

    // Features — left image
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat                   descriptors;  // N × 32, CV_8U (ORB)

    // Features — right image (stereo; empty if monocular)
    cv::Mat                   image_right;
    std::vector<cv::KeyPoint> keypoints_right;
    cv::Mat                   descriptors_right;

    // Right x-coordinate per left keypoint (-1.0f = no stereo match)
    std::vector<float> uR;

    // Map associations (one per keypoint; nullptr = unmatched)
    std::vector<std::shared_ptr<MapPoint>> map_points;

    // Pose: world → camera (T_cw)
    Eigen::Isometry3d T_cw = Eigen::Isometry3d::Identity();

    // Whether this frame is promoted to a keyframe
    bool is_keyframe = false;

    // ── Helpers ──────────────────────────────────────────────────────────────
    /// Camera-to-world transform (inverse of T_cw)
    Eigen::Isometry3d T_wc() const { return T_cw.inverse(); }

    /// Camera centre in world coordinates
    Eigen::Vector3d camera_center() const { return T_wc().translation(); }

    /// Descriptor row as a byte pointer (for GPU upload)
    const uint8_t* desc_ptr() const {
        return descriptors.data;
    }

    int num_features() const { return static_cast<int>(keypoints.size()); }

    /// Count keypoints with valid map point associations
    int num_tracked() const;

private:
    Frame() = default;
};

}  // namespace slam
