#pragma once

#include "slam/camera.hpp"
#include "slam/frame.hpp"
#include "slam/map.hpp"
#include <array>
#include <memory>
#include <string>
#include <vector>

// Forward-declare deep types to avoid pulling deep headers into every TU
namespace deep { struct SemiDensePoint3D; }

// forward-declare Rerun types to avoid including the full SDK in every TU
namespace rerun { class RecordingStream; }

namespace slam {

/// logs SLAM state to a Rerun.io recording stream via TCP (default: localhost:9876).
/// call log_frame() every frame and log_map() after each BA.
class Visualizer {
public:
    struct Config {
        std::string app_id   = "vslam2";
        std::string addr     = "127.0.0.1:9876";
        bool log_image       = true;
        bool log_keypoints   = true;
    };

    using Ptr = std::shared_ptr<Visualizer>;
    static Ptr create(const Config& cfg = Config{});

    ~Visualizer();

    /// log camera intrinsics once so Rerun renders a frustum + image panel; call before log_frame()
    void log_pinhole(const Camera& cam);

    /// log the current frame: image, pose, 2D keypoints
    void log_frame(const Frame::Ptr& frame);

    /// log the current map: 3D point cloud
    void log_map(const Map::Ptr& map, double timestamp = 0.0);

    /// log the SLAM trajectory, rebuilt every call from BA-refined KF poses.
    /// automatically reflects BA corrections without any extra bookkeeping.
    void log_trajectory(const Map::Ptr& map,
                        const Frame::Ptr& current_frame,
                        double ts);

    /// overlay GT camera centres as a static orange trajectory.
    /// call once before the main loop with poses from data/poses/XX.txt (KITTI format).
    void log_ground_truth(const std::vector<std::array<float, 3>>& centers);

    /// log semi-dense point cloud from SemiDenseDisparity (visualization only).
    /// entity path: world/map/semi_dense  — NOT inserted into Map or Ceres.
    void log_semi_dense(const std::vector<deep::SemiDensePoint3D>& pts, double ts);

private:
    Visualizer() = default;

    Config cfg_;
    Camera cam_;   // stored from log_pinhole(); used in log_frame()
    std::unique_ptr<rerun::RecordingStream> rec_;

    // no trajectory buffer — rebuilt from map->all_keyframes() every call
};

}  // namespace slam
