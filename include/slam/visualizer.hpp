#pragma once

#include "slam/camera.hpp"
#include "slam/frame.hpp"
#include "slam/map.hpp"
#include <array>
#include <memory>
#include <string>
#include <vector>

// Forward-declare Rerun types to avoid including the full SDK in every TU
namespace rerun { class RecordingStream; }

namespace slam {

/// Logs SLAM state to a Rerun.io recording stream.
///
/// Opens a TCP connection to the Rerun viewer (default: localhost:9876).
/// Call log_frame() after every tracked frame, and log_map() after every BA.
///
/// Rerun entity paths:
///   "camera/image"      — current frame (grayscale)
///   "camera/pose"       — camera transform (world → camera)
///   "camera/keypoints"  — 2D feature detections
///   "map/points"        — all active 3D map point positions
///   "map/trajectory"    — camera centre history (LineStrips3D)
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

    /// Log camera intrinsics once so Rerun renders a proper frustum + image panel.
    /// Call this once after create(), before the first log_frame().
    void log_pinhole(const Camera& cam);

    /// Log the current frame: image, pose, 2-D keypoints
    void log_frame(const Frame::Ptr& frame);

    /// Log the current map: 3-D point cloud
    void log_map(const Map::Ptr& map, double timestamp = 0.0);

    /// Log the SLAM trajectory, rebuilt every call from BA-refined keyframe poses.
    /// Also appends the current live frame position if it is tracked and not yet
    /// a keyframe.  Call this every frame — it automatically reflects the latest
    /// BA refinement without any extra bookkeeping.
    void log_trajectory(const Map::Ptr& map,
                        const Frame::Ptr& current_frame,
                        double ts);

    /// Overlay ground-truth camera centres as a static orange trajectory.
    /// Call once before the main loop with all GT poses loaded from
    /// data/poses/XX.txt (KITTI format: 3×4 row-major matrices).
    void log_ground_truth(const std::vector<std::array<float, 3>>& centers);

private:
    Visualizer() = default;

    Config cfg_;
    Camera cam_;   // stored from log_pinhole(); used for depth projection in log_frame()
    std::unique_ptr<rerun::RecordingStream> rec_;

    // No trajectory buffer: trajectory is rebuilt from map->all_keyframes() every call.
};

}  // namespace slam
