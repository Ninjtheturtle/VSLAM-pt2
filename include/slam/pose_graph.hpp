#pragma once

#include "slam/camera.hpp"
#include "slam/map.hpp"
#include <Eigen/Geometry>
#include <memory>
#include <vector>

namespace slam {

/// Pose Graph Optimization (PGO) over all keyframes.
///
/// Detects loop-closure candidates via co-visibility (shared map-point
/// observations) between the newest keyframe and KFs outside the local BA
/// window.  When a candidate is found, a relative-pose edge is added and a
/// lightweight Ceres PGO is run over ALL keyframe poses.
///
/// This corrects the slow yaw/pitch drift that accumulates beyond the 20-KF
/// local-BA window without requiring a visual vocabulary (DBoW).
class PoseGraph {
public:
    using Ptr = std::shared_ptr<PoseGraph>;

    struct Config {
        int  min_shared_points = 15;   // minimum shared observations for co-visibility edge
        int  pgo_interval      = 5;    // run PGO every N new keyframes
        int  max_iterations    = 30;   // Ceres iterations for PGO solve
        double w_t             = 10.0; // translation residual weight (1/σ_t, σ_t=0.1 m)
        double w_r             = 50.0; // rotation residual weight   (1/σ_r, σ_r=0.02 rad)
        int  visual_min_inliers = 20;  // PnP inliers required to accept a visual loop edge
        int  visual_sample_step = 10;  // sample every Nth KF outside BA window for visual search
    };

    static Ptr create(Map::Ptr map, const Camera& cam, const Config& cfg = Config{});

    /// Register a newly inserted keyframe.  Does NOT run detection or PGO;
    /// call detect_and_add_loops() + optimize() explicitly on a schedule.
    void add_keyframe(Frame::Ptr kf);

    /// Scan KFs outside the local BA window for co-visibility edges with the
    /// most recently registered KF.  Sets has_new_loops() if any are found.
    void detect_and_add_loops();

    /// Appearance-based loop detection: match descriptors of `query` against
    /// sampled KFs outside the BA window.  For each candidate with enough
    /// descriptor matches, run PnP to verify the loop geometrically.  Adds a
    /// loop edge and sets has_new_loops() if inliers >= cfg_.visual_min_inliers.
    void detect_and_add_loops_visual(Frame::Ptr query);

    /// True if the last detect_and_add_loops() call found at least one new edge.
    bool has_new_loops() const { return new_loops_; }

    /// Run Ceres PGO over all KF poses using accumulated edges.
    /// Writes optimised poses back to kf->T_cw for all keyframes.
    void optimize();

    int num_edges() const { return static_cast<int>(edges_.size()); }

private:
    struct Edge {
        long id_a, id_b;
        // Measured relative transform: T_ab = T_a_cw * T_b_cw.inverse()
        // i.e., transforms camera-B frame to camera-A frame.
        double R_meas[9];  // row-major rotation
        double t_meas[3];  // translation
    };

    Camera     cam_;
    Map::Ptr   map_;
    Config     cfg_;
    std::vector<Frame::Ptr> kf_order_;  // insertion order (mirrors map, for gauge anchor)
    std::vector<Edge>       edges_;
    bool new_loops_ = false;

    PoseGraph() = default;
};

}  // namespace slam
