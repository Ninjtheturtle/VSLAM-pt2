#pragma once

#include "slam/camera.hpp"
#include "slam/map.hpp"
#include <memory>

namespace slam {

/// Sliding-window local bundle adjustment using Ceres Solver.
///
/// Optimizes the poses of the N most recent keyframes and all map points
/// visible in those keyframes.  The oldest keyframe's pose is held fixed
/// to remove gauge freedom (no absolute reference in monocular SLAM).
///
/// Cost function: reprojection error with Huber loss (δ = 1.0 pixel).
/// Parameterization: 6-DOF pose = [angle-axis ω (3D) | translation t (3D)]
///
/// Analytical Jacobians are provided for efficiency (avoids auto-diff overhead
/// on the reprojection cost, which is a tight inner loop in BA).
class LocalBA {
public:
    struct Config {
        int   max_iterations     = 60;
        int   window_size        = Map::kWindowSize;
        double huber_delta       = 1.0;   // pixels
        bool   verbose           = false;
    };

    using Ptr = std::shared_ptr<LocalBA>;
    static Ptr create(const Camera& cam, Map::Ptr map,
                      const Config& cfg = Config{});

    /// Run one BA iteration on the current sliding window.
    /// Called after inserting a new keyframe.
    void optimize();

private:
    Camera   cam_;
    Map::Ptr map_;
    Config   cfg_;

    LocalBA() = default;
};

// ─── Analytical Jacobian Cost Function ───────────────────────────────────────
//
// See local_ba.cpp for the implementation.
// Declared here so unit tests can instantiate it directly.

struct ReprojectionCost {
    // Observation: image coordinates and camera intrinsics
    double u_obs, v_obs;
    double fx, fy, cx, cy;

    ReprojectionCost(double u, double v, double fx, double fy,
                     double cx, double cy)
        : u_obs(u), v_obs(v), fx(fx), fy(fy), cx(cx), cy(cy) {}

    // Ceres analytic cost function signature:
    // pose[6]  = [omega_x, omega_y, omega_z, tx, ty, tz]
    // point[3] = [X, Y, Z] world frame
    // residuals[2] = [u - u_obs, v - v_obs]
    bool operator()(const double* const pose,
                    const double* const point,
                    double* residuals,
                    double** jacobians) const;

    static constexpr int kNumResiduals    = 2;
    static constexpr int kNumPoseParams   = 6;
    static constexpr int kNumPointParams  = 3;
};

// ─── Stereo Analytical Jacobian Cost Function ─────────────────────────────────
//
// Adds a third residual for the right camera reprojection:
//   u_R = fx * (X_c - baseline) / Z_c + cx
//
// residuals[2] = u_R_proj - u_R_obs
//
// The right camera shares the same intrinsics K and is translated by
// [-baseline, 0, 0] in the left camera frame (rectified stereo).

struct StereoReprojectionCost {
    double u_L_obs, v_L_obs, u_R_obs;
    double fx, fy, cx, cy, baseline;

    StereoReprojectionCost(double uL, double vL, double uR,
                           double fx_, double fy_, double cx_, double cy_,
                           double b)
        : u_L_obs(uL), v_L_obs(vL), u_R_obs(uR),
          fx(fx_), fy(fy_), cx(cx_), cy(cy_), baseline(b) {}

    // residuals[3] = [u_L - u_L_obs, v_L - v_L_obs, u_R - u_R_obs]
    bool operator()(const double* const pose,
                    const double* const point,
                    double* residuals,
                    double** jacobians) const;

    static constexpr int kNumResiduals   = 3;
    static constexpr int kNumPoseParams  = 6;
    static constexpr int kNumPointParams = 3;
};

}  // namespace slam
