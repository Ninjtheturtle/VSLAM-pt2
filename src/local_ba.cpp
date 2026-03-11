// local_ba.cpp
//
// Sliding-window Local Bundle Adjustment using Ceres Solver.
//
// Pose parameterization
// ─────────────────────
// Each camera pose is represented as a 6-vector:
//   pose[0..2]  = angle-axis vector ω  (SO3 Lie algebra; |ω| = rotation angle)
//   pose[3..5]  = translation vector t
//
// The camera-frame point is:  X_c = R(ω) * X_w + t
//
// Analytical Jacobians
// ────────────────────
// For the reprojection cost  r = [u - u_obs, v - v_obs] where
//   u = fx * (X_c[0]/X_c[2]) + cx
//   v = fy * (X_c[1]/X_c[2]) + cy
//
// We provide exact first-order Jacobians of r w.r.t. pose and w.r.t. X_w.
// This is faster than auto-diff for a reprojection function.
//
// ∂r/∂X_c  (2×3, the projection Jacobian):
//   J_proj = [ fx/Zc,     0,  -fx*Xc/Zc² ]
//             [     0,  fy/Zc, -fy*Yc/Zc² ]
//
// ∂X_c/∂t  = I₃   (translation Jacobian)
//
// ∂X_c/∂ω  = -[X_c]×   (rotation Jacobian, skew-symmetric of camera point)
// (exact only for small |ω|; for larger rotations derived from Rodrigues)
//
// ∂X_c/∂X_w = R(ω)  (point Jacobian, the rotation matrix)

#include "slam/local_ba.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>  // AngleAxisRotatePoint, AngleAxisToRotationMatrix

#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace slam {

// ─── Analytical Cost Function ─────────────────────────────────────────────────

bool ReprojectionCost::operator()(const double* const pose,   // [ω₀,ω₁,ω₂, t₀,t₁,t₂]
                                  const double* const point,  // [X,Y,Z] world
                                  double* residuals, double** jacobians) const {
    // ── Transform world point to camera frame ─────────────────────────────────
    // X_c = R(ω) * X_w + t  using Ceres AngleAxisRotatePoint
    double Xc[3];
    ceres::AngleAxisRotatePoint(pose, point, Xc);
    Xc[0] += pose[3];
    Xc[1] += pose[4];
    Xc[2] += pose[5];

    const double inv_Zc = 1.0 / Xc[2];
    const double inv_Zc2 = inv_Zc * inv_Zc;

    // ── Residuals ─────────────────────────────────────────────────────────────
    const double u_proj = fx * Xc[0] * inv_Zc + cx;
    const double v_proj = fy * Xc[1] * inv_Zc + cy;
    residuals[0] = u_proj - u_obs;
    residuals[1] = v_proj - v_obs;

    // ── Jacobians ─────────────────────────────────────────────────────────────
    if (!jacobians) return true;

    // J_proj: ∂[u,v]/∂X_c  (2×3)
    //   row 0: [ fx/Zc,      0,  -fx*Xc/Zc² ]
    //   row 1: [     0,  fy/Zc,  -fy*Yc/Zc² ]
    const double jp00 = fx * inv_Zc;
    const double jp02 = -fx * Xc[0] * inv_Zc2;
    const double jp11 = fy * inv_Zc;
    const double jp12 = -fy * Xc[1] * inv_Zc2;

    // ── Jacobian w.r.t. pose (2×6) ───────────────────────────────────────────
    if (jacobians[0]) {
        // ∂X_c/∂ω (3×3)
        // From the Rodrigues formula derivative at X_c:
        //   ∂(R(ω) * X_w)/∂ω = -[X_c]×  (skew-symmetric)
        // (This is the exact form used by Ceres internally.)
        //
        // [X_c]× = [  0,  -Zc,  Yc ]
        //           [ Zc,    0, -Xc ]
        //           [-Yc,   Xc,   0 ]
        //
        // ∂X_c/∂ω = -[X_c]×
        //          = [  0,   Zc, -Yc ]
        //            [-Zc,    0,  Xc ]
        //            [ Yc,  -Xc,   0 ]
        //
        // Note: this is the linearized (small-angle) version valid for
        // the increment around the current ω.  Ceres handles the manifold
        // update, so this local Jacobian is correct for the solver.

        const double dXc_dw[3][3] = {
            {0.0, Xc[2], -Xc[1]}, {-Xc[2], 0.0, Xc[0]}, {Xc[1], -Xc[0], 0.0}};

        // ∂r/∂ω = J_proj * ∂X_c/∂ω  (2×3)
        double* j_pose = jacobians[0];  // row-major 2×6

        // Column 0 (dω₀)
        j_pose[0] = jp00 * dXc_dw[0][0] + jp02 * dXc_dw[2][0];  // ∂u/∂ω₀
        j_pose[6] = jp11 * dXc_dw[1][0] + jp12 * dXc_dw[2][0];  // ∂v/∂ω₀
        // Column 1 (dω₁)
        j_pose[1] = jp00 * dXc_dw[0][1] + jp02 * dXc_dw[2][1];  // ∂u/∂ω₁
        j_pose[7] = jp11 * dXc_dw[1][1] + jp12 * dXc_dw[2][1];  // ∂v/∂ω₁
        // Column 2 (dω₂)
        j_pose[2] = jp00 * dXc_dw[0][2] + jp02 * dXc_dw[2][2];  // ∂u/∂ω₂
        j_pose[8] = jp11 * dXc_dw[1][2] + jp12 * dXc_dw[2][2];  // ∂v/∂ω₂

        // ∂X_c/∂t = I₃  → ∂r/∂t = J_proj  (columns 3,4,5)
        j_pose[3] = jp00;   // ∂u/∂t₀
        j_pose[4] = 0.0;    // ∂u/∂t₁
        j_pose[5] = jp02;   // ∂u/∂t₂
        j_pose[9] = 0.0;    // ∂v/∂t₀
        j_pose[10] = jp11;  // ∂v/∂t₁
        j_pose[11] = jp12;  // ∂v/∂t₂
    }

    // ── Jacobian w.r.t. 3D point X_w (2×3) ──────────────────────────────────
    if (jacobians[1]) {
        // ∂X_c/∂X_w = R(ω)  (3×3 rotation matrix)
        // Extract R from angle-axis using Ceres utility
        double R[9];  // row-major 3×3
        ceres::AngleAxisToRotationMatrix(pose, R);
        // R layout: R[row*3 + col]

        double* j_point = jacobians[1];  // row-major 2×3

        // ∂r/∂X_w = J_proj * R  (2×3)
        // Row 0 (∂u/∂X_w):
        j_point[0] = jp00 * R[0] + jp02 * R[6];  // ∂u/∂X
        j_point[1] = jp00 * R[1] + jp02 * R[7];  // ∂u/∂Y
        j_point[2] = jp00 * R[2] + jp02 * R[8];  // ∂u/∂Z
        // Row 1 (∂v/∂X_w):
        j_point[3] = jp11 * R[3] + jp12 * R[6];  // ∂v/∂X
        j_point[4] = jp11 * R[4] + jp12 * R[7];  // ∂v/∂Y
        j_point[5] = jp11 * R[5] + jp12 * R[8];  // ∂v/∂Z
    }

    return true;
}

// ─── Ceres wrapper adapting ReprojectionCost ───────────────────────────────

class AnalyticReprojectionCostFunction
    : public ceres::SizedCostFunction<ReprojectionCost::kNumResiduals,    // 2 residuals
                                      ReprojectionCost::kNumPoseParams,   // 6 pose params
                                      ReprojectionCost::kNumPointParams>  // 3 point params
{
   public:
    explicit AnalyticReprojectionCostFunction(const ReprojectionCost& cost) : cost_(cost) {}

    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const override {
        return cost_(parameters[0], parameters[1], residuals, jacobians);
    }

   private:
    ReprojectionCost cost_;
};

// ─── Stereo Analytical Cost Function ──────────────────────────────────────────
//
// Residuals (3):
//   r[0] = u_L_proj - u_L_obs
//   r[1] = v_L_proj - v_L_obs
//   r[2] = u_R_proj - u_R_obs   where u_R_proj = fx*(Xc[0]-b)/Zc + cx
//
// Jacobian rows 0-1 are identical to ReprojectionCost.
// Row 2 (∂u_R/∂...):
//   ∂u_R/∂X_c = [ fx/Zc, 0, -fx*(Xc[0]-b)/Zc² ]  (same form as left u but X→X-b)
//   ∂u_R/∂ω   = ∂u_R/∂X_c · ∂X_c/∂ω   (same ∂X_c/∂ω as above)
//   ∂u_R/∂t   = ∂u_R/∂X_c · I₃          (columns 3,4,5)
//   ∂u_R/∂X_w = ∂u_R/∂X_c · R

bool StereoReprojectionCost::operator()(
    const double* const pose,
    const double* const point,
    double* residuals, double** jacobians) const
{
    double Xc[3];
    ceres::AngleAxisRotatePoint(pose, point, Xc);
    Xc[0] += pose[3];
    Xc[1] += pose[4];
    Xc[2] += pose[5];

    const double inv_Zc  = 1.0 / Xc[2];
    const double inv_Zc2 = inv_Zc * inv_Zc;

    const double u_L_proj = fx * Xc[0] * inv_Zc + cx;
    const double v_L_proj = fy * Xc[1] * inv_Zc + cy;
    const double u_R_proj = fx * (Xc[0] - baseline) * inv_Zc + cx;

    residuals[0] = u_L_proj - u_L_obs;
    residuals[1] = v_L_proj - v_L_obs;
    residuals[2] = u_R_proj - u_R_obs;

    if (!jacobians) return true;

    // Shared projection Jacobian terms
    const double jp00 = fx * inv_Zc;
    const double jp02 = -fx * Xc[0] * inv_Zc2;
    const double jp11 = fy * inv_Zc;
    const double jp12 = -fy * Xc[1] * inv_Zc2;
    // Right u Jacobian w.r.t. X_c:  [ fx/Zc, 0, -fx*(Xc[0]-b)/Zc² ]
    const double jr02 = -fx * (Xc[0] - baseline) * inv_Zc2;

    const double dXc_dw[3][3] = {
        {0.0, Xc[2], -Xc[1]}, {-Xc[2], 0.0, Xc[0]}, {Xc[1], -Xc[0], 0.0}};

    if (jacobians[0]) {
        double* j = jacobians[0];  // row-major 3×6
        // Row 0 (∂u_L/∂pose) — same as ReprojectionCost
        j[0]  = jp00 * dXc_dw[0][0] + jp02 * dXc_dw[2][0];
        j[1]  = jp00 * dXc_dw[0][1] + jp02 * dXc_dw[2][1];
        j[2]  = jp00 * dXc_dw[0][2] + jp02 * dXc_dw[2][2];
        j[3]  = jp00;   j[4]  = 0.0;   j[5]  = jp02;
        // Row 1 (∂v_L/∂pose)
        j[6]  = jp11 * dXc_dw[1][0] + jp12 * dXc_dw[2][0];
        j[7]  = jp11 * dXc_dw[1][1] + jp12 * dXc_dw[2][1];
        j[8]  = jp11 * dXc_dw[1][2] + jp12 * dXc_dw[2][2];
        j[9]  = 0.0;    j[10] = jp11;  j[11] = jp12;
        // Row 2 (∂u_R/∂pose)  — same ∂X_c/∂ω; right proj uses jp00 (same fx/Zc) and jr02
        j[12] = jp00 * dXc_dw[0][0] + jr02 * dXc_dw[2][0];
        j[13] = jp00 * dXc_dw[0][1] + jr02 * dXc_dw[2][1];
        j[14] = jp00 * dXc_dw[0][2] + jr02 * dXc_dw[2][2];
        j[15] = jp00;   j[16] = 0.0;   j[17] = jr02;
    }

    if (jacobians[1]) {
        double R[9];
        ceres::AngleAxisToRotationMatrix(pose, R);
        double* j = jacobians[1];  // row-major 3×3
        // Row 0 (∂u_L/∂X_w)
        j[0] = jp00 * R[0] + jp02 * R[6];
        j[1] = jp00 * R[1] + jp02 * R[7];
        j[2] = jp00 * R[2] + jp02 * R[8];
        // Row 1 (∂v_L/∂X_w)
        j[3] = jp11 * R[3] + jp12 * R[6];
        j[4] = jp11 * R[4] + jp12 * R[7];
        j[5] = jp11 * R[5] + jp12 * R[8];
        // Row 2 (∂u_R/∂X_w)  — uses jr02 instead of jp02
        j[6] = jp00 * R[0] + jr02 * R[6];
        j[7] = jp00 * R[1] + jr02 * R[7];
        j[8] = jp00 * R[2] + jr02 * R[8];
    }

    return true;
}

class AnalyticStereoReprojectionCostFunction
    : public ceres::SizedCostFunction<StereoReprojectionCost::kNumResiduals,   // 3 residuals
                                      StereoReprojectionCost::kNumPoseParams,  // 6 pose params
                                      StereoReprojectionCost::kNumPointParams> // 3 point params
{
   public:
    explicit AnalyticStereoReprojectionCostFunction(const StereoReprojectionCost& cost)
        : cost_(cost) {}

    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const override {
        return cost_(parameters[0], parameters[1], residuals, jacobians);
    }

   private:
    StereoReprojectionCost cost_;
};

// ─── Pose Prior Cost (motion model smoothing) ─────────────────────────────────
//
// Soft anchor: penalises deviation of each non-anchor KF from its pre-BA
// (PnP-estimated) pose.  Prevents BA from placing KFs at physically implausible
// positions when the reprojection residual count is very low (sparse features,
// dark turns, near-LOST states).
// Weight 0.5 per residual component: only dominates when < ~10 reprojection
// observations remain for a KF (50× weaker than 1px reprojection at full coverage).
struct PosePriorCost {
    double prior[6];
    double w_r, w_t;

    template <typename T>
    bool operator()(const T* const pose, T* res) const {
        for (int i = 0; i < 3; ++i) res[i]   = T(w_r) * (pose[i]   - T(prior[i]));
        for (int i = 0; i < 3; ++i) res[i+3] = T(w_t) * (pose[i+3] - T(prior[i+3]));
        return true;
    }

    static ceres::CostFunction* Create(const double* p, double w_r, double w_t) {
        auto* c = new PosePriorCost;
        std::copy(p, p + 6, c->prior);
        c->w_r = w_r; c->w_t = w_t;
        return new ceres::AutoDiffCostFunction<PosePriorCost, 6, 6>(c);
    }
};


// ─── Pose ↔ Isometry3d conversion helpers ────────────────────────────────────

/// Convert Eigen::Isometry3d → 6-DOF pose vector [ω, t]
static void isometry_to_pose(const Eigen::Isometry3d& T, double* pose) {
    // Angle-axis from rotation matrix
    Eigen::AngleAxisd aa(T.rotation());
    Eigen::Vector3d omega = aa.angle() * aa.axis();
    pose[0] = omega.x();
    pose[1] = omega.y();
    pose[2] = omega.z();
    pose[3] = T.translation().x();
    pose[4] = T.translation().y();
    pose[5] = T.translation().z();
}

/// Convert 6-DOF pose vector [ω, t] → Eigen::Isometry3d
static Eigen::Isometry3d pose_to_isometry(const double* pose) {
    Eigen::Vector3d omega(pose[0], pose[1], pose[2]);
    double angle = omega.norm();
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    if (angle > 1e-9) {
        T.linear() = Eigen::AngleAxisd(angle, omega / angle).toRotationMatrix();
    }
    T.translation() << pose[3], pose[4], pose[5];
    return T;
}

// ─── LocalBA::Factory & optimize() ───────────────────────────────────────────

LocalBA::Ptr LocalBA::create(const Camera& cam, Map::Ptr map, const Config& cfg) {
    auto ba = std::shared_ptr<LocalBA>(new LocalBA());
    ba->cam_ = cam;
    ba->map_ = map;
    ba->cfg_ = cfg;
    return ba;
}

void LocalBA::optimize() {
    // ── 1. Gather the sliding window of keyframes ─────────────────────────────
    auto window = map_->local_window();
    if (window.size() < 2) return;

    // ── 2. Collect all map points visible in the window ──────────────────────
    std::unordered_map<long, MapPoint::Ptr> active_points;
    for (auto& kf : window) {
        for (auto& mp : kf->map_points) {
            if (mp && !mp->is_bad) {
                active_points[mp->id] = mp;
            }
        }
    }
    if (active_points.empty()) return;

    // ── 3. Allocate parameter blocks ─────────────────────────────────────────
    // Poses: window_size × 6 doubles
    std::unordered_map<long, std::vector<double>> pose_params;
    for (auto& kf : window) {
        pose_params[kf->id].resize(6);
        isometry_to_pose(kf->T_cw, pose_params[kf->id].data());
    }

    // Points: map_point_id × 3 doubles
    std::unordered_map<long, std::array<double, 3>> point_params;
    for (auto& [id, mp] : active_points) {
        point_params[id] = {mp->position.x(), mp->position.y(), mp->position.z()};
    }

    // ── 3b. Record pre-BA poses for PosePriorCost (soft anchor) ──────────────
    std::unordered_map<long, std::array<double, 6>> prior_poses;
    for (auto& kf : window) {
        prior_poses[kf->id] = {};
        std::copy(pose_params[kf->id].begin(), pose_params[kf->id].end(),
                  prior_poses[kf->id].begin());
    }

    // ── 4. Build Ceres Problem ────────────────────────────────────────────────
    //
    // Turn-adaptive Huber kernel:
    // During sharp turns, distant features have large apparent pixel velocity
    // but noisy depth estimates — their reprojection residuals are inflated by
    // the rotation and can corrupt the BA result.  A tighter Huber δ
    // (smaller → transitions to linear sooner) downweights these outliers
    // more aggressively while leaving genuine inliers (small residuals) alone.
    //
    // Threshold: 0.05 rad/KF ≈ 3°/KF — above normal straight driving but well
    // below the sharp-turn regime (>0.1 rad/KF).
    double effective_huber = cfg_.huber_delta;
    if (window.size() >= 2) {
        const auto& kf_last = window.back();
        const auto& kf_prev = window[window.size() - 2];
        Eigen::Isometry3d T_rel = kf_last->T_cw * kf_prev->T_cw.inverse();
        double kf_angle = Eigen::AngleAxisd(T_rel.rotation()).angle();
        if (kf_angle > 0.05) {
            // Sharp turn: halve the Huber threshold to reject noisy distant features
            effective_huber = cfg_.huber_delta * 0.5;
        }
    }

    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(effective_huber);

    for (auto& kf : window) {
        double* pose = pose_params[kf->id].data();

        for (int kp_idx = 0; kp_idx < (int)kf->keypoints.size(); ++kp_idx) {
            auto& mp = kf->map_points[kp_idx];
            if (!mp || mp->is_bad) continue;

            auto pit = point_params.find(mp->id);
            if (pit == point_params.end()) continue;

            double* pt = pit->second.data();
            const cv::Point2f& obs = kf->keypoints[kp_idx].pt;

            // Use stereo cost (3 residuals) when a valid right-image observation exists.
            if (cam_.is_stereo() &&
                kp_idx < (int)kf->uR.size() && kf->uR[kp_idx] >= 0.0f) {
                StereoReprojectionCost cost(obs.x, obs.y, kf->uR[kp_idx],
                                            cam_.fx, cam_.fy, cam_.cx, cam_.cy,
                                            cam_.baseline);
                problem.AddResidualBlock(
                    new AnalyticStereoReprojectionCostFunction(cost), loss, pose, pt);
            } else {
                ReprojectionCost cost(obs.x, obs.y, cam_.fx, cam_.fy, cam_.cx, cam_.cy);
                problem.AddResidualBlock(
                    new AnalyticReprojectionCostFunction(cost), loss, pose, pt);
            }
        }

        problem.AddParameterBlock(pose, 6);

    }

    // Add point parameter blocks
    for (auto& [id, pt] : point_params) {
        problem.AddParameterBlock(pt.data(), 3);
    }

    // Fix oldest keyframe to remove gauge freedom (stereo SLAM)
    if (!window.empty()) {
        double* oldest_pose = pose_params[window.front()->id].data();
        problem.SetParameterBlockConstant(oldest_pose);
    }

    // ── 4b. Pose prior (motion model soft anchor) ────────────────────────────
    // Soft anchor to the pre-BA PnP estimate for each non-anchor KF.
    // Prevents BA from diverging to implausible positions when reprojection
    // observations are very sparse (dark turns, near-LOST states).
    // Jacobian is O(1) — numerically safe regardless of absolute pose magnitude.
    for (auto& kf : window) {
        if (kf == window.front()) continue;  // anchor is fixed — skip
        auto it = prior_poses.find(kf->id);
        if (it == prior_poses.end()) continue;
        problem.AddResidualBlock(
            PosePriorCost::Create(it->second.data(), 0.5, 0.5),
            nullptr, pose_params[kf->id].data());
    }

    // ── 5. Solve ──────────────────────────────────────────────────────────────
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = cfg_.max_iterations;
    options.minimizer_progress_to_stdout = cfg_.verbose;
    options.num_threads = 4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (cfg_.verbose) {
        std::cout << summary.BriefReport() << "\n";
    }

    // ── 6. Write back optimized poses ─────────────────────────────────────────
    for (auto& kf : window) {
        kf->T_cw = pose_to_isometry(pose_params[kf->id].data());
    }

    // ── 7. Write back optimized 3D point positions ────────────────────────────
    for (auto& [id, mp] : active_points) {
        auto& pt = point_params[id];
        mp->position = Eigen::Vector3d(pt[0], pt[1], pt[2]);
    }

    // ── 8. Post-BA reprojection culling ───────────────────────────────────────
    // Compute reprojection errors using the optimised poses and points still
    // available in pose_params / point_params.  Mark any map point whose error
    // exceeds 4 px in any window frame as bad.  This removes persistently noisy
    // points (small-baseline triangulations, RANSAC survivors) and keeps the map
    // clean for future tracking.
    {
        const double cull_thresh2 = 36.0;   // 6 px²
        for (auto& kf : window) {
            const double* pose = pose_params.at(kf->id).data();
            for (int kp_idx = 0; kp_idx < (int)kf->keypoints.size(); ++kp_idx) {
                auto& mp = kf->map_points[kp_idx];
                if (!mp || mp->is_bad) continue;
                auto pit = point_params.find(mp->id);
                if (pit == point_params.end()) continue;
                const double* pt = pit->second.data();

                double Xc[3];
                ceres::AngleAxisRotatePoint(pose, pt, Xc);
                Xc[0] += pose[3]; Xc[1] += pose[4]; Xc[2] += pose[5];
                if (Xc[2] <= 0.0) { mp->is_bad = true; continue; }

                // Cull stereo points beyond 150m — depth unreliable at that range.
                if (cam_.is_stereo() &&
                    kp_idx < (int)kf->uR.size() && kf->uR[kp_idx] >= 0.0f &&
                    Xc[2] > 150.0) {
                    mp->is_bad = true; continue;
                }

                double u  = cam_.fx * Xc[0] / Xc[2] + cam_.cx;
                double v  = cam_.fy * Xc[1] / Xc[2] + cam_.cy;
                double du = u - kf->keypoints[kp_idx].pt.x;
                double dv = v - kf->keypoints[kp_idx].pt.y;
                if (du*du + dv*dv > cull_thresh2) { mp->is_bad = true; continue; }

                // Stereo: also cull if right-camera reprojection exceeds threshold
                if (cam_.is_stereo() &&
                    kp_idx < (int)kf->uR.size() && kf->uR[kp_idx] >= 0.0f) {
                    double u_R  = cam_.fx * (Xc[0] - cam_.baseline) / Xc[2] + cam_.cx;
                    double dur  = u_R - kf->uR[kp_idx];
                    if (dur * dur > cull_thresh2) mp->is_bad = true;
                }
            }
        }
    }

    // Cleanup map points flagged as bad by other threads
    map_->cleanup_bad_map_points();
}

}  // namespace slam
