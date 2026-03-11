// pose_graph.cpp
//
// Pose Graph Optimization using co-visibility loop detection.
//
// Each edge stores the relative transform T_AB = T_A_cw * T_B_cw.inverse()
// (measured after local BA).  The PGO residual is the SE(3) log of the error:
//
//   delta = T_AB_meas * T_B_est * T_A_est.inverse()
//
// which should be Identity at optimum.  The 6-DOF residual is
// [w_r * omega_delta (3), w_t * t_delta (3)].

#include "slam/pose_graph.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <unordered_set>

namespace slam {

// ─── Relative Pose Cost (auto-diff, 6 residuals) ─────────────────────────────

struct RelPoseCost {
    double R_meas[9];  // row-major rotation of T_AB_meas
    double t_meas[3];
    double w_r;        // rotation weight
    double w_t;        // translation weight

    template <typename T>
    bool operator()(const T* const pa, const T* const pb, T* res) const
    {
        // Rotation matrices from angle-axis vectors
        T Ra[9], Rb[9];
        ceres::AngleAxisToRotationMatrix(pa, Ra);
        ceres::AngleAxisToRotationMatrix(pb, Rb);

        const T* ta = pa + 3;
        const T* tb = pb + 3;

        // Cast measurement (double → T)
        T Rm[9], tm[3];
        for (int i = 0; i < 9; ++i) Rm[i] = T(R_meas[i]);
        for (int i = 0; i < 3; ++i) tm[i] = T(t_meas[i]);

        // R_ba = Rb * Ra^T,   t_ba = tb - R_ba * ta
        T R_ba[9], t_ba[3];
        for (int r = 0; r < 3; ++r) {
            t_ba[r] = tb[r];
            for (int c = 0; c < 3; ++c) {
                R_ba[r*3+c] = T(0);
                for (int k = 0; k < 3; ++k)
                    R_ba[r*3+c] += Rb[r*3+k] * Ra[c*3+k];  // Ra^T[k,c] = Ra[c,k]
                t_ba[r] -= R_ba[r*3+c] * ta[c];
            }
        }

        // R_delta = Rm * R_ba,   t_delta = Rm * t_ba + tm
        T R_delta[9], t_delta[3];
        for (int r = 0; r < 3; ++r) {
            t_delta[r] = tm[r];
            for (int c = 0; c < 3; ++c) {
                R_delta[r*3+c] = T(0);
                for (int k = 0; k < 3; ++k)
                    R_delta[r*3+c] += Rm[r*3+k] * R_ba[k*3+c];
                t_delta[r] += Rm[r*3+c] * t_ba[c];
            }
        }

        // Angle-axis of R_delta
        T omega[3];
        ceres::RotationMatrixToAngleAxis(R_delta, omega);

        res[0] = T(w_r) * omega[0];
        res[1] = T(w_r) * omega[1];
        res[2] = T(w_r) * omega[2];
        res[3] = T(w_t) * t_delta[0];
        res[4] = T(w_t) * t_delta[1];
        res[5] = T(w_t) * t_delta[2];
        return true;
    }

    static ceres::CostFunction* Create(const double* R, const double* t,
                                       double w_r, double w_t)
    {
        auto* c = new RelPoseCost;
        for (int i = 0; i < 9; ++i) c->R_meas[i] = R[i];
        for (int i = 0; i < 3; ++i) c->t_meas[i] = t[i];
        c->w_r = w_r;
        c->w_t = w_t;
        return new ceres::AutoDiffCostFunction<RelPoseCost, 6, 6, 6>(c);
    }
};

// ─── Pose helpers ─────────────────────────────────────────────────────────────

static void isometry_to_pose(const Eigen::Isometry3d& T, double* pose)
{
    Eigen::AngleAxisd aa(T.rotation());
    Eigen::Vector3d om = aa.angle() * aa.axis();
    pose[0] = om.x(); pose[1] = om.y(); pose[2] = om.z();
    pose[3] = T.translation().x();
    pose[4] = T.translation().y();
    pose[5] = T.translation().z();
}

static Eigen::Isometry3d pose_to_isometry(const double* pose)
{
    Eigen::Vector3d om(pose[0], pose[1], pose[2]);
    double angle = om.norm();
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    if (angle > 1e-9)
        T.linear() = Eigen::AngleAxisd(angle, om / angle).toRotationMatrix();
    T.translation() << pose[3], pose[4], pose[5];
    return T;
}

// ─── PoseGraph implementation ─────────────────────────────────────────────────

PoseGraph::Ptr PoseGraph::create(Map::Ptr map, const Camera& cam, const Config& cfg)
{
    auto pg = std::shared_ptr<PoseGraph>(new PoseGraph());
    pg->cam_ = cam;
    pg->map_ = map;
    pg->cfg_ = cfg;
    return pg;
}

void PoseGraph::add_keyframe(Frame::Ptr kf)
{
    kf_order_.push_back(kf);
}

void PoseGraph::detect_and_add_loops()
{
    new_loops_ = false;
    if (kf_order_.empty()) return;

    Frame::Ptr query = kf_order_.back();

    // Collect IDs of KFs currently inside the local BA window (exclude from loop search)
    std::unordered_set<long> window_ids;
    for (auto& kf : map_->local_window())
        window_ids.insert(kf->id);

    for (auto& kf : map_->all_keyframes()) {
        if (kf->id == query->id) continue;
        if (window_ids.count(kf->id))  continue;  // inside BA window — skip

        int shared = map_->count_shared_map_points(kf->id, query->id);
        if (shared < cfg_.min_shared_points) continue;

        // Compute relative pose T_AB = T_A_cw * T_B_cw.inverse()
        Eigen::Isometry3d T_AB = kf->T_cw * query->T_cw.inverse();

        // Extract R (row-major) and t
        Edge e;
        e.id_a = kf->id;
        e.id_b = query->id;
        Eigen::Matrix3d R = T_AB.rotation();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                e.R_meas[r*3+c] = R(r, c);
        e.t_meas[0] = T_AB.translation().x();
        e.t_meas[1] = T_AB.translation().y();
        e.t_meas[2] = T_AB.translation().z();

        edges_.push_back(e);
        new_loops_ = true;

        std::cout << "[PGO] Loop edge: KF " << kf->id
                  << " ↔ KF " << query->id
                  << " (" << shared << " shared pts)\n";
    }
}

void PoseGraph::optimize()
{
    auto all_kfs = map_->all_keyframes();
    if (all_kfs.size() < 2 || edges_.empty()) return;

    // Allocate pose parameter blocks (one 6-vector per KF)
    std::unordered_map<long, std::vector<double>> pose_params;
    for (auto& kf : all_kfs) {
        pose_params[kf->id].resize(6);
        isometry_to_pose(kf->T_cw, pose_params[kf->id].data());
    }

    ceres::Problem problem;

    // Add relative-pose edges
    for (auto& e : edges_) {
        auto it_a = pose_params.find(e.id_a);
        auto it_b = pose_params.find(e.id_b);
        if (it_a == pose_params.end() || it_b == pose_params.end()) continue;

        problem.AddResidualBlock(
            RelPoseCost::Create(e.R_meas, e.t_meas, cfg_.w_r, cfg_.w_t),
            nullptr,
            it_a->second.data(),
            it_b->second.data());
    }

    if (problem.NumResidualBlocks() == 0) return;

    // Register all parameter blocks and fix oldest KF as gauge anchor
    for (auto& kf : all_kfs)
        problem.AddParameterBlock(pose_params[kf->id].data(), 6);
    problem.SetParameterBlockConstant(pose_params[all_kfs.front()->id].data());

    ceres::Solver::Options options;
    options.linear_solver_type        = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations        = cfg_.max_iterations;
    options.minimizer_progress_to_stdout = false;
    options.num_threads               = 4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Write back optimised poses
    for (auto& kf : all_kfs) {
        auto it = pose_params.find(kf->id);
        if (it != pose_params.end())
            kf->T_cw = pose_to_isometry(it->second.data());
    }

    std::cout << "[PGO] Optimized " << all_kfs.size()
              << " KFs with " << edges_.size() << " edges  ("
              << summary.BriefReport() << ")\n";

    new_loops_ = false;  // reset until next detection round
}

// ─── Appearance-based (descriptor) loop detection ────────────────────────────
//
// For each KF outside the local BA window (sampled every visual_sample_step),
// match its ORB descriptors against the query KF using BFMatcher + ratio test.
// If enough descriptors match, run PnP RANSAC using the candidate KF's 3D map
// points against the query KF's 2D keypoints to verify geometrically.
// A loop edge is added if PnP inliers >= cfg_.visual_min_inliers.
//
// This works even when old map points have no late-frame observations (the
// fundamental limitation of co-visibility detection), because it matches
// appearance descriptors directly across arbitrary KF pairs.

void PoseGraph::detect_and_add_loops_visual(Frame::Ptr /*query*/)
{
    // Visual loop detection disabled: raw BFMatcher produces false-positive
    // loop edges on repetitive scenes (KITTI urban) even with PnP verification.
    // Re-enable only when a proper visual vocabulary (DBoW) is available.
}

}  // namespace slam
