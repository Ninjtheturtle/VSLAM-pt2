#include "slam/tracker.hpp"
#include "slam/map_point.hpp"
#include "cuda/hamming_matcher.cuh"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/SVD>
#include <iostream>
#include <atomic>
#include <unordered_set>

namespace slam {

// ─── Global frame ID counter ──────────────────────────────────────────────────
static std::atomic<long> g_frame_id{0};
static std::atomic<long> g_point_id{0};

// ─── Factory ─────────────────────────────────────────────────────────────────

Tracker::Ptr Tracker::create(const Camera& cam, Map::Ptr map, const Config& cfg)
{
    auto t = std::shared_ptr<Tracker>(new Tracker());
    t->cam_ = cam;
    t->map_ = map;
    t->cfg_ = cfg;
    t->orb_ = cv::ORB::create(
        cfg.orb_features,
        cfg.orb_scale_factor,
        cfg.orb_levels,
        cfg.orb_edge_threshold
    );
    return t;
}

// ─── Main entry point ────────────────────────────────────────────────────────

bool Tracker::track(Frame::Ptr frame)
{
    // Extract ORB features on left image
    orb_->detectAndCompute(frame->image_gray, cv::noArray(),
                           frame->keypoints, frame->descriptors);
    frame->map_points.resize(frame->keypoints.size(), nullptr);

    // Extract ORB features on right image and perform stereo epipolar matching
    if (!frame->image_right.empty()) {
        orb_->detectAndCompute(frame->image_right, cv::noArray(),
                               frame->keypoints_right, frame->descriptors_right);
        frame->uR.assign(frame->keypoints.size(), -1.0f);
        match_stereo(frame);
    }

    // ── LOST: try to relocalize against the full map before reinitializing ─────
    if (state_ == TrackingState::LOST) {
        if (try_relocalize(frame)) {
            std::cout << "[Tracker] Relocalized successfully — resuming on existing map\n";
            velocity_valid_ = false;   // old velocity invalid after gap
            state_          = TrackingState::OK;
            last_frame_     = frame;
            return true;
        }
        // Relocalization failed — reset map and tracker state to avoid contaminating
        // the existing map with new keyframes placed at T_cw = Identity.  Without
        // this reset, BA would receive frames from two different world origins and
        // diverge immediately.
        std::cerr << "[Tracker] LOST — relocalization failed, resetting map + re-initializing\n";
        // Propagate pose so initialize() roots the new map at the last known world position.
        // Without this, frame->T_cw stays at Identity (Frame default) → trajectory resets to (0,0,0).
        frame->T_cw = velocity_valid_ ? (velocity_ * last_frame_->T_cw) : last_frame_->T_cw;
        map_->reset();
        last_keyframe_       = nullptr;
        last_kf_pnp_tracked_ = 0;
        velocity_valid_      = false;
        last_frame_          = frame;
        state_               = TrackingState::NOT_INITIALIZED;
        return false;
    }

    if (state_ == TrackingState::NOT_INITIALIZED) {
        // initialize() manages last_frame_ internally:
        //   - insufficient disparity → keep last_frame_ anchored (accumulate baseline)
        //   - other failures         → advance last_frame_ = frame
        //   - success                → advance last_frame_ = frame (after velocity_)
        return initialize(frame);
    }

    // OK state
    bool ok = track_with_motion_model(frame);
    if (ok) ok = track_local_map(frame);
    if (ok) {
        lost_streak_ = 0;
    } else {
        ++lost_streak_;
        if (lost_streak_ < 8) {
            // Short dead-reckoning: cover brief match failures without a full LOST reset.
            frame->T_cw = velocity_valid_
                ? velocity_ * last_frame_->T_cw
                : last_frame_->T_cw;
            last_frame_ = frame;
            return false;
        }
        lost_streak_ = 0;
        state_ = TrackingState::LOST;
    }
    last_frame_ = frame;
    return ok;
}

// ─── Initialization ──────────────────────────────────────────────────────────

bool Tracker::initialize(Frame::Ptr frame)
{
    // ── Stereo path: single-frame metric initialization ───────────────────────
    if (cam_.is_stereo() && !frame->uR.empty()) {
        // Propagate last known pose so reinit roots at current world position.
        // Without this, frame->T_cw is Identity (Frame default) and
        // triangulate_stereo() places all new map points at world origin.
        if (last_frame_) {
            frame->T_cw = last_frame_->T_cw;
        }
        int n_pts = triangulate_stereo(frame);
        if (n_pts < 50) {
            std::cerr << "[Tracker] Stereo init: too few points (" << n_pts << "), retrying\n";
            last_frame_ = frame;
            return false;
        }
        insert_keyframe(frame);
        velocity_valid_ = false;   // no velocity until second tracked frame
        state_          = TrackingState::OK;
        last_frame_     = frame;
        std::cout << "[Tracker] Stereo initialized: " << n_pts << " metric map points\n";
        return true;
    }

    // ── Monocular fallback: temporal two-frame initialization ─────────────────
    if (!last_frame_) {
        last_frame_ = frame;
        return false;
    }

    auto matches = match_descriptors(last_frame_->descriptors,
                                     frame->descriptors, /*ratio=*/true);
    if ((int)matches.size() < 50) {
        std::cerr << "[Tracker] Init: too few matches (" << matches.size()
                  << "), q=" << last_frame_->descriptors.rows
                  << " t=" << frame->descriptors.rows << "\n";
        last_frame_ = frame;
        return false;
    }

    std::vector<cv::Point2f> pts0, pts1;
    for (auto& m : matches) {
        pts0.push_back(last_frame_->keypoints[m.queryIdx].pt);
        pts1.push_back(frame->keypoints[m.trainIdx].pt);
    }

    // Require sufficient 2D feature displacement to avoid degenerate init
    {
        double sum_disp = 0.0;
        for (size_t i = 0; i < pts0.size(); ++i) {
            double dx = pts1[i].x - pts0[i].x;
            double dy = pts1[i].y - pts0[i].y;
            sum_disp += std::sqrt(dx*dx + dy*dy);
        }
        double mean_disp = sum_disp / pts0.size();
        if (mean_disp < 5.0) {
            std::cerr << "[Tracker] Init: low disparity (" << mean_disp
                      << " px, need 5.0), holding anchor\n";
            return false;
        }
    }

    cv::Mat E, mask;
    E = cv::findEssentialMat(pts0, pts1,
                             cam_.K_cv(), cv::RANSAC,
                             0.999, 1.0, 1000, mask);
    if (E.empty()) {
        std::cerr << "[Tracker] Init: essential matrix empty\n";
        last_frame_ = frame; return false;
    }

    cv::Mat R_cv, t_cv;
    int inliers = cv::recoverPose(E, pts0, pts1,
                                  cam_.K_cv(), R_cv, t_cv, mask);
    if (inliers < 20) {
        std::cerr << "[Tracker] Init: recoverPose inliers too low (" << inliers << ")\n";
        last_frame_ = frame; return false;
    }

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            R(r, c) = R_cv.at<double>(r, c);
    t << t_cv.at<double>(0), t_cv.at<double>(1), t_cv.at<double>(2);

    Eigen::Isometry3d T_rel = Eigen::Isometry3d::Identity();
    T_rel.linear()      = R;
    T_rel.translation() = t;
    frame->T_cw = T_rel * last_frame_->T_cw;

    int n_pts = triangulate_and_add(last_frame_, frame, matches);
    if (n_pts < 20) {
        std::cerr << "[Tracker] Init: triangulation too sparse (" << n_pts << " pts)\n";
        last_frame_ = frame; return false;
    }

    // Scale to ~20 m median depth (monocular scale ambiguity workaround)
    {
        std::vector<double> depths;
        depths.reserve(n_pts);
        for (auto& mp : map_->all_map_points()) {
            Eigen::Vector3d Xc = frame->T_cw * mp->position;
            if (Xc.z() > 0.0) depths.push_back(Xc.z());
        }
        if (!depths.empty()) {
            std::sort(depths.begin(), depths.end());
            double median = depths[depths.size() / 2];
            double scale  = 20.0 / median;
            Eigen::Vector3d C0 = last_frame_->camera_center();
            Eigen::Vector3d C1 = frame->camera_center();
            Eigen::Vector3d C1_s = C0 + scale * (C1 - C0);
            frame->T_cw.translation() = -frame->T_cw.linear() * C1_s;
            for (auto& mp : map_->all_map_points())
                mp->position = C0 + scale * (mp->position - C0);
        }
    }

    insert_keyframe(last_frame_);
    insert_keyframe(frame);

    velocity_       = frame->T_cw * last_frame_->T_cw.inverse();
    velocity_valid_ = true;
    state_          = TrackingState::OK;
    last_frame_     = frame;

    std::cout << "[Tracker] Monocular initialized with " << n_pts << " map points\n";
    return true;
}

// ─── Tracking with constant-velocity model ────────────────────────────────────
//
// Builds 3D-2D correspondences by GPU-matching descriptors from ALL local
// keyframes' map points against the current frame.  This avoids the rapid
// attrition that occurs when only last_frame_'s sparse point set is used,
// while preserving descriptor-based verification (no false geometric matches).

bool Tracker::track_with_motion_model(Frame::Ptr frame)
{
    // Predict pose
    if (velocity_valid_) {
        frame->T_cw = velocity_ * last_frame_->T_cw;
    } else {
        frame->T_cw = last_frame_->T_cw;
    }

    // Build descriptor pool: one row per unique map point from local keyframes.
    // We use the keyframe's stored ORB descriptor row for each mapped keypoint.
    cv::Mat                    pool_desc;
    std::vector<MapPoint::Ptr> pool_mps;
    {
        std::unordered_set<long> seen_ids;
        // Use a wider window (30 KFs) so that when KFs are inserted rapidly
        // (small baseline, low-texture scene), the pool still has enough
        // diverse 3D-2D candidates for robust PnP.
        for (auto& kf : map_->local_window(30)) {
            if (kf->descriptors.empty()) continue;
            for (int i = 0; i < (int)kf->map_points.size(); ++i) {
                auto& mp = kf->map_points[i];
                if (!mp || mp->is_bad) continue;
                if (mp->observed_times < 2) continue; // skip single-observation (unverified) points
                if (!seen_ids.insert(mp->id).second) continue; // skip duplicates
                if (i >= kf->descriptors.rows) continue;
                pool_desc.push_back(kf->descriptors.row(i));
                pool_mps.push_back(mp);
            }
        }
    }
    if ((int)pool_desc.rows < cfg_.pnp_min_inliers) {
        std::cerr << "[Tracker] Track: pool too small (" << pool_desc.rows << " pts)\n";
        return false;
    }

    // ─── Correspondence building: two-phase matching ──────────────────────────
    //   Phase 1 (preferred): project pool map points with predicted T_cw and
    //     search for nearest unmatched keypoint within search_r px (no ratio test).
    //     Spatial proximity replaces the ratio test, making this robust to
    //     repetitive-texture roads and sharp turns.
    //   Phase 2 (fallback): GPU Hamming + ratio test when Phase 1 yields too few.
    // ─────────────────────────────────────────────────────────────────────────────
    std::vector<cv::Point3f>   pts3d;
    std::vector<cv::Point2f>   pts2d;
    std::vector<int>           match_idxs;
    std::vector<MapPoint::Ptr> match_mps;
    std::unordered_set<int>    used_kp;

    // ── Phase 1: projection-based spatial matching ────────────────────────────
    {
        const int   cell     = 16;
        const int   frame_w  = frame->image_gray.cols;
        const int   frame_h  = frame->image_gray.rows;
        const int   n_cols_g = (frame_w + cell - 1) / cell;
        const int   n_rows_g = (frame_h + cell - 1) / cell;
        // Widen the search window when the motion model predicts significant rotation (turn).
        // Constant-velocity extrapolation overshoots at turns; the extra radius compensates so
        // that Phase 1 still finds ~300+ matches instead of collapsing to ~50.
        //   0°/frame  →  40 px   (straight road)
        //   1.7°/frame → 60 px
        //   3.4°/frame → 80 px
        //   ≥5.2°/frame → 120 px (cap)
        const double pred_ang = velocity_valid_
            ? Eigen::AngleAxisd(velocity_.rotation()).angle() : 0.0;
        // When velocity is invalid (one frame after BA), widen base radius to compensate
        // for 1-frame camera displacement without a rotation prediction.
        const float base_r = velocity_valid_ ? 40.0f : 70.0f;
        const float search_r = (pred_ang > 0.03)
            ? std::min(120.0f, base_r + float(pred_ang / 0.03) * 20.0f)
            : base_r;
        const int   max_ham  = cfg_.hamming_threshold;

        // Build spatial grid: cell → list of keypoint indices
        std::vector<std::vector<int>> kp_grid(n_cols_g * n_rows_g);
        for (int j = 0; j < (int)frame->keypoints.size(); ++j) {
            int cx = std::min(n_cols_g - 1, (int)(frame->keypoints[j].pt.x / cell));
            int cy = std::min(n_rows_g - 1, (int)(frame->keypoints[j].pt.y / cell));
            if (cx >= 0 && cy >= 0) kp_grid[cy * n_cols_g + cx].push_back(j);
        }

        for (int mi = 0; mi < (int)pool_mps.size(); ++mi) {
            auto& mp = pool_mps[mi];
            Eigen::Vector3d Xc = frame->T_cw * mp->position;   // predicted T_cw
            if (Xc.z() <= 0.0) continue;
            float u = (float)(cam_.fx * Xc.x() / Xc.z() + cam_.cx);
            float v = (float)(cam_.fy * Xc.y() / Xc.z() + cam_.cy);
            if (u < 0 || u >= frame_w || v < 0 || v >= frame_h) continue;

            int cx0 = std::max(0,            (int)((u - search_r) / cell));
            int cx1 = std::min(n_cols_g - 1, (int)((u + search_r) / cell));
            int cy0 = std::max(0,            (int)((v - search_r) / cell));
            int cy1 = std::min(n_rows_g - 1, (int)((v + search_r) / cell));

            int best_j = -1, best_d = max_ham + 1;
            for (int gy = cy0; gy <= cy1; ++gy)
                for (int gx = cx0; gx <= cx1; ++gx)
                    for (int kp_j : kp_grid[gy * n_cols_g + gx]) {
                        if (used_kp.count(kp_j)) continue;
                        int d = cv::norm(pool_desc.row(mi),
                                         frame->descriptors.row(kp_j),
                                         cv::NORM_HAMMING);
                        if (d < best_d) { best_d = d; best_j = kp_j; }
                    }

            if (best_j < 0) continue;
            if (!used_kp.insert(best_j).second) continue;
            auto& p = mp->position;
            pts3d.push_back({(float)p.x(), (float)p.y(), (float)p.z()});
            pts2d.push_back(frame->keypoints[best_j].pt);
            match_idxs.push_back(best_j);
            match_mps.push_back(mp);
        }
        std::cout << "[Tracker] Proj-match: " << pts3d.size()
                  << "/" << pool_mps.size() << " pts\n";
    }

    // ── Phase 2: GPU Hamming fallback ─────────────────────────────────────────
    if ((int)pts3d.size() < cfg_.pnp_min_inliers) {
        pts3d.clear(); pts2d.clear(); match_idxs.clear(); match_mps.clear(); used_kp.clear();
        auto raw_matches = match_descriptors(pool_desc, frame->descriptors, /*use_ratio=*/true);
        for (auto& m : raw_matches) {
            if (!used_kp.insert(m.trainIdx).second) continue;
            auto& mp = pool_mps[m.queryIdx];
            auto& p  = mp->position;
            pts3d.push_back({(float)p.x(), (float)p.y(), (float)p.z()});
            pts2d.push_back(frame->keypoints[m.trainIdx].pt);
            match_idxs.push_back(m.trainIdx);
            match_mps.push_back(mp);
        }
    }

    if ((int)pts3d.size() < cfg_.pnp_min_inliers) {
        std::cerr << "[Tracker] Track: too few correspondences (" << pts3d.size() << ")\n";
        return false;
    }

    // PnP RANSAC with velocity-predicted pose as initial guess
    cv::Mat rvec(3, 1, CV_64F), tvec(3, 1, CV_64F), inlier_mask;
    {
        Eigen::AngleAxisd   aa(frame->T_cw.rotation());
        Eigen::Vector3d     ax = aa.angle() * aa.axis();
        rvec.at<double>(0) = ax.x();
        rvec.at<double>(1) = ax.y();
        rvec.at<double>(2) = ax.z();
        tvec.at<double>(0) = frame->T_cw.translation().x();
        tvec.at<double>(1) = frame->T_cw.translation().y();
        tvec.at<double>(2) = frame->T_cw.translation().z();
    }
    // Only use the velocity prediction as an initial guess when velocity is
    // actually valid AND the predicted rotation is small.  At high angular
    // rates (sharp turns) the constant-velocity extrapolation overshoots;
    // using it as a RANSAC seed biases hypothesis sampling away from the true
    // pose.  With useExtrinsicGuess=false RANSAC searches freely.
    // Threshold: 0.3 rad ≈ 17° — well above normal driving but below overshoot zone.
    const double pred_angle = velocity_valid_
        ? Eigen::AngleAxisd(velocity_.rotation()).angle() : 0.0;
    const bool use_guess = velocity_valid_ && (pred_angle < 0.3);

    bool ok = cv::solvePnPRansac(
        pts3d, pts2d, cam_.K_cv(), cam_.dist_cv(),
        rvec, tvec, /*useExtrinsicGuess=*/use_guess,
        cfg_.pnp_iterations, cfg_.pnp_reprojection, 0.99,
        inlier_mask, cv::SOLVEPNP_SQPNP);
    if (!ok) {
        std::cerr << "[Tracker] Track: solvePnPRansac failed (" << pts3d.size() << " corr)\n";
        return false;
    }

    // solvePnPRansac returns CV_32S index vector OR CV_8U mask depending on OpenCV version.
    std::vector<bool> is_inlier(pts3d.size(), false);
    int n_inliers = 0;
    if (inlier_mask.type() == CV_8U) {
        // Mask mode: element k is 1 for inlier, 0 for outlier
        for (int k = 0; k < inlier_mask.rows && k < (int)pts3d.size(); ++k)
            if (inlier_mask.at<uint8_t>(k)) { is_inlier[k] = true; ++n_inliers; }
    } else {
        // Index mode (CV_32S): each element is an inlier index into pts3d
        for (int k = 0; k < inlier_mask.rows; ++k) {
            int idx = inlier_mask.at<int>(k);
            if (idx >= 0 && idx < (int)pts3d.size()) { is_inlier[idx] = true; ++n_inliers; }
        }
    }
    if (n_inliers < cfg_.pnp_min_inliers) {
        std::cerr << "[Tracker] Track: PnP inliers too few (" << n_inliers << ")\n";
        return false;
    }

    // LM refinement on inliers
    std::vector<cv::Point3f> in3d;
    std::vector<cv::Point2f> in2d;
    for (int i = 0; i < (int)pts3d.size(); ++i)
        if (is_inlier[i]) { in3d.push_back(pts3d[i]); in2d.push_back(pts2d[i]); }
    cv::solvePnP(in3d, in2d, cam_.K_cv(), cam_.dist_cv(),
                 rvec, tvec, /*useExtrinsicGuess=*/true, cv::SOLVEPNP_ITERATIVE);

    // ── Build candidate pose from rvec/tvec ──────────────────────────────────
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    Eigen::Isometry3d T_cw_candidate = Eigen::Isometry3d::Identity();
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            T_cw_candidate.linear()(r, c) = R_cv.at<double>(r, c);
    T_cw_candidate.translation() << tvec.at<double>(0),
                                     tvec.at<double>(1),
                                     tvec.at<double>(2);

    // ── Delta-rotation & translation sanity checks ────────────────────────────
    // A driving car cannot physically rotate >~29° or translate >5 m between
    // consecutive frames at 10 Hz.  Rejecting implausible solutions here
    // prevents bad poses from corrupting velocity_.
    {
        Eigen::Isometry3d delta = T_cw_candidate * last_frame_->T_cw.inverse();
        double delta_angle = Eigen::AngleAxisd(delta.rotation()).angle();
        if (delta_angle > 0.5) {    // 0.5 rad ≈ 29° — allows combined yaw+roll/pitch error at tight corners
            std::cerr << "[Tracker] PnP rejected: delta rot "
                      << (delta_angle * (180.0 / 3.14159265358979323846)) << " deg\n";
            return false;
        }
        double delta_trans = delta.translation().norm();
        if (delta_trans > 3.0) {   // >3 m/frame ≈ 108 km/h at 10 Hz
            std::cerr << "[Tracker] PnP rejected: delta trans "
                      << delta_trans << " m\n";
            return false;
        }
    }

    // Commit pose
    frame->T_cw = T_cw_candidate;

    // Assign inlier map points
    for (int i = 0; i < (int)pts3d.size(); ++i)
        if (is_inlier[i])
            frame->map_points[match_idxs[i]] = match_mps[i];

    // ── Project-and-search: augment tracked point count ──────────────────────
    // After PnP sets frame->T_cw, project all local map points that were NOT
    // tried by the initial descriptor-pool matching.  For each projected point
    // whose location is within 15 px of an unmatched keypoint AND whose
    // descriptor distance is below threshold, assign the map point.
    // This increases num_tracked() 3-5× without an additional RANSAC pass,
    // leading to fewer keyframe insertions and a denser, more stable map.
    {
        const int   cell      = 16;
        const int   frame_w   = frame->image_gray.cols;
        const int   frame_h   = frame->image_gray.rows;
        const int   n_cols_g  = (frame_w + cell - 1) / cell;
        const int   n_rows_g  = (frame_h + cell - 1) / cell;
        const float search_r  = 15.0f;
        const int   max_ham   = 50;
        const float max_repr2 = 25.0f;   // 5 px validation threshold²

        // Spatial grid: cell → list of keypoint indices that still need a map point
        std::vector<std::vector<int>> kp_grid(n_cols_g * n_rows_g);
        for (int j = 0; j < (int)frame->keypoints.size(); ++j) {
            if (frame->map_points[j]) continue;
            if (used_kp.count(j)) continue;
            int cx = std::min(n_cols_g - 1, (int)(frame->keypoints[j].pt.x / cell));
            int cy = std::min(n_rows_g - 1, (int)(frame->keypoints[j].pt.y / cell));
            if (cx >= 0 && cy >= 0) kp_grid[cy * n_cols_g + cx].push_back(j);
        }

        // Track which map-point IDs were already tried in the descriptor-pool phase
        std::unordered_set<long> proj_seen;
        for (auto& mp : pool_mps) proj_seen.insert(mp->id);

        for (auto& kf : map_->local_window(30)) {
            if (kf->descriptors.empty()) continue;
            for (int i = 0; i < (int)kf->map_points.size(); ++i) {
                auto& mp = kf->map_points[i];
                if (!mp || mp->is_bad) continue;
                if (!proj_seen.insert(mp->id).second) continue;  // already tried
                if (i >= kf->descriptors.rows) continue;

                // Project into frame using committed T_cw
                Eigen::Vector3d Xc = frame->T_cw * mp->position;
                if (Xc.z() <= 0.0) continue;
                float u = (float)(cam_.fx * Xc.x() / Xc.z() + cam_.cx);
                float v = (float)(cam_.fy * Xc.y() / Xc.z() + cam_.cy);
                if (u < 0 || u >= frame_w || v < 0 || v >= frame_h) continue;

                // Search grid cells in (u±search_r, v±search_r)
                int cx0 = std::max(0,            (int)((u - search_r) / cell));
                int cx1 = std::min(n_cols_g - 1, (int)((u + search_r) / cell));
                int cy0 = std::max(0,            (int)((v - search_r) / cell));
                int cy1 = std::min(n_rows_g - 1, (int)((v + search_r) / cell));

                int best_j = -1, best_d = max_ham;
                for (int gy = cy0; gy <= cy1; ++gy)
                    for (int gx = cx0; gx <= cx1; ++gx)
                        for (int kp_j : kp_grid[gy * n_cols_g + gx]) {
                            int d = cv::norm(kf->descriptors.row(i),
                                            frame->descriptors.row(kp_j),
                                            cv::NORM_HAMMING);
                            if (d < best_d) { best_d = d; best_j = kp_j; }
                        }

                if (best_j < 0 || used_kp.count(best_j)) continue;

                // Validate with reprojection error at committed pose
                float du = u - frame->keypoints[best_j].pt.x;
                float dv = v - frame->keypoints[best_j].pt.y;
                if (du*du + dv*dv > max_repr2) continue;

                used_kp.insert(best_j);
                frame->map_points[best_j] = mp;
            }
        }
    }

    velocity_       = frame->T_cw * last_frame_->T_cw.inverse();
    velocity_valid_ = true;
    return true;
}

// ─── Local map tracking ───────────────────────────────────────────────────────

bool Tracker::track_local_map(Frame::Ptr frame)
{
    if (need_new_keyframe(frame)) {
        insert_keyframe(frame);
    }
    return frame->num_tracked() >= cfg_.pnp_min_inliers;
}

// ─── Keyframe decision ────────────────────────────────────────────────────────

bool Tracker::need_new_keyframe(Frame::Ptr frame) const
{
    if (!last_keyframe_) return true;

    int tracked = frame->num_tracked();  // PnP inliers only (not yet triangulated)

    // Use last_kf_pnp_tracked_ — the PnP-inlier count saved BEFORE triangulation
    // at the last keyframe insertion.  Using last_keyframe_->num_tracked() would
    // be inflated by newly triangulated points, making the ratio always < 0.8.
    if (tracked < cfg_.min_tracked_points) return true;
    if (last_kf_pnp_tracked_ > 0 && (float)tracked / last_kf_pnp_tracked_ < 0.8f) return true;

    // Parallax trigger removed: at KITTI 10 Hz, parallax > 1° fires every frame →
    // every frame becomes a KF → BA invalidates velocity_ every frame → Phase 1
    // always uses zero-rotation prediction → turn features fall outside search radius
    // → LOST cascade.  Count-based triggers (tracked < 80, ratio < 0.8) are sufficient.

    return false;
}

double Tracker::compute_median_parallax(Frame::Ptr frame, Frame::Ptr ref_kf) const
{
    if (!ref_kf) return 0.0;
    const Eigen::Vector3d C_ref = ref_kf->camera_center();
    const Eigen::Vector3d C_cur = frame->camera_center();
    std::vector<double> angles;
    for (int i = 0; i < (int)frame->map_points.size(); ++i) {
        auto& mp = frame->map_points[i];
        if (!mp || mp->is_bad) continue;
        Eigen::Vector3d v1 = (mp->position - C_ref).normalized();
        Eigen::Vector3d v2 = (mp->position - C_cur).normalized();
        double cos_a = std::max(-1.0, std::min(1.0, v1.dot(v2)));
        angles.push_back(std::acos(std::abs(cos_a)));
    }
    if (angles.empty()) return 0.0;
    std::sort(angles.begin(), angles.end());
    return angles[angles.size() / 2];
}

void Tracker::insert_keyframe(Frame::Ptr frame)
{
    // Save PnP-inlier count BEFORE triangulation inflates frame->map_points.
    last_kf_pnp_tracked_ = frame->num_tracked();

    frame->id = g_frame_id++;
    frame->is_keyframe = true;

    // Triangulate new map points between this keyframe and the last 3 KFs.
    // Iterating oldest-first gives the KF with the largest baseline first pick of
    // unmatched keypoints; newer KFs fill in the rest.
    // The !frame->map_points[m.trainIdx] guard prevents double-assignment.
    for (auto& tri_kf : map_->local_window(3)) {
        if (tri_kf->id == frame->id) continue;
        auto kf_matches = match_descriptors(tri_kf->descriptors,
                                            frame->descriptors, /*ratio=*/true);
        std::vector<cv::DMatch> new_matches;
        for (auto& m : kf_matches) {
            if (m.trainIdx < (int)frame->map_points.size() &&
                !frame->map_points[m.trainIdx]) {
                new_matches.push_back(m);
            }
        }
        if (!new_matches.empty()) {
            int n_new = triangulate_and_add(tri_kf, frame, new_matches);
            if (n_new > 0)
                std::cout << "[Tracker] KF " << frame->id
                          << ": triangulated " << n_new
                          << " pts vs KF " << tri_kf->id << "\n";
        }
    }

    // Stereo enrichment: add metric-depth map points for any keypoint still
    // unmapped.  This keeps the map dense with scale-consistent points across
    // all keyframes, not just the initialization frame.
    if (cam_.is_stereo() && !frame->uR.empty()) {
        int n_stereo = triangulate_stereo(frame);
        if (n_stereo > 0)
            std::cout << "[Tracker] KF " << frame->id
                      << ": stereo added " << n_stereo << " metric pts\n";
    }

    map_->insert_keyframe(frame);
    last_keyframe_ = frame;
}

// ─── GPU Descriptor Matching ─────────────────────────────────────────────────

std::vector<cv::DMatch> Tracker::match_descriptors(
    const cv::Mat& query_desc,
    const cv::Mat& train_desc,
    bool use_ratio)
{
    int N_q = query_desc.rows;
    int N_t = train_desc.rows;

    if (N_q == 0 || N_t == 0) return {};

    // Ensure descriptors are continuous and CV_8U
    cv::Mat q = query_desc.isContinuous() ? query_desc : query_desc.clone();
    cv::Mat t = train_desc.isContinuous()  ? train_desc  : train_desc.clone();

    std::vector<int> best_idx(N_q, -1);
    std::vector<int> best_dist(N_q, kMaxHamming);

    if (use_ratio) {
        cuda_match_hamming_ratio(
            q.data, t.data, N_q, N_t, cfg_.lowe_ratio,
            best_idx.data(), best_dist.data());
    } else {
        cuda_match_hamming(
            q.data, t.data, N_q, N_t,
            best_idx.data(), best_dist.data());
    }

    std::vector<cv::DMatch> matches;
    matches.reserve(N_q);
    for (int i = 0; i < N_q; ++i) {
        if (best_idx[i] >= 0 && best_dist[i] <= cfg_.hamming_threshold) {
            matches.push_back(cv::DMatch(i, best_idx[i], (float)best_dist[i]));
        }
    }
    return matches;
}

// ─── Triangulation ───────────────────────────────────────────────────────────

int Tracker::triangulate_and_add(Frame::Ptr ref, Frame::Ptr cur,
                                  const std::vector<cv::DMatch>& matches)
{
    // Build projection matrices (3×4)
    auto make_proj = [&](const Eigen::Isometry3d& T_cw) -> cv::Mat {
        cv::Mat P(3, 4, CV_64F);
        Eigen::Matrix<double, 3, 4> Rt;
        Rt.block<3,3>(0,0) = T_cw.rotation();
        Rt.block<3,1>(0,3) = T_cw.translation();
        Eigen::Matrix<double, 3, 4> KRt = cam_.K() * Rt;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++)
                P.at<double>(r, c) = KRt(r, c);
        return P;
    };

    cv::Mat P0 = make_proj(ref->T_cw);
    cv::Mat P1 = make_proj(cur->T_cw);

    // Collect matched point pairs
    std::vector<cv::Point2f> pts0, pts1;
    std::vector<int>         ref_kp_idxs, cur_kp_idxs;
    for (auto& m : matches) {
        pts0.push_back(ref->keypoints[m.queryIdx].pt);
        pts1.push_back(cur->keypoints[m.trainIdx].pt);
        ref_kp_idxs.push_back(m.queryIdx);
        cur_kp_idxs.push_back(m.trainIdx);
    }

    cv::Mat pts4d;
    cv::triangulatePoints(P0, P1, pts0, pts1, pts4d);  // 4×N homogeneous

    int n_added = 0;
    for (int i = 0; i < pts4d.cols; ++i) {
        float w = pts4d.at<float>(3, i);   // triangulatePoints outputs CV_32F
        if (std::abs(w) < 1e-6f) continue;

        Eigen::Vector3d Xw(pts4d.at<float>(0, i) / w,
                           pts4d.at<float>(1, i) / w,
                           pts4d.at<float>(2, i) / w);

        // Depth check in both cameras
        Eigen::Vector3d Xc0 = ref->T_cw * Xw;
        Eigen::Vector3d Xc1 = cur->T_cw * Xw;
        if (Xc0.z() < 0.05 || Xc1.z() < 0.05) continue;
        if (Xc0.z() > 200.0 || Xc1.z() > 200.0) continue;

        // Parallax check: skip near-degenerate triangulations (< ~1.1°).
        // Points with tiny parallax have depth uncertainty of 100s of metres
        // and pollute BA.
        {
            Eigen::Vector3d O0 = ref->camera_center();
            Eigen::Vector3d O1 = cur->camera_center();
            double cos_pa = std::abs((Xw - O0).normalized().dot((Xw - O1).normalized()));
            if (cos_pa > 0.9998) continue;
        }

        auto mp = MapPoint::create(Xw, g_point_id++);
        mp->add_observation(ref->id, ref_kp_idxs[i]);
        mp->add_observation(cur->id, cur_kp_idxs[i]);

        ref->map_points[ref_kp_idxs[i]] = mp;
        cur->map_points[cur_kp_idxs[i]] = mp;

        map_->insert_map_point(mp);
        ++n_added;
    }
    return n_added;
}

// ─── Relocalization ───────────────────────────────────────────────────────────
//
// When LOST, match the current frame against ALL keyframes' map points.
// Key differences from track_with_motion_model:
//   • Pool from map_->all_keyframes() — full map, not just last 30 KFs
//   • useExtrinsicGuess=false — no valid velocity to warm-start from
//   • Stricter inlier threshold: pnp_min_inliers * 3 = 45

bool Tracker::try_relocalize(Frame::Ptr frame)
{
    // ── Build descriptor pool from ALL keyframes ──────────────────────────────
    cv::Mat                    pool_desc;
    std::vector<MapPoint::Ptr> pool_mps;
    {
        std::unordered_set<long> seen_ids;
        for (auto& kf : map_->all_keyframes()) {
            if (kf->descriptors.empty()) continue;
            for (int i = 0; i < (int)kf->map_points.size(); ++i) {
                auto& mp = kf->map_points[i];
                if (!mp || mp->is_bad) continue;
                if (!seen_ids.insert(mp->id).second) continue;
                if (i >= kf->descriptors.rows) continue;
                pool_desc.push_back(kf->descriptors.row(i));
                pool_mps.push_back(mp);
            }
        }
    }
    if (pool_desc.rows < cfg_.pnp_min_inliers) return false;

    std::cout << "[Reloc] Matching against " << pool_desc.rows << " global map pts\n";

    // ── GPU descriptor matching ───────────────────────────────────────────────
    auto raw_matches = match_descriptors(pool_desc, frame->descriptors, true);

    // ── Build 3D-2D correspondences ───────────────────────────────────────────
    std::vector<cv::Point3f>   pts3d;
    std::vector<cv::Point2f>   pts2d;
    std::vector<int>           match_idxs;
    std::vector<MapPoint::Ptr> match_mps;
    std::unordered_set<int>    used_kp;

    for (auto& m : raw_matches) {
        if (!used_kp.insert(m.trainIdx).second) continue;
        auto& mp = pool_mps[m.queryIdx];
        auto& p  = mp->position;
        pts3d.push_back({(float)p.x(), (float)p.y(), (float)p.z()});
        pts2d.push_back(frame->keypoints[m.trainIdx].pt);
        match_idxs.push_back(m.trainIdx);
        match_mps.push_back(mp);
    }
    if ((int)pts3d.size() < cfg_.pnp_min_inliers) {
        std::cerr << "[Reloc] Too few correspondences (" << pts3d.size() << ")\n";
        return false;
    }
    std::cout << "[Reloc] " << pts3d.size() << " 3D-2D correspondences\n";

    // ── PnP RANSAC — no initial guess ─────────────────────────────────────────
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat inlier_mask;

    bool ok = cv::solvePnPRansac(
        pts3d, pts2d, cam_.K_cv(), cam_.dist_cv(),
        rvec, tvec, /*useExtrinsicGuess=*/false,
        cfg_.pnp_iterations, cfg_.pnp_reprojection, 0.99,
        inlier_mask, cv::SOLVEPNP_SQPNP);

    if (!ok) { std::cerr << "[Reloc] solvePnPRansac failed\n"; return false; }

    std::vector<bool> is_inlier(pts3d.size(), false);
    int n_reloc_inliers = 0;
    if (inlier_mask.type() == CV_8U) {
        for (int k = 0; k < inlier_mask.rows && k < (int)pts3d.size(); ++k)
            if (inlier_mask.at<uint8_t>(k)) { is_inlier[k] = true; ++n_reloc_inliers; }
    } else {
        for (int k = 0; k < inlier_mask.rows; ++k) {
            int idx = inlier_mask.at<int>(k);
            if (idx >= 0 && idx < (int)pts3d.size()) { is_inlier[idx] = true; ++n_reloc_inliers; }
        }
    }

    const int reloc_min = cfg_.pnp_min_inliers * 2;  // 30
    if (n_reloc_inliers < reloc_min) {
        std::cerr << "[Reloc] Inliers too few (" << n_reloc_inliers
                  << " < " << reloc_min << ")\n";
        return false;
    }

    // ── LM refinement on inliers ──────────────────────────────────────────────
    std::vector<cv::Point3f> in3d;
    std::vector<cv::Point2f> in2d;
    for (int i = 0; i < (int)pts3d.size(); ++i)
        if (is_inlier[i]) { in3d.push_back(pts3d[i]); in2d.push_back(pts2d[i]); }
    cv::solvePnP(in3d, in2d, cam_.K_cv(), cam_.dist_cv(),
                 rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

    // ── Update frame pose ─────────────────────────────────────────────────────
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            frame->T_cw.linear()(r, c) = R_cv.at<double>(r, c);
    frame->T_cw.translation() << tvec.at<double>(0),
                                  tvec.at<double>(1),
                                  tvec.at<double>(2);

    // ── Assign inlier map points ──────────────────────────────────────────────
    for (int i = 0; i < (int)pts3d.size(); ++i)
        if (is_inlier[i])
            frame->map_points[match_idxs[i]] = match_mps[i];

    std::cout << "[Reloc] SUCCESS — " << inlier_mask.rows << " inliers\n";
    return true;
}

// ─── Stereo Epipolar Matching ─────────────────────────────────────────────────

void Tracker::match_stereo(Frame::Ptr frame)
{
    if (frame->descriptors.empty() || frame->descriptors_right.empty()) return;
    int N_q = frame->descriptors.rows;
    int N_t = frame->descriptors_right.rows;
    if (N_q == 0 || N_t == 0) return;

    std::vector<float> y_q(N_q), y_t(N_t), x_q(N_q), x_t(N_t);
    for (int i = 0; i < N_q; ++i) {
        y_q[i] = frame->keypoints[i].pt.y;
        x_q[i] = frame->keypoints[i].pt.x;
    }
    for (int i = 0; i < N_t; ++i) {
        y_t[i] = frame->keypoints_right[i].pt.y;
        x_t[i] = frame->keypoints_right[i].pt.x;
    }

    cv::Mat q = frame->descriptors.isContinuous()       ? frame->descriptors       : frame->descriptors.clone();
    cv::Mat t = frame->descriptors_right.isContinuous() ? frame->descriptors_right : frame->descriptors_right.clone();

    std::vector<int> best_idx(N_q, -1), best_dist(N_q, kMaxHamming);
    cuda_match_stereo_epipolar(
        q.data, t.data, N_q, N_t,
        y_q.data(), y_t.data(), x_q.data(), x_t.data(),
        cfg_.stereo_epi_tol, cfg_.stereo_d_min, cfg_.stereo_d_max,
        cfg_.lowe_ratio, best_idx.data(), best_dist.data());

    for (int i = 0; i < N_q; ++i) {
        if (best_idx[i] >= 0 && best_dist[i] <= cfg_.hamming_threshold) {
            frame->uR[i] = frame->keypoints_right[best_idx[i]].pt.x;
        }
    }
}

// ─── Stereo Triangulation ─────────────────────────────────────────────────────

int Tracker::triangulate_stereo(Frame::Ptr frame)
{
    if (frame->uR.empty()) return 0;
    int n_added = 0;
    for (int i = 0; i < (int)frame->keypoints.size(); ++i) {
        if (frame->uR[i] < 0.0f) continue;      // no stereo match
        if (frame->map_points[i]) continue;       // already has a map point

        float u_L = frame->keypoints[i].pt.x;
        float v_L = frame->keypoints[i].pt.y;
        float u_R = frame->uR[i];
        float d   = u_L - u_R;
        if (d < cfg_.stereo_d_min || d > cfg_.stereo_d_max) continue;

        double Z = cam_.fx * cam_.baseline / (double)d;
        double X = ((double)u_L - cam_.cx) * Z / cam_.fx;
        double Y = ((double)v_L - cam_.cy) * Z / cam_.fy;
        if (Z < 0.5 || Z > 150.0) continue;

        Eigen::Vector3d Xw = frame->T_cw.inverse() * Eigen::Vector3d(X, Y, Z);

        auto mp = MapPoint::create(Xw, g_point_id++);
        mp->observed_times = 2;   // stereo = two-view constraint; treat as verified
        frame->map_points[i] = mp;
        map_->insert_map_point(mp);
        ++n_added;
    }
    return n_added;
}

// ─── Post-BA velocity refresh ─────────────────────────────────────────────────
//
// After local_ba->optimize() updates keyframe poses and map point positions,
// the velocity_ stored in the tracker is stale (it was computed from pre-BA
// PnP estimates).  Re-derive it from the last two BA-refined keyframes so
// that the next frame's prediction is consistent with the refined map.

void Tracker::notify_ba_update()
{
    // Do NOT re-derive velocity_ from inter-KF poses after BA.
    //
    // BA adjusts KF poses using hundreds of new stereo observations added by
    // insert_keyframe() (triangulate_stereo + multi-KF triangulation).  The
    // resulting KF-to-KF delta encodes the BA correction, NOT physical camera
    // motion.  Applying it as velocity_ produces a wildly wrong Phase 1
    // prediction → Proj-match collapses from ~480 to ~50 → LOST cascade.
    //
    // Setting velocity_valid_ = false forces the next frame to predict from
    // last_frame_->T_cw (the BA-refined most-recent-KF pose) directly.
    // Since the next frame is physically adjacent to that KF, Phase 1 from
    // that pose gives ~400+ matches.  After one successful track(), velocity_
    // re-derives naturally as (frame_N+1 * KF^{-1}) — correct inter-frame.
    velocity_valid_ = false;
}

}  // namespace slam
