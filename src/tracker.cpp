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
    // Extract ORB features
    orb_->detectAndCompute(frame->image_gray, cv::noArray(),
                           frame->keypoints, frame->descriptors);
    frame->map_points.resize(frame->keypoints.size(), nullptr);

    // ── LOST: try to relocalize against the full map before reinitializing ─────
    if (state_ == TrackingState::LOST) {
        if (try_relocalize(frame)) {
            std::cout << "[Tracker] Relocalized successfully — resuming on existing map\n";
            velocity_valid_ = false;   // old velocity invalid after gap
            state_          = TrackingState::OK;
            last_frame_     = frame;
            return true;
        }
        // Relocalization failed — fall back to fresh re-initialization
        std::cerr << "[Tracker] LOST — relocalization failed, re-initializing\n";
        velocity_valid_ = false;
        last_frame_     = frame;
        state_          = TrackingState::NOT_INITIALIZED;
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
    if (!ok) state_ = TrackingState::LOST;
    last_frame_ = frame;
    return ok;
}

// ─── Initialization ──────────────────────────────────────────────────────────

bool Tracker::initialize(Frame::Ptr frame)
{
    if (!last_frame_) {
        // First frame — anchor it as the reference; return false (need a 2nd frame).
        last_frame_ = frame;
        return false;
    }

    // Match between reference frame and current frame
    auto matches = match_descriptors(last_frame_->descriptors,
                                     frame->descriptors, /*ratio=*/true);
    if ((int)matches.size() < 50) {
        // Too few features in this pair — try the current frame as a new reference.
        std::cerr << "[Tracker] Init: too few matches (" << matches.size()
                  << "), q=" << last_frame_->descriptors.rows
                  << " t=" << frame->descriptors.rows << "\n";
        last_frame_ = frame;
        return false;
    }

    // Build point correspondences
    std::vector<cv::Point2f> pts0, pts1;
    for (auto& m : matches) {
        pts0.push_back(last_frame_->keypoints[m.queryIdx].pt);
        pts1.push_back(frame->keypoints[m.trainIdx].pt);
    }

    // Require sufficient 2D feature displacement — prevents degenerate init
    // when consecutive frames have near-zero baseline (car nearly stationary).
    // When disparity is insufficient, keep last_frame_ ANCHORED so that
    // baseline accumulates over subsequent frames instead of resetting every frame.
    {
        double sum_disp = 0.0;
        for (size_t i = 0; i < pts0.size(); ++i) {
            double dx = pts1[i].x - pts0[i].x;
            double dy = pts1[i].y - pts0[i].y;
            sum_disp += std::sqrt(dx*dx + dy*dy);
        }
        double mean_disp = sum_disp / pts0.size();
        if (mean_disp < cfg_.init_min_disparity) {
            // Insufficient baseline — do NOT advance last_frame_; wait for motion.
            std::cerr << "[Tracker] Init: low disparity (" << mean_disp
                      << " px, need " << cfg_.init_min_disparity << "), holding anchor\n";
            return false;
        }
    }

    // Essential matrix
    cv::Mat E, mask;
    E = cv::findEssentialMat(pts0, pts1,
                             cam_.K_cv(), cv::RANSAC,
                             0.999, 1.0, 1000, mask);
    if (E.empty()) {
        std::cerr << "[Tracker] Init: essential matrix empty (matches=" << matches.size() << ")\n";
        last_frame_ = frame; return false;
    }

    cv::Mat R_cv, t_cv;
    int inliers = cv::recoverPose(E, pts0, pts1,
                                  cam_.K_cv(), R_cv, t_cv, mask);
    if (inliers < 20) {
        std::cerr << "[Tracker] Init: recoverPose inliers too low (" << inliers << ")\n";
        last_frame_ = frame; return false;
    }

    // Set reference frame as identity, current frame as recovered pose
    last_frame_->T_cw = Eigen::Isometry3d::Identity();

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            R(r, c) = R_cv.at<double>(r, c);
    t << t_cv.at<double>(0), t_cv.at<double>(1), t_cv.at<double>(2);

    frame->T_cw.linear()      = R;
    frame->T_cw.translation() = t;

    // Triangulate initial map
    int n_pts = triangulate_and_add(last_frame_, frame, matches);
    if (n_pts < 20) {
        std::cerr << "[Tracker] Init: triangulation too sparse (" << n_pts << " pts)\n";
        last_frame_ = frame; return false;
    }

    // ── Scale normalisation ───────────────────────────────────────────────────
    // recoverPose returns a unit-length translation; the actual metric scale is
    // unknown.  Normalise so that the median depth of newly triangulated points
    // (as seen from the current frame) equals cfg_.init_median_depth metres.
    // This gives approximately metric scale for outdoor KITTI scenes.
    if (cfg_.init_median_depth > 0.0f) {
        std::vector<double> depths;
        depths.reserve(n_pts);
        for (auto& mp : map_->all_map_points()) {
            Eigen::Vector3d Xc = frame->T_cw * mp->position;
            if (Xc.z() > 0.0) depths.push_back(Xc.z());
        }
        if (!depths.empty()) {
            std::sort(depths.begin(), depths.end());
            double median = depths[depths.size() / 2];
            double scale  = cfg_.init_median_depth / median;
            // Scale current frame translation (last_frame_ is at origin — no change)
            frame->T_cw.translation() *= scale;
            // Scale all triangulated map points
            for (auto& mp : map_->all_map_points())
                mp->position *= scale;
            std::cout << "[Tracker] Init scale: median depth " << median
                      << " m → " << cfg_.init_median_depth
                      << " m  (scale=" << scale << ")\n";
        }
    }

    // Insert both as keyframes (insert_keyframe assigns IDs via g_frame_id)
    insert_keyframe(last_frame_);
    insert_keyframe(frame);

    // Compute velocity BEFORE advancing last_frame_ (depends on ref frame pose)
    velocity_       = frame->T_cw * last_frame_->T_cw.inverse();
    velocity_valid_ = true;
    state_          = TrackingState::OK;

    // Advance last_frame_ to current (track() no longer does this for NOT_INIT state)
    last_frame_ = frame;

    std::cout << "[Tracker] Initialized with " << n_pts << " map points\n";
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

    // GPU descriptor matching: pool (query) → current frame (train)
    auto raw_matches = match_descriptors(pool_desc, frame->descriptors,
                                         /*use_ratio=*/true);

    std::vector<cv::Point3f>   pts3d;
    std::vector<cv::Point2f>   pts2d;
    std::vector<int>           match_idxs;
    std::vector<MapPoint::Ptr> match_mps;
    std::unordered_set<int>    used_kp;

    for (auto& m : raw_matches) {
        if (!used_kp.insert(m.trainIdx).second) continue; // one match per keypoint
        auto& mp = pool_mps[m.queryIdx];
        auto& p  = mp->position;
        pts3d.push_back({(float)p.x(), (float)p.y(), (float)p.z()});
        pts2d.push_back(frame->keypoints[m.trainIdx].pt);
        match_idxs.push_back(m.trainIdx);
        match_mps.push_back(mp);
    }
    if ((int)pts3d.size() < cfg_.pnp_min_inliers) {
        std::cerr << "[Tracker] Track: too few 3D-2D matches (" << pts3d.size() << ")\n";
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
    // actually valid.  After relocalization or reinit, velocity_valid_=false
    // and the guess would be zero-motion, which biases RANSAC away from the
    // true pose.  With useExtrinsicGuess=false RANSAC searches freely.
    bool ok = cv::solvePnPRansac(
        pts3d, pts2d, cam_.K_cv(), cam_.dist_cv(),
        rvec, tvec, /*useExtrinsicGuess=*/velocity_valid_,
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
        if (delta_angle > 0.5) {   // 0.5 rad ≈ 29°
            std::cerr << "[Tracker] PnP rejected: delta rot "
                      << (delta_angle * (180.0 / 3.14159265358979323846)) << " deg\n";
            return false;
        }
        double delta_trans = delta.translation().norm();
        if (delta_trans > 5.0) {   // >5 m/frame ≈ >180 km/h at 10 Hz
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

    return false;
}

void Tracker::insert_keyframe(Frame::Ptr frame)
{
    // Save PnP-inlier count BEFORE triangulation inflates frame->map_points.
    last_kf_pnp_tracked_ = frame->num_tracked();

    frame->id = g_frame_id++;
    frame->is_keyframe = true;

    // Triangulate new map points between this keyframe and the previous one.
    // Only attempt unmatched keypoints in the new frame (those without a map point).
    if (last_keyframe_) {
        auto kf_matches = match_descriptors(last_keyframe_->descriptors,
                                            frame->descriptors, /*ratio=*/true);
        std::vector<cv::DMatch> new_matches;
        for (auto& m : kf_matches) {
            if (m.trainIdx < (int)frame->map_points.size() &&
                !frame->map_points[m.trainIdx]) {
                new_matches.push_back(m);
            }
        }
        if (!new_matches.empty()) {
            int n_new = triangulate_and_add(last_keyframe_, frame, new_matches);
            if (n_new > 0)
                std::cout << "[Tracker] KF " << frame->id
                          << ": triangulated " << n_new << " new map pts\n";
        }
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

    const int reloc_min = cfg_.pnp_min_inliers * 3;  // 45
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

// ─── Post-BA velocity refresh ─────────────────────────────────────────────────
//
// After local_ba->optimize() updates keyframe poses and map point positions,
// the velocity_ stored in the tracker is stale (it was computed from pre-BA
// PnP estimates).  Re-derive it from the last two BA-refined keyframes so
// that the next frame's prediction is consistent with the refined map.

void Tracker::notify_ba_update()
{
    auto kfs = map_->local_window(2);
    if (kfs.size() < 2) return;

    // kfs.back() = most recent KF (just optimized)
    // kfs[size-2] = the one before it
    velocity_       = kfs.back()->T_cw * kfs[kfs.size() - 2]->T_cw.inverse();
    velocity_valid_ = true;
}

}  // namespace slam
