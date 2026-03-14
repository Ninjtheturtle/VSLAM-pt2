#include "slam/tracker.hpp"
#include "slam/map_point.hpp"
#include "cuda/hamming_matcher.cuh"
#include "cuda/l2_matcher.cuh"

// Deep frontend components (only included in TUs that need TRT/torch headers)
#include "deep/xfeat_extractor.hpp"
#include "deep/lighterglue_async.hpp"
#include "deep/ttt_autoencoder.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/SVD>
#include <iostream>
#include <atomic>
#include <unordered_set>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace slam {

// global frame and point ID counters
static std::atomic<long> g_frame_id{0};
static std::atomic<long> g_point_id{0};

// factory

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

Tracker::Ptr Tracker::create_hybrid(
    const Camera& cam, Map::Ptr map,
    std::shared_ptr<deep::XFeatExtractor>   xfeat,
    std::shared_ptr<deep::LighterGlueAsync> lighter_glue,
    std::shared_ptr<deep::TTTLoopDetector>  ttt,
    const Config& cfg)
{
    auto t = create(cam, map, cfg);  // sets up ORB as fallback
    t->xfeat_        = std::move(xfeat);
    t->lighter_glue_ = std::move(lighter_glue);
    t->ttt_          = std::move(ttt);
    t->hybrid_mode_  = true;
    return t;
}

// ---------------------------------------------------------------------------
// Hybrid feature extraction
// ---------------------------------------------------------------------------
void Tracker::extract_features_hybrid(Frame::Ptr frame)
{
    deep::XFeatResult res;
    try {
        res = xfeat_->extract(frame->image_gray);
    } catch (const c10::Error& e) {
        fprintf(stderr, "[Hybrid] c10::Error in extract: %s\n", e.what());
        return;
    } catch (const std::exception& e) {
        fprintf(stderr, "[Hybrid] std::exception in extract: %s\n", e.what());
        return;
    } catch (...) {
        fprintf(stderr, "[Hybrid] Unknown exception in extract\n");
        return;
    }
    frame->keypoints.resize(res.N);
    frame->map_points.resize(res.N, nullptr);
    frame->match_confidence.assign(res.N, 1.0f);  // initial default

    for (int i = 0; i < res.N; ++i) {
        frame->keypoints[i].pt.x    = res.kp_x[i];
        frame->keypoints[i].pt.y    = res.kp_y[i];
        frame->keypoints[i].response = res.scores[i];
    }

    // Promote FP16 descriptors → CV_32F for Ceres and existing matching code
    frame->xfeat_descriptors = cv::Mat(res.N, deep::kXFeatDescDim, CV_32F);
    if (res.N > 0 && res.descriptors_pinned) {
        for (int i = 0; i < res.N; ++i)
            for (int j = 0; j < deep::kXFeatDescDim; ++j)
                frame->xfeat_descriptors.at<float>(i, j) =
                    __half2float(res.descriptors_pinned[i * deep::kXFeatDescDim + j]);
    }

    // Cache feat_map for KF semi-dense disparity (device→host copy)
    if (res.feat_map_device && res.feat_map_h > 0 && res.feat_map_w > 0) {
        int C   = deep::kXFeatFeatMapC;
        int fh  = res.feat_map_h;
        int fw  = res.feat_map_w;
        frame->feat_map_left = cv::Mat(C * fh, fw, CV_32F);
        cudaMemcpy(frame->feat_map_left.ptr<float>(), res.feat_map_device,
                   C * fh * fw * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Right image: extract XFeat on right channel and run GPU stereo match
    if (!frame->image_right.empty()) {
        auto res_r = xfeat_->extract(frame->image_right);
        frame->keypoints_right.resize(res_r.N);
        for (int i = 0; i < res_r.N; ++i) {
            frame->keypoints_right[i].pt.x    = res_r.kp_x[i];
            frame->keypoints_right[i].pt.y    = res_r.kp_y[i];
            frame->keypoints_right[i].response = res_r.scores[i];
        }

        // Cache right feat_map for semi-dense
        if (res_r.feat_map_device && res_r.feat_map_h > 0) {
            int C = deep::kXFeatFeatMapC, fh = res_r.feat_map_h, fw = res_r.feat_map_w;
            frame->feat_map_right = cv::Mat(C * fh, fw, CV_32F);
            cudaMemcpy(frame->feat_map_right.ptr<float>(), res_r.feat_map_device,
                       C * fh * fw * sizeof(float), cudaMemcpyDeviceToHost);
        }

        // Build stereo FP16 descriptor matrices for GPU epipolar match
        // Re-use existing cuda_match_stereo_epipolar with ORB-32-byte fields,
        // OR use the new L2 variant via match_l2_fp16 along epipolar.
        // For simplicity: convert XFeat descriptors to ORB-compatible format is
        // not feasible; instead use match_l2_fp16 with epipolar mask applied after.
        frame->uR.assign(res.N, -1.0f);
        if (res.N > 0 && res_r.N > 0 && res.descriptors_pinned && res_r.descriptors_pinned) {
            ensure_l2_buffers(res.N, res_r.N);
            cudaMemcpy(d_query_descs_, res.descriptors_pinned,
                       res.N * deep::kXFeatDescDim * sizeof(__half), cudaMemcpyHostToDevice);
            cudaMemcpy(d_train_descs_, res_r.descriptors_pinned,
                       res_r.N * deep::kXFeatDescDim * sizeof(__half), cudaMemcpyHostToDevice);

            cuda_match_l2_fp16(d_query_descs_, d_train_descs_,
                                res.N, res_r.N, deep::kXFeatDescDim,
                                cfg_.l2_ratio,
                                d_best_idx_, d_best_dist_, d_pseudo_conf_,
                                /*stream=*/0);
            cudaDeviceSynchronize();

            std::vector<int>   best_idx(res.N);
            std::vector<float> best_conf(res.N);
            cudaMemcpy(best_idx.data(), d_best_idx_,    res.N*sizeof(int),   cudaMemcpyDeviceToHost);
            cudaMemcpy(best_conf.data(), d_pseudo_conf_, res.N*sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < res.N; ++i) {
                int j = best_idx[i];
                if (j < 0) continue;
                float dy   = std::abs(frame->keypoints[i].pt.y - frame->keypoints_right[j].pt.y);
                float disp = frame->keypoints[i].pt.x - frame->keypoints_right[j].pt.x;
                if (dy > cfg_.stereo_epi_tol) continue;
                if (disp < cfg_.stereo_d_min || disp > cfg_.stereo_d_max) continue;
                frame->uR[i] = frame->keypoints_right[j].pt.x;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// L2 buffer management
// ---------------------------------------------------------------------------
void Tracker::ensure_l2_buffers(int N_q, int N_t)
{
    int needed = std::max(N_q, N_t);
    if (needed <= d_buf_capacity_) return;

    cudaFree(d_query_descs_); cudaFree(d_train_descs_);
    cudaFree(d_best_idx_);    cudaFree(d_best_dist_); cudaFree(d_pseudo_conf_);

    int cap = needed + 512;
    cudaMalloc(&d_query_descs_, cap * deep::kXFeatDescDim * sizeof(__half));
    cudaMalloc(&d_train_descs_, cap * deep::kXFeatDescDim * sizeof(__half));
    cudaMalloc(&d_best_idx_,    cap * sizeof(int));
    cudaMalloc(&d_best_dist_,   cap * sizeof(float));
    cudaMalloc(&d_pseudo_conf_, cap * sizeof(float));
    d_buf_capacity_ = cap;
}

// ---------------------------------------------------------------------------
// FP16 L2 matching (returns DMatch list + confidence per match)
// ---------------------------------------------------------------------------
std::vector<cv::DMatch> Tracker::match_l2_fp16(
    const cv::Mat& query_descs_fp32,
    const cv::Mat& train_descs_fp32,
    std::vector<float>& out_confidence)
{
    int N_q = query_descs_fp32.rows;
    int N_t = train_descs_fp32.rows;
    if (N_q == 0 || N_t == 0) return {};

    ensure_l2_buffers(N_q, N_t);

    // Convert FP32 → FP16 on host (small overhead; avoids extra device kernel)
    std::vector<__half> q_h16(N_q * deep::kXFeatDescDim);
    std::vector<__half> t_h16(N_t * deep::kXFeatDescDim);
    for (int i = 0; i < N_q * deep::kXFeatDescDim; ++i)
        q_h16[i] = __float2half(query_descs_fp32.ptr<float>()[i]);
    for (int i = 0; i < N_t * deep::kXFeatDescDim; ++i)
        t_h16[i] = __float2half(train_descs_fp32.ptr<float>()[i]);

    cudaMemcpy(d_query_descs_, q_h16.data(), N_q*deep::kXFeatDescDim*sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_descs_, t_h16.data(), N_t*deep::kXFeatDescDim*sizeof(__half), cudaMemcpyHostToDevice);

    cuda_match_l2_fp16(d_query_descs_, d_train_descs_,
                       N_q, N_t, deep::kXFeatDescDim,
                       cfg_.l2_ratio,
                       d_best_idx_, d_best_dist_, d_pseudo_conf_,
                       /*stream=*/0);
    cudaDeviceSynchronize();

    std::vector<int>   best_idx(N_q);
    std::vector<float> best_conf(N_q);
    cudaMemcpy(best_idx.data(),  d_best_idx_,    N_q*sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(best_conf.data(), d_pseudo_conf_, N_q*sizeof(float), cudaMemcpyDeviceToHost);

    out_confidence.resize(N_q, 0.1f);
    std::vector<cv::DMatch> matches;
    for (int i = 0; i < N_q; ++i) {
        if (best_idx[i] < 0) continue;
        out_confidence[i] = best_conf[i];
        matches.push_back(cv::DMatch(i, best_idx[i], 0.0f));
    }
    return matches;
}

// ---------------------------------------------------------------------------
// Submit LighterGlue relocalization job (non-blocking)
// ---------------------------------------------------------------------------
bool Tracker::submit_reloc_job(Frame::Ptr frame)
{
    if (!lighter_glue_ || !lighter_glue_->is_idle()) return false;
    if (frame->xfeat_descriptors.empty()) return false;

    // Ask TTT for the best loop-closure candidate KF
    std::vector<std::vector<float>> query_descs_fp32;
    for (int i = 0; i < frame->xfeat_descriptors.rows; ++i) {
        const float* row = frame->xfeat_descriptors.ptr<float>(i);
        query_descs_fp32.emplace_back(row, row + deep::kXFeatDescDim);
    }
    auto candidates = ttt_ ? ttt_->query_loop_candidates(query_descs_fp32, 1)
                           : std::vector<long>{};

    long cand_kf_id = -1;
    Frame::Ptr cand_kf;
    if (!candidates.empty()) {
        cand_kf_id = candidates[0];
        cand_kf    = map_->get_keyframe(cand_kf_id);
    }
    if (!cand_kf || cand_kf->xfeat_descriptors.empty()) {
        // No TTT candidate — fall back to last good KF
        auto all_kfs = map_->local_window(5);
        if (all_kfs.empty()) return false;
        cand_kf    = all_kfs.back();
        cand_kf_id = cand_kf->id;
    }

    deep::RelocJob job;
    static long job_counter = 0;
    job.job_id          = ++job_counter;
    job.query_frame_id  = frame->id;
    job.candidate_kf_id = cand_kf_id;

    int N_q = frame->xfeat_descriptors.rows;
    job.query_descs.resize(N_q * deep::kXFeatDescDim);
    for (int i = 0; i < N_q * deep::kXFeatDescDim; ++i)
        job.query_descs[i] = __float2half(frame->xfeat_descriptors.ptr<float>()[i]);
    for (int i = 0; i < N_q; ++i) {
        job.query_kp_x.push_back(frame->keypoints[i].pt.x);
        job.query_kp_y.push_back(frame->keypoints[i].pt.y);
    }

    int N_t = cand_kf->xfeat_descriptors.rows;
    job.candidate_descs.resize(N_t * deep::kXFeatDescDim);
    for (int i = 0; i < N_t * deep::kXFeatDescDim; ++i)
        job.candidate_descs[i] = __float2half(cand_kf->xfeat_descriptors.ptr<float>()[i]);
    for (int i = 0; i < N_t; ++i) {
        job.candidate_kp_x.push_back(cand_kf->keypoints[i].pt.x);
        job.candidate_kp_y.push_back(cand_kf->keypoints[i].pt.y);
    }

    return lighter_glue_->submit_job(std::move(job));
}

// ---------------------------------------------------------------------------
// Apply a successful RelocResult (run PnP against matched KF map points)
// ---------------------------------------------------------------------------
bool Tracker::apply_reloc_result(Frame::Ptr frame, const deep::RelocResult& result)
{
    Frame::Ptr cand_kf = map_->get_keyframe(result.candidate_kf_id);
    if (!cand_kf) return false;

    // Build 3D-2D correspondences
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    std::vector<float>       confs;
    std::vector<int>         query_idxs;

    for (const auto& m : result.matches) {
        if (m.query_idx >= (int)frame->keypoints.size()) continue;
        if (m.train_idx >= (int)cand_kf->keypoints.size()) continue;
        auto& mp = cand_kf->map_points[m.train_idx];
        if (!mp || mp->is_bad) continue;

        auto& pos = mp->position;
        pts3d.push_back({(float)pos.x(), (float)pos.y(), (float)pos.z()});
        pts2d.push_back(frame->keypoints[m.query_idx].pt);
        confs.push_back(m.confidence);
        query_idxs.push_back(m.query_idx);
    }

    if ((int)pts3d.size() < cfg_.pnp_min_inliers * 2) return false;

    cv::Mat rvec, tvec, inlier_mask;
    bool ok = cv::solvePnPRansac(
        pts3d, pts2d, cam_.K_cv(), cam_.dist_cv(),
        rvec, tvec, /*useExtrinsicGuess=*/false,
        cfg_.pnp_iterations, cfg_.pnp_reprojection,
        0.99, inlier_mask, cv::SOLVEPNP_SQPNP);

    if (!ok) return false;

    int n_inliers = cv::countNonZero(inlier_mask);
    if (n_inliers < cfg_.pnp_min_inliers) return false;

    // Commit pose
    cv::Mat R_mat; cv::Rodrigues(rvec, R_mat);
    Eigen::Matrix3d R; Eigen::Vector3d t;
    for (int i = 0; i < 3; ++i) {
        t(i) = tvec.at<double>(i);
        for (int j = 0; j < 3; ++j)
            R(i, j) = R_mat.at<double>(i, j);
    }
    frame->T_cw.linear()      = R;
    frame->T_cw.translation() = t;

    // Assign map points and confidence weights for inlier correspondences
    for (int k = 0; k < (int)inlier_mask.rows; ++k) {
        if (!inlier_mask.at<uint8_t>(k)) continue;
        int qi = query_idxs[k];
        auto& mp = cand_kf->map_points[
            result.matches[k].train_idx];  // re-index from original match list
        if (mp && !mp->is_bad) {
            frame->map_points[qi]       = mp;
            frame->match_confidence[qi] = result.matches[k].confidence;
        }
    }

    std::cout << "[Tracker] LG Relocalized: " << n_inliers << "/" << pts3d.size() << " inliers\n";
    return true;
}

// main entry point

bool Tracker::track(Frame::Ptr frame)
{
    // --- Feature extraction ---
    if (hybrid_mode_) {
        extract_features_hybrid(frame);
        // In hybrid mode, ORB descriptors are not extracted; xfeat_descriptors is the source.
        // We still need frame->descriptors to be non-empty for any code path that still
        // uses the ORB-based match_descriptors() fallback (e.g. monocular init).
        // Provide a dummy 0-row mat; caller code checks keypoints.size() instead.
        frame->descriptors = cv::Mat(0, 32, CV_8U);
    } else {
        // Legacy ORB path
        orb_->detectAndCompute(frame->image_gray, cv::noArray(),
                               frame->keypoints, frame->descriptors);
        frame->map_points.resize(frame->keypoints.size(), nullptr);
        if (!frame->image_right.empty()) {
            orb_->detectAndCompute(frame->image_right, cv::noArray(),
                                   frame->keypoints_right, frame->descriptors_right);
            frame->uR.assign(frame->keypoints.size(), -1.0f);
            match_stereo(frame);
        }
    }

    // -----------------------------------------------------------------------
    // RELOCALIZING — poll LighterGlue for async result each frame
    // -----------------------------------------------------------------------
    if (state_ == TrackingState::RELOCALIZING) {
        ++reloc_wait_frames_;

        // Dead-reckon while waiting
        frame->T_cw = velocity_valid_
            ? velocity_ * last_frame_->T_cw
            : last_frame_->T_cw;

        // Poll (non-blocking)
        if (lighter_glue_) {
            auto result = lighter_glue_->try_get_result();
            if (result.has_value()) {
                if (result->success && apply_reloc_result(frame, *result)) {
                    std::cout << "[Tracker] LG Relocalized — resuming\n";
                    velocity_valid_   = false;
                    reloc_wait_frames_ = 0;
                    state_            = TrackingState::OK;
                    last_frame_       = frame;
                    return true;
                } else {
                    // LG returned but failed — fall through to NOT_INITIALIZED
                    std::cerr << "[Tracker] LG reloc failed, resetting\n";
                    goto reset_and_reinit;
                }
            }
        }
        if (reloc_wait_frames_ >= cfg_.reloc_timeout) {
            std::cerr << "[Tracker] LG reloc timeout, resetting\n";
            goto reset_and_reinit;
        }
        last_frame_ = frame;
        return false;
    }

    // -----------------------------------------------------------------------
    // LOST — submit LG job (or fall back to legacy relocalize + reset)
    // -----------------------------------------------------------------------
    if (state_ == TrackingState::LOST) {
        if (hybrid_mode_ && lighter_glue_) {
            bool submitted = submit_reloc_job(frame);
            if (submitted) {
                state_ = TrackingState::RELOCALIZING;
                reloc_wait_frames_ = 0;
                // Coast this frame
                frame->T_cw = velocity_valid_
                    ? velocity_ * last_frame_->T_cw : last_frame_->T_cw;
                last_frame_ = frame;
                return false;
            }
            // LG busy — coast one more frame
            frame->T_cw = velocity_valid_
                ? velocity_ * last_frame_->T_cw : last_frame_->T_cw;
            last_frame_ = frame;
            return false;
        }

        // Legacy path: synchronous relocalize
        if (try_relocalize(frame)) {
            std::cout << "[Tracker] Relocalized successfully\n";
            velocity_valid_ = false;
            state_          = TrackingState::OK;
            last_frame_     = frame;
            return true;
        }

        reset_and_reinit:
        std::cerr << "[Tracker] LOST — resetting map + re-initializing\n";
        frame->T_cw = velocity_valid_ ? (velocity_ * last_frame_->T_cw) : last_frame_->T_cw;
        map_->reset();
        last_keyframe_       = nullptr;
        last_kf_pnp_tracked_ = 0;
        velocity_valid_      = false;
        reloc_wait_frames_   = 0;
        last_frame_          = frame;
        state_               = TrackingState::NOT_INITIALIZED;
        return false;
    }

    if (state_ == TrackingState::NOT_INITIALIZED)
        return initialize(frame);

    // -----------------------------------------------------------------------
    // OK / COASTING — normal tracking
    // -----------------------------------------------------------------------
    bool ok = track_with_motion_model(frame);
    if (ok) ok = track_local_map(frame);
    if (ok) {
        lost_streak_ = 0;
        if (state_ == TrackingState::COASTING)
            state_ = TrackingState::OK;
    } else {
        ++lost_streak_;
        if (lost_streak_ < cfg_.coast_limit) {
            state_ = TrackingState::COASTING;
            // coast: dead-reckon
            frame->T_cw = velocity_valid_
                ? velocity_ * last_frame_->T_cw
                : last_frame_->T_cw;
            // Pre-submit LG job in hybrid mode so reloc has a head start
            if (hybrid_mode_ && lighter_glue_ && lost_streak_ == 1)
                submit_reloc_job(frame);
            last_frame_ = frame;
            return false;
        }
        lost_streak_ = 0;
        state_ = TrackingState::LOST;
    }
    last_frame_ = frame;
    return ok;
}

// initialization

bool Tracker::initialize(Frame::Ptr frame)
{
    if (cam_.is_stereo() && !frame->uR.empty()) {
        // stereo path: propagate last pose so reinit doesn't jump to origin
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

    // monocular fallback
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

    // require sufficient 2D feature displacement to avoid degenerate init
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

    // scale to ~20 m median depth (monocular scale ambiguity workaround)
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

// constant-velocity tracking — matches against all local KF map points

bool Tracker::track_with_motion_model(Frame::Ptr frame)
{
    // predict pose
    if (velocity_valid_) {
        frame->T_cw = velocity_ * last_frame_->T_cw;
    } else {
        frame->T_cw = last_frame_->T_cw;
    }

    // build descriptor pool from the last 30 KFs
    // In hybrid mode use XFeat (CV_32F); in ORB mode use ORB (CV_8U)
    cv::Mat                    pool_desc;
    std::vector<MapPoint::Ptr> pool_mps;
    {
        std::unordered_set<long> seen_ids;
        for (auto& kf : map_->local_window(30)) {
            const cv::Mat& kf_descs = hybrid_mode_ ? kf->xfeat_descriptors : kf->descriptors;
            if (kf_descs.empty()) continue;
            for (int i = 0; i < (int)kf->map_points.size(); ++i) {
                auto& mp = kf->map_points[i];
                if (!mp || mp->is_bad) continue;
                if (mp->observed_times < 2) continue;
                if (!seen_ids.insert(mp->id).second) continue;
                if (i >= kf_descs.rows) continue;
                pool_desc.push_back(kf_descs.row(i));
                pool_mps.push_back(mp);
            }
        }
    }
    if ((int)pool_desc.rows < cfg_.pnp_min_inliers) {
        std::cerr << "[Tracker] Track: pool too small (" << pool_desc.rows << " pts)\n";
        return false;
    }

    // phase 1: spatial search around projected pool points. phase 2: GPU Hamming fallback.
    std::vector<cv::Point3f>   pts3d;
    std::vector<cv::Point2f>   pts2d;
    std::vector<int>           match_idxs;
    std::vector<MapPoint::Ptr> match_mps;
    std::unordered_set<int>    used_kp;

    // phase 1: projection-based spatial matching
    {
        const int   cell     = 16;
        const int   frame_w  = frame->image_gray.cols;
        const int   frame_h  = frame->image_gray.rows;
        const int   n_cols_g = (frame_w + cell - 1) / cell;
        const int   n_rows_g = (frame_h + cell - 1) / cell;
        // widen search at turns; base_r=70 when velocity is invalid (one frame after BA)
        const double pred_ang = velocity_valid_
            ? Eigen::AngleAxisd(velocity_.rotation()).angle() : 0.0;
        const float base_r = velocity_valid_ ? 40.0f : 70.0f;
        const float search_r = (pred_ang > 0.03)
            ? std::min(120.0f, base_r + float(pred_ang / 0.03) * 20.0f)
            : base_r;
        const int   max_ham  = cfg_.hamming_threshold;

        // build spatial grid: cell → list of keypoint indices
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

            int best_j = -1;
            float best_d = hybrid_mode_ ? 0.8f : float(max_ham + 1);
            const cv::Mat& frame_descs = hybrid_mode_ ? frame->xfeat_descriptors
                                                       : frame->descriptors;
            for (int gy = cy0; gy <= cy1; ++gy)
                for (int gx = cx0; gx <= cx1; ++gx)
                    for (int kp_j : kp_grid[gy * n_cols_g + gx]) {
                        if (used_kp.count(kp_j)) continue;
                        if (kp_j >= frame_descs.rows) continue;
                        float d;
                        if (hybrid_mode_) {
                            // L2 distance for XFeat (L2-normalised, range [0,2])
                            d = (float)cv::norm(pool_desc.row(mi),
                                                frame_descs.row(kp_j),
                                                cv::NORM_L2);
                        } else {
                            d = (float)cv::norm(pool_desc.row(mi),
                                                frame_descs.row(kp_j),
                                                cv::NORM_HAMMING);
                        }
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
        fprintf(stderr, "[Tracker] Proj-match: %d/%d pts\n",
                (int)pts3d.size(), (int)pool_mps.size());
    }

    // phase 2: fallback (L2 for hybrid, GPU Hamming for ORB)
    if ((int)pts3d.size() < cfg_.pnp_min_inliers) {
        pts3d.clear(); pts2d.clear(); match_idxs.clear(); match_mps.clear(); used_kp.clear();
        std::vector<cv::DMatch> raw_matches;
        if (hybrid_mode_) {
            std::vector<float> dummy_conf;
            raw_matches = match_l2_fp16(pool_desc, frame->xfeat_descriptors, dummy_conf);
        } else {
            raw_matches = match_descriptors(pool_desc, frame->descriptors, /*use_ratio=*/true);
        }
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
    // skip initial guess at sharp turns — overshoots bias RANSAC (threshold: 0.3 rad ≈ 17°)
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

    // inlier mask can be CV_8U (mask) or CV_32S (index vector) depending on OpenCV version
    std::vector<bool> is_inlier(pts3d.size(), false);
    int n_inliers = 0;
    if (inlier_mask.type() == CV_8U) {
        for (int k = 0; k < inlier_mask.rows && k < (int)pts3d.size(); ++k)
            if (inlier_mask.at<uint8_t>(k)) { is_inlier[k] = true; ++n_inliers; }
    } else {
        // index mode (CV_32S): each element is an inlier index into pts3d
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

    // build candidate pose from rvec/tvec
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    Eigen::Isometry3d T_cw_candidate = Eigen::Isometry3d::Identity();
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            T_cw_candidate.linear()(r, c) = R_cv.at<double>(r, c);
    T_cw_candidate.translation() << tvec.at<double>(0),
                                     tvec.at<double>(1),
                                     tvec.at<double>(2);

    // delta sanity checks — a car can't rotate >~29° or move >5 m between frames at 10 Hz
    {
        Eigen::Isometry3d delta = T_cw_candidate * last_frame_->T_cw.inverse();
        double delta_angle = Eigen::AngleAxisd(delta.rotation()).angle();
        if (delta_angle > 0.5) {    // 0.5 rad ≈ 29°
            std::cerr << "[Tracker] PnP rejected: delta rot "
                      << (delta_angle * (180.0 / 3.14159265358979323846)) << " deg\n";
            return false;
        }
        double delta_trans = delta.translation().norm();
        if (delta_trans > 5.0) {   // >5 m/frame ≈ 180 km/h at 10 Hz
            std::cerr << "[Tracker] PnP rejected: delta trans "
                      << delta_trans << " m\n";
            return false;
        }
    }

    // commit pose
    frame->T_cw = T_cw_candidate;

    // assign inlier map points
    for (int i = 0; i < (int)pts3d.size(); ++i)
        if (is_inlier[i])
            frame->map_points[match_idxs[i]] = match_mps[i];

    // project-and-search: pull in more map points after PnP without a second RANSAC pass
    {
        const int   cell      = 16;
        const int   frame_w   = frame->image_gray.cols;
        const int   frame_h   = frame->image_gray.rows;
        const int   n_cols_g  = (frame_w + cell - 1) / cell;
        const int   n_rows_g  = (frame_h + cell - 1) / cell;
        const float search_r  = 15.0f;
        const int   max_ham   = 50;
        const float max_repr2 = 25.0f;   // 5 px validation threshold²

        // spatial grid: only keypoints that still need a map point
        std::vector<std::vector<int>> kp_grid(n_cols_g * n_rows_g);
        for (int j = 0; j < (int)frame->keypoints.size(); ++j) {
            if (frame->map_points[j]) continue;
            if (used_kp.count(j)) continue;
            int cx = std::min(n_cols_g - 1, (int)(frame->keypoints[j].pt.x / cell));
            int cy = std::min(n_rows_g - 1, (int)(frame->keypoints[j].pt.y / cell));
            if (cx >= 0 && cy >= 0) kp_grid[cy * n_cols_g + cx].push_back(j);
        }

        // track which map-point IDs were already tried in the pool phase
        std::unordered_set<long> proj_seen;
        for (auto& mp : pool_mps) proj_seen.insert(mp->id);

        for (auto& kf : map_->local_window(30)) {
            if (kf->descriptors.empty()) continue;
            for (int i = 0; i < (int)kf->map_points.size(); ++i) {
                auto& mp = kf->map_points[i];
                if (!mp || mp->is_bad) continue;
                if (!proj_seen.insert(mp->id).second) continue;  // already tried
                if (i >= kf->descriptors.rows) continue;

                // project into frame using committed T_cw
                Eigen::Vector3d Xc = frame->T_cw * mp->position;
                if (Xc.z() <= 0.0) continue;
                float u = (float)(cam_.fx * Xc.x() / Xc.z() + cam_.cx);
                float v = (float)(cam_.fy * Xc.y() / Xc.z() + cam_.cy);
                if (u < 0 || u >= frame_w || v < 0 || v >= frame_h) continue;

                // search grid cells in (u±search_r, v±search_r)
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

                // validate with reprojection error at committed pose
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

// local map tracking

bool Tracker::track_local_map(Frame::Ptr frame)
{
    if (need_new_keyframe(frame)) {
        insert_keyframe(frame);
    }
    return frame->num_tracked() >= cfg_.pnp_min_inliers;
}

// keyframe decision

bool Tracker::need_new_keyframe(Frame::Ptr frame) const
{
    if (!last_keyframe_) return true;

    int tracked = frame->num_tracked();  // PnP inliers only (not yet triangulated)

    // compare against last_kf_pnp_tracked_ (pre-triangulation count) to avoid inflated ratio
    if (tracked < cfg_.min_tracked_points) return true;
    if (last_kf_pnp_tracked_ > 0 && (float)tracked / last_kf_pnp_tracked_ < 0.8f) return true;

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
    // save PnP inlier count BEFORE triangulation inflates frame->map_points
    last_kf_pnp_tracked_ = frame->num_tracked();

    frame->id = g_frame_id++;
    frame->is_keyframe = true;

    // triangulate against last 3 KFs (oldest first for largest baseline)
    for (auto& tri_kf : map_->local_window(3)) {
        if (tri_kf->id == frame->id) continue;
        // Use FP16 L2 matching in hybrid mode, ORB Hamming otherwise
        std::vector<cv::DMatch> kf_matches;
        if (hybrid_mode_ && !frame->xfeat_descriptors.empty()
                         && !tri_kf->xfeat_descriptors.empty()) {
            std::vector<float> dummy_conf;
            kf_matches = match_l2_fp16(tri_kf->xfeat_descriptors,
                                       frame->xfeat_descriptors, dummy_conf);
        } else {
            kf_matches = match_descriptors(tri_kf->descriptors,
                                           frame->descriptors, /*ratio=*/true);
        }
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

    // stereo enrichment: fill unmapped keypoints with metric-depth points
    if (cam_.is_stereo() && !frame->uR.empty()) {
        int n_stereo = triangulate_stereo(frame);
        if (n_stereo > 0)
            std::cout << "[Tracker] KF " << frame->id
                      << ": stereo added " << n_stereo << " metric pts\n";
    }

    map_->insert_keyframe(frame);
    last_keyframe_ = frame;

    // Push XFeat descriptors to TTT loop detector (non-blocking)
    if (hybrid_mode_ && ttt_ && !frame->xfeat_descriptors.empty()) {
        deep::TTTUpdateJob ttt_job;
        ttt_job.kf_id       = frame->id;
        ttt_job.kf_position = frame->camera_center();
        for (int i = 0; i < frame->xfeat_descriptors.rows; ++i) {
            const float* row = frame->xfeat_descriptors.ptr<float>(i);
            ttt_job.descs.emplace_back(row, row + deep::kXFeatDescDim);
        }
        ttt_->push_keyframe(std::move(ttt_job));
    }
}

// GPU descriptor matching

std::vector<cv::DMatch> Tracker::match_descriptors(
    const cv::Mat& query_desc,
    const cv::Mat& train_desc,
    bool use_ratio)
{
    int N_q = query_desc.rows;
    int N_t = train_desc.rows;

    if (N_q == 0 || N_t == 0) return {};

    // ensure descriptors are continuous and CV_8U
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

// triangulation

int Tracker::triangulate_and_add(Frame::Ptr ref, Frame::Ptr cur,
                                  const std::vector<cv::DMatch>& matches)
{
    // build projection matrices (3×4)
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

    // collect matched point pairs
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

        // depth check in both cameras
        Eigen::Vector3d Xc0 = ref->T_cw * Xw;
        Eigen::Vector3d Xc1 = cur->T_cw * Xw;
        if (Xc0.z() < 0.05 || Xc1.z() < 0.05) continue;
        if (Xc0.z() > 200.0 || Xc1.z() > 200.0) continue;

        // skip near-degenerate triangulations (cos_pa > 0.9998 ≈ <1.1° parallax)
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

// relocalization — match against all KFs with a stricter inlier threshold

bool Tracker::try_relocalize(Frame::Ptr frame)
{
    // build pool from ALL keyframes
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

    // GPU descriptor matching
    auto raw_matches = match_descriptors(pool_desc, frame->descriptors, true);

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

    // PnP RANSAC — no initial guess (no valid velocity)
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

    // LM refinement on inliers
    std::vector<cv::Point3f> in3d;
    std::vector<cv::Point2f> in2d;
    for (int i = 0; i < (int)pts3d.size(); ++i)
        if (is_inlier[i]) { in3d.push_back(pts3d[i]); in2d.push_back(pts2d[i]); }
    cv::solvePnP(in3d, in2d, cam_.K_cv(), cam_.dist_cv(),
                 rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

    // update frame pose
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            frame->T_cw.linear()(r, c) = R_cv.at<double>(r, c);
    frame->T_cw.translation() << tvec.at<double>(0),
                                  tvec.at<double>(1),
                                  tvec.at<double>(2);

    // assign inlier map points
    for (int i = 0; i < (int)pts3d.size(); ++i)
        if (is_inlier[i])
            frame->map_points[match_idxs[i]] = match_mps[i];

    std::cout << "[Reloc] SUCCESS — " << inlier_mask.rows << " inliers\n";
    return true;
}

// stereo epipolar matching

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

// stereo triangulation

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

// invalidate velocity after BA — inter-KF delta is a correction, not physical motion

void Tracker::notify_ba_update()
{
    velocity_valid_ = false;
}

}  // namespace slam
