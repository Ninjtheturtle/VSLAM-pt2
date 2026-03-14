#pragma once

#include "slam/camera.hpp"
#include "slam/frame.hpp"
#include "slam/map.hpp"
#include <opencv2/features2d.hpp>
#include <memory>

// Forward-declare deep components to avoid pulling TRT/torch into every TU
namespace deep {
class XFeatExtractor;
class LighterGlueAsync;
class TTTLoopDetector;
struct XFeatResult;
struct RelocResult;
}

namespace slam {

enum class TrackingState {
    NOT_INITIALIZED,  // waiting for sufficient stereo/mono initialization
    OK,               // normal tracking
    COASTING,         // tracking failed; dead-reckoning with velocity model
    LOST,             // coasting limit exceeded; awaiting LighterGlue relocalization
    RELOCALIZING      // LighterGlue job submitted; waiting for async result
};

/// front-end tracker — hybrid deep-geometric pipeline:
///   1. extract XFeat keypoints + FP16 descriptors (TensorRT)
///   2. if not initialized: stereo single-frame init (or monocular two-frame)
///   3. if OK: predict pose with constant velocity, refine with FP16-L2-matched
///      map-point reprojections and PnP RANSAC
///   4. confidence weights from L2 ratio → frame->match_confidence[]
///   5. coasting up to 8 frames on velocity model before LOST
///   6. LOST: submit async LighterGlue job for relocalization (non-blocking)
///   7. RELOCALIZING: poll LighterGlue result each frame; apply on success
///   8. keyframe insertion: push descriptors to TTT loop detector (non-blocking)
class Tracker {
public:
    struct Config {
        // XFeat / feature extraction
        int   max_keypoints      = 2000;
        float anms_min_response  = 0.005f;
        float l2_ratio           = 0.9f;   // Lowe ratio for FP16 L2 matching

        // Legacy ORB fields retained for monocular fallback and init
        int   orb_features       = 2000;
        float orb_scale_factor   = 1.2f;
        int   orb_levels         = 8;
        int   orb_edge_threshold = 31;
        int   hamming_threshold  = 60;
        float lowe_ratio         = 0.75f;

        int   min_tracked_points = 80;
        int   pnp_iterations     = 200;
        float pnp_reprojection   = 5.5f;
        int   pnp_min_inliers    = 15;
        float stereo_epi_tol     = 2.0f;
        float stereo_d_min       = 3.0f;
        float stereo_d_max       = 300.0f;

        // Coasting / LOST behavior
        int   coast_limit        = 8;   // frames before declaring LOST
        int   reloc_timeout      = 20;  // frames before giving up on LG reloc
    };

    using Ptr = std::shared_ptr<Tracker>;

    // Basic factory (no deep components; falls back to ORB pipeline)
    static Ptr create(const Camera& cam, Map::Ptr map,
                      const Config& cfg = Config{});

    // Full hybrid factory — takes ownership of deep component pointers
    static Ptr create_hybrid(
        const Camera& cam, Map::Ptr map,
        std::shared_ptr<deep::XFeatExtractor>  xfeat,
        std::shared_ptr<deep::LighterGlueAsync> lighter_glue,
        std::shared_ptr<deep::TTTLoopDetector>  ttt,
        const Config& cfg = Config{});

    /// process a new frame; returns true if tracking succeeded
    bool track(Frame::Ptr frame);

    TrackingState state() const { return state_; }

    /// call after BA to invalidate the stale velocity estimate.
    /// see notify_ba_update() in tracker.cpp for why we don't re-derive velocity_ from BA poses.
    void notify_ba_update();

private:
    bool initialize(Frame::Ptr frame);
    bool track_with_motion_model(Frame::Ptr frame);
    bool track_local_map(Frame::Ptr frame);
    bool need_new_keyframe(Frame::Ptr frame) const;
    void insert_keyframe(Frame::Ptr frame);

    // --- Hybrid-mode feature extraction ---
    // Runs XFeat TRT inference, populates frame->keypoints, xfeat_descriptors,
    // match_confidence (initialized to 1.0), and feat_map_{left,right}.
    void extract_features_hybrid(Frame::Ptr frame);

    // Ensures device L2 matching buffers have capacity for N_q × N_t descriptors.
    void ensure_l2_buffers(int N_q, int N_t);

    // FP16 L2 matching using XFeat descriptors.
    // Outputs matches with confidence weights in frame->match_confidence.
    std::vector<cv::DMatch> match_l2_fp16(
        const cv::Mat& query_descs_fp32,   // [N_q × 64] CV_32F (from frame)
        const cv::Mat& train_descs_fp32,   // [N_t × 64] CV_32F (from pool)
        std::vector<float>& out_confidence // [N_q] pseudo-confidence per query match
    );

    // Submit a relocalization job to LighterGlueAsync.
    // Picks best TTT candidate. Non-blocking; returns false if LG is busy.
    bool submit_reloc_job(Frame::Ptr frame);

    // Apply a successful RelocResult: build 3D-2D correspondences and run PnP.
    bool apply_reloc_result(Frame::Ptr frame, const deep::RelocResult& result);

    /// GPU Hamming matcher; returns cv::DMatch vector filtered by ratio test
    std::vector<cv::DMatch> match_descriptors(
        const cv::Mat& query_desc,
        const cv::Mat& train_desc,
        bool use_ratio = true
    );

    /// triangulate points between two frames and add them to the map
    int triangulate_and_add(Frame::Ptr ref, Frame::Ptr cur,
                            const std::vector<cv::DMatch>& matches);

    /// GPU stereo epipolar matching: fills frame->uR with right x-coords
    void match_stereo(Frame::Ptr frame);

    /// triangulate metric map points from a single stereo frame (frame->uR must be set)
    int triangulate_stereo(Frame::Ptr frame);

    /// attempt to recover pose against the full global map when LOST
    bool try_relocalize(Frame::Ptr frame);

    /// median angular parallax (radians) of tracked map points between frame and ref_kf
    double compute_median_parallax(Frame::Ptr frame, Frame::Ptr ref_kf) const;

    Camera         cam_;
    Map::Ptr       map_;
    Config         cfg_;
    TrackingState  state_ = TrackingState::NOT_INITIALIZED;

    // --- Legacy ORB frontend (monocular fallback / non-hybrid mode) ---
    cv::Ptr<cv::ORB> orb_;

    // --- Deep frontend components (hybrid mode; null in non-hybrid) ---
    std::shared_ptr<deep::XFeatExtractor>   xfeat_;
    std::shared_ptr<deep::LighterGlueAsync> lighter_glue_;
    std::shared_ptr<deep::TTTLoopDetector>  ttt_;
    bool hybrid_mode_ = false;

    // L2 matching device buffers (allocated lazily on first use)
    // Only non-null in hybrid mode.
    __half* d_query_descs_ = nullptr;
    __half* d_train_descs_ = nullptr;
    int*    d_best_idx_    = nullptr;
    float*  d_best_dist_   = nullptr;
    float*  d_pseudo_conf_ = nullptr;
    int     d_buf_capacity_ = 0;  // number of descriptors each buffer can hold

    Frame::Ptr last_frame_;
    Frame::Ptr last_keyframe_;

    // PnP inlier count at the last KF insertion (pre-triangulation)
    int last_kf_pnp_tracked_ = 0;

    // constant velocity motion model
    Eigen::Isometry3d velocity_ = Eigen::Isometry3d::Identity();
    bool velocity_valid_ = false;

    // coasting / lost
    int lost_streak_     = 0;
    int reloc_wait_frames_ = 0;  // frames elapsed since LG job was submitted
};

}  // namespace slam
