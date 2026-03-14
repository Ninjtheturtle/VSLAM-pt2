// VSLAM entry point.
// usage: vslam.exe --sequence <path/to/kitti/sequence/XX> [--start N] [--end N]
//                  [--no-viz] [--hybrid] [--xfeat <engine>] [--lg <engine>]

#include "slam/camera.hpp"
#include "slam/frame.hpp"
#include "slam/map.hpp"
#include "slam/tracker.hpp"
#include "slam/local_ba.hpp"
#include "slam/pose_graph.hpp"
#include "slam/visualizer.hpp"

// Deep frontend (only compiled when ENABLE_DEEP_FRONTEND is defined in CMake)
#ifdef ENABLE_DEEP_FRONTEND
#include "deep/xfeat_extractor.hpp"
#include "deep/lighterglue_async.hpp"
#include "deep/ttt_autoencoder.hpp"
#include "deep/semi_dense_disparity.hpp"
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <atomic>

namespace fs = std::filesystem;

// KITTI sequence loader

struct KittiSequence {
    std::string sequence_path;
    std::vector<std::string> image_paths;        // left  (image_0/)
    std::vector<std::string> image_right_paths;  // right (image_1/); empty if not found
    std::vector<double>      timestamps;   // seconds
    slam::Camera             camera;

    static KittiSequence load(const std::string& seq_path)
    {
        KittiSequence seq;
        seq.sequence_path = seq_path;

        // camera calibration (also extracts stereo baseline from P1)
        seq.camera = slam::Camera::from_kitti_calib(seq_path + "/calib.txt");

        // timestamps
        std::ifstream tf(seq_path + "/times.txt");
        if (!tf.is_open())
            throw std::runtime_error("Cannot open times.txt in " + seq_path);
        double t;
        while (tf >> t) seq.timestamps.push_back(t);

        // left image paths (image_0/)
        fs::path img_dir = fs::path(seq_path) / "image_0";
        if (!fs::exists(img_dir))
            throw std::runtime_error("image_0/ not found in " + seq_path);

        std::vector<fs::path> paths;
        for (auto& entry : fs::directory_iterator(img_dir)) {
            if (entry.path().extension() == ".png")
                paths.push_back(entry.path());
        }
        std::sort(paths.begin(), paths.end());
        for (auto& p : paths)
            seq.image_paths.push_back(p.string());

        if (seq.image_paths.empty())
            throw std::runtime_error("No .png images found in " + img_dir.string());

        // right image paths (image_1/) — optional; enables stereo mode
        fs::path img_dir_r = fs::path(seq_path) / "image_1";
        if (fs::exists(img_dir_r)) {
            std::vector<fs::path> rpaths;
            for (auto& entry : fs::directory_iterator(img_dir_r)) {
                if (entry.path().extension() == ".png")
                    rpaths.push_back(entry.path());
            }
            std::sort(rpaths.begin(), rpaths.end());
            for (auto& p : rpaths)
                seq.image_right_paths.push_back(p.string());
        }

        std::cout << "[KITTI] Loaded " << seq.image_paths.size()
                  << " frames from " << seq_path;
        if (!seq.image_right_paths.empty())
            std::cout << " (stereo, b=" << seq.camera.baseline << " m)";
        std::cout << "\n";

        // calibration diagnostic
        std::cout << "[KITTI] Intrinsics: fx=" << seq.camera.fx
                  << "  fy=" << seq.camera.fy
                  << "  cx=" << seq.camera.cx
                  << "  cy=" << seq.camera.cy << "\n";
        if (seq.camera.is_stereo()) {
            std::cout << "[KITTI] Stereo baseline: " << seq.camera.baseline << " m";
            if (seq.camera.baseline < 0.3 || seq.camera.baseline > 0.8)
                std::cout << "  *** WARNING: outside expected range [0.30, 0.80] m"
                              " — verify calib.txt uses P0/P1 (grayscale), not P2/P3 (color)";
            std::cout << "\n";
        }

        return seq;
    }
};

// simple argument parser

struct Args {
    std::string sequence_path;
    int  start_idx        = 0;
    int  end_idx          = -1;   // -1 = all frames
    bool no_viz           = false;
    bool hybrid           = false; // enable deep frontend (requires ENABLE_DEEP_FRONTEND)
    std::string xfeat_engine;      // path to XFeat TRT engine
    std::string lg_engine;         // path to LighterGlue TRT engine
};

Args parse_args(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--sequence" && i + 1 < argc) {
            args.sequence_path = argv[++i];
        } else if (a == "--start" && i + 1 < argc) {
            args.start_idx = std::stoi(argv[++i]);
        } else if (a == "--end" && i + 1 < argc) {
            args.end_idx = std::stoi(argv[++i]);
        } else if (a == "--no-viz") {
            args.no_viz = true;
        } else if (a == "--hybrid") {
            args.hybrid = true;
        } else if (a == "--xfeat" && i + 1 < argc) {
            args.xfeat_engine = argv[++i];
        } else if (a == "--lg" && i + 1 < argc) {
            args.lg_engine = argv[++i];
        } else if (a == "--help" || a == "-h") {
            std::cout << "Usage: vslam.exe --sequence <path> [--start N] [--end N] [--no-viz]\n";
            exit(0);
        }
    }
    if (args.sequence_path.empty()) {
        std::cerr << "Error: --sequence <path> is required\n";
        exit(1);
    }
    return args;
}

// ground-truth helpers — derive poses path e.g. "data/sequences/00" → "data/poses/00.txt"
static std::string derive_gt_path(const std::string& seq_path)
{
    std::string p = seq_path;
    while (!p.empty() && (p.back() == '/' || p.back() == '\\')) p.pop_back();
    size_t s1   = p.find_last_of("/\\");
    std::string seq_id = (s1 == std::string::npos) ? p : p.substr(s1 + 1);
    std::string up1    = (s1 == std::string::npos) ? "." : p.substr(0, s1);
    size_t s2   = up1.find_last_of("/\\");
    std::string base   = (s2 == std::string::npos) ? "." : up1.substr(0, s2);
    return base + "/poses/" + seq_id + ".txt";
}

// load KITTI pose file — returns camera-centre (tx, ty, tz) per frame
static std::vector<std::array<float, 3>> load_gt_centers(const std::string& path)
{
    std::vector<std::array<float, 3>> out;
    std::ifstream f(path);
    if (!f.is_open()) return out;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        double v[12];
        for (int i = 0; i < 12; ++i) ss >> v[i];
        // row-major [R|t]: indices 3, 7, 11 are tx, ty, tz
        out.push_back({(float)v[3], (float)v[7], (float)v[11]});
    }
    return out;
}

// main

int main(int argc, char** argv)
{
    Args args = parse_args(argc, argv);

    // load sequence
    KittiSequence seq = KittiSequence::load(args.sequence_path);

    // set camera image dimensions from first frame
    {
        cv::Mat img = cv::imread(seq.image_paths[0], cv::IMREAD_GRAYSCALE);
        seq.camera.width  = img.cols;
        seq.camera.height = img.rows;
    }

    // build SLAM system
    auto map        = slam::Map::create();
    auto local_ba   = slam::LocalBA::create(seq.camera, map);
    auto pose_graph = slam::PoseGraph::create(map, seq.camera);
    slam::Visualizer::Ptr viz;

    // --- Deep frontend (hybrid mode) ---
    slam::Tracker::Ptr tracker;

#ifdef ENABLE_DEEP_FRONTEND
    if (args.hybrid) {
        std::string xfeat_path = args.xfeat_engine.empty()
            ? "models/xfeat_fp16.engine" : args.xfeat_engine;
        std::string lg_path = args.lg_engine.empty()
            ? "models/lighterglue_fp16.engine" : args.lg_engine;

        deep::XFeatExtractor::Config xfeat_cfg;
        xfeat_cfg.engine_path  = xfeat_path;
        xfeat_cfg.img_width    = seq.camera.width;
        xfeat_cfg.img_height   = seq.camera.height;
        auto xfeat = deep::XFeatExtractor::create(xfeat_cfg);

        deep::LighterGlueAsync::Config lg_cfg;
        lg_cfg.engine_path = lg_path;
        auto lg = deep::LighterGlueAsync::create(lg_cfg);

        deep::TTTLoopDetector::Config ttt_cfg;
        auto ttt = deep::TTTLoopDetector::create(ttt_cfg);

        tracker = slam::Tracker::create_hybrid(
            seq.camera, map, std::move(xfeat), std::move(lg), std::move(ttt));
        std::cout << "[VSLAM] Hybrid deep-geometric mode enabled\n";
    } else
#endif
    {
        tracker = slam::Tracker::create(seq.camera, map);
    }
    if (!args.no_viz) {
        viz = slam::Visualizer::create();
        viz->log_pinhole(seq.camera);

        std::string gt_path = derive_gt_path(args.sequence_path);
        auto gt_centers = load_gt_centers(gt_path);
        if (!gt_centers.empty()) {
            viz->log_ground_truth(gt_centers);
        }
    }

    // frame range
    int n_frames  = static_cast<int>(seq.image_paths.size());
    int start_idx = std::max(0, args.start_idx);
    int end_idx   = (args.end_idx < 0) ? n_frames : std::min(args.end_idx, n_frames);

    // main tracking loop
    long frame_count = 0;
    auto t_start_wall = std::chrono::steady_clock::now();

    for (int i = start_idx; i < end_idx; ++i) {
        // load image
        cv::Mat img = cv::imread(seq.image_paths[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "[VSLAM] Failed to load: " << seq.image_paths[i] << "\n";
            continue;
        }

        double ts = (i < (int)seq.timestamps.size()) ? seq.timestamps[i] : (double)i;
        auto frame = slam::Frame::create(img, ts, i);

        // attach right image for stereo mode
        if (i < (int)seq.image_right_paths.size()) {
            frame->image_right = cv::imread(seq.image_right_paths[i], cv::IMREAD_GRAYSCALE);
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        // track
        bool ok = tracker->track(frame);

        // trajectory segmentation is handled automatically in log_trajectory() via gap detection

        auto t1 = std::chrono::high_resolution_clock::now();
        double track_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // run local BA after each new KF; notify tracker so next prediction uses refined poses
        if (frame->is_keyframe && map->num_keyframes() >= 2) {
            local_ba->optimize();
            tracker->notify_ba_update();

            // register KF with pose graph; run loop detection + PGO every 5 KFs
            pose_graph->add_keyframe(frame);
            if (map->num_keyframes() % 5 == 0) {
                // co-visibility detection (fast, catches near-revisits)
                pose_graph->detect_and_add_loops();
                // appearance-based detection (works for long-range loop closures)
                pose_graph->detect_and_add_loops_visual(frame);
                if (pose_graph->has_new_loops()) {
                    pose_graph->optimize();
                    tracker->notify_ba_update();
                }
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        double ba_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        // visualization — image/keypoints every frame; map cloud every KF
        if (viz) {
            viz->log_frame(frame);
            viz->log_trajectory(map, frame, ts);
            if (frame->is_keyframe) {
                viz->log_map(map, ts);

#ifdef ENABLE_DEEP_FRONTEND
                // Semi-dense disparity (hybrid mode, stereo KFs only)
                if (args.hybrid && !frame->feat_map_left.empty()
                                && !frame->feat_map_right.empty()) {
                    deep::SemiDenseDisparity::Config sd_cfg;
                    sd_cfg.baseline  = (float)seq.camera.baseline;
                    sd_cfg.fx        = (float)seq.camera.fx;
                    sd_cfg.cx        = (float)seq.camera.cx;
                    sd_cfg.cy        = (float)seq.camera.cy;
                    static deep::SemiDenseDisparity semi_dense(sd_cfg);

                    auto sd_pts = semi_dense.compute(
                        frame->feat_map_left, frame->feat_map_right, frame->T_wc());
                    viz->log_semi_dense(sd_pts, ts);

                    // Release feature maps to free memory after use
                    frame->feat_map_left.release();
                    frame->feat_map_right.release();
                }
#endif
            }
        }

        ++frame_count;

        // per-frame status
        Eigen::Vector3d pos = frame->camera_center();
        fprintf(stderr, "[%05d] track=%.1fms ba=%.1fms tracked=%3d kf=%zu pts=%zu "
               "pos=(%.2f,%.2f,%.2f) %s\n",
               i, track_ms, ba_ms,
               frame->num_tracked(),
               map->num_keyframes(),
               map->num_map_points(),
               pos.x(), pos.y(), pos.z(),
               ok ? "OK" : "LOST");
    }

    // summary
    auto t_end_wall = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end_wall - t_start_wall).count();
    double fps = frame_count / elapsed_s;

    std::cout << "\n[VSLAM] Done. "
              << frame_count << " frames in " << elapsed_s << "s = "
              << fps << " FPS\n"
              << "  Keyframes : " << map->num_keyframes() << "\n"
              << "  Map points: " << map->num_map_points() << "\n";

    return 0;
}
