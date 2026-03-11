// main.cpp — VSLAM system entry point
//
// Usage:
//   vslam.exe --sequence <path/to/kitti/sequence/XX>
//             [--start <frame_idx>]
//             [--end   <frame_idx>]
//             [--no-viz]

#include "slam/camera.hpp"
#include "slam/frame.hpp"
#include "slam/map.hpp"
#include "slam/tracker.hpp"
#include "slam/local_ba.hpp"
#include "slam/pose_graph.hpp"
#include "slam/visualizer.hpp"

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

// ─── KITTI Sequence Loader ────────────────────────────────────────────────────

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

        // Camera calibration (also extracts stereo baseline from P1)
        seq.camera = slam::Camera::from_kitti_calib(seq_path + "/calib.txt");

        // Timestamps
        std::ifstream tf(seq_path + "/times.txt");
        if (!tf.is_open())
            throw std::runtime_error("Cannot open times.txt in " + seq_path);
        double t;
        while (tf >> t) seq.timestamps.push_back(t);

        // Left image paths (image_0/)
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

        // Right image paths (image_1/) — optional; enables stereo mode
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

        // ── Calibration diagnostic ───────────────────────────────────────────
        // Decomposed from P0 (intrinsics) and P1 (baseline = -P1[3]/fx).
        // KITTI seq 00 expected: fx≈718.9, cy≈185.2, baseline≈0.537 m
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

// ─── Simple argument parser ───────────────────────────────────────────────────

struct Args {
    std::string sequence_path;
    int start_idx  = 0;
    int end_idx    = -1;   // -1 = all frames
    bool no_viz    = false;
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

// ─── Ground-truth helpers ─────────────────────────────────────────────────────

// Derive the KITTI GT poses path from the sequence path.
// e.g.  "data/sequences/00"  →  "data/poses/00.txt"
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

// Load KITTI-format pose file (3×4 row-major matrices, one per line).
// Returns the camera-centre column (4th column = tx, ty, tz) for every frame.
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
        // Row-major [R|t]: indices 3, 7, 11 are tx, ty, tz
        out.push_back({(float)v[3], (float)v[7], (float)v[11]});
    }
    return out;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    Args args = parse_args(argc, argv);

    // Load sequence
    KittiSequence seq = KittiSequence::load(args.sequence_path);

    // Set camera image dimensions from first frame
    {
        cv::Mat img = cv::imread(seq.image_paths[0], cv::IMREAD_GRAYSCALE);
        seq.camera.width  = img.cols;
        seq.camera.height = img.rows;
    }

    // Build SLAM system
    auto map        = slam::Map::create();
    auto tracker    = slam::Tracker::create(seq.camera, map);
    auto local_ba   = slam::LocalBA::create(seq.camera, map);
    auto pose_graph = slam::PoseGraph::create(map, seq.camera);
    slam::Visualizer::Ptr viz;

    if (!args.no_viz) {
        viz = slam::Visualizer::create();  // default: log_image=true, log_keypoints=true
        viz->log_pinhole(seq.camera);      // tells Rerun to show camera frustum + image panel

        // Overlay ground-truth trajectory (orange) if poses file is available.
        std::string gt_path = derive_gt_path(args.sequence_path);
        auto gt_centers = load_gt_centers(gt_path);
        if (!gt_centers.empty()) {
            viz->log_ground_truth(gt_centers);
            std::cout << "[VSLAM] GT overlay: " << gt_centers.size()
                      << " poses from " << gt_path << "\n";
        } else {
            std::cout << "[VSLAM] No GT file at " << gt_path << " — skipping overlay\n";
        }
    }

    // Frame range
    int n_frames  = static_cast<int>(seq.image_paths.size());
    int start_idx = std::max(0, args.start_idx);
    int end_idx   = (args.end_idx < 0) ? n_frames : std::min(args.end_idx, n_frames);

    std::cout << "[VSLAM] Processing frames " << start_idx
              << " to " << end_idx - 1 << "\n";

    // ── Main tracking loop ────────────────────────────────────────────────────
    long frame_count = 0;
    auto t_start_wall = std::chrono::steady_clock::now();

    for (int i = start_idx; i < end_idx; ++i) {
        // Load image
        cv::Mat img = cv::imread(seq.image_paths[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "[VSLAM] Failed to load: " << seq.image_paths[i] << "\n";
            continue;
        }

        double ts = (i < (int)seq.timestamps.size()) ? seq.timestamps[i] : (double)i;
        auto frame = slam::Frame::create(img, ts, i);

        // Attach right image for stereo mode
        if (i < (int)seq.image_right_paths.size()) {
            frame->image_right = cv::imread(seq.image_right_paths[i], cv::IMREAD_GRAYSCALE);
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        // Track
        bool ok = tracker->track(frame);

        // (Trajectory segmentation is handled automatically in log_trajectory()
        //  via spatial-gap detection — no manual segment call needed here.)

        auto t1 = std::chrono::high_resolution_clock::now();
        double track_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Local BA after each new keyframe; refresh velocity so the next
        // frame's prediction is based on the BA-refined poses, not pre-BA PnP.
        if (frame->is_keyframe && map->num_keyframes() >= 2) {
            local_ba->optimize();
            tracker->notify_ba_update();

            // Register new KF with pose graph; run loop detection + PGO every 5 KFs.
            pose_graph->add_keyframe(frame);
            if (map->num_keyframes() % 5 == 0) {
                // Co-visibility detection (fast, catches near-revisits)
                pose_graph->detect_and_add_loops();
                // Appearance-based detection (works for long-range loop closures)
                pose_graph->detect_and_add_loops_visual(frame);
                if (pose_graph->has_new_loops()) {
                    pose_graph->optimize();
                    tracker->notify_ba_update();
                }
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        double ba_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        // Visualization — image/keypoints every frame; BA-aware trajectory every frame;
        // map point cloud every keyframe.
        if (viz) {
            viz->log_frame(frame);
            viz->log_trajectory(map, frame, ts);
            if (frame->is_keyframe)
                viz->log_map(map, ts);
        }

        ++frame_count;

        // Per-frame status
        Eigen::Vector3d pos = frame->camera_center();
        printf("[%05d] track=%.1fms ba=%.1fms tracked=%3d kf=%zu pts=%zu "
               "pos=(%.2f,%.2f,%.2f) %s\n",
               i, track_ms, ba_ms,
               frame->num_tracked(),
               map->num_keyframes(),
               map->num_map_points(),
               pos.x(), pos.y(), pos.z(),
               ok ? "OK" : "LOST");
    }

    // ── Summary ───────────────────────────────────────────────────────────────
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
