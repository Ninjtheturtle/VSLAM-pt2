#include "slam/camera.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace slam {

Camera Camera::from_kitti_calib(const std::string& calib_file)
{
    // KITTI calib.txt format — each row is a label followed by 12 values.
    // P0: 3×4 projection matrix for left grayscale camera  (baseline = 0)
    // P1: 3×4 projection matrix for right grayscale camera (P1[3] = -fx·b)
    // Baseline b = -P1[3] / fx

    std::ifstream f(calib_file);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open calib file: " + calib_file);
    }

    Camera cam;
    bool got_p0 = false, got_p1 = false;

    std::string line;
    while (std::getline(f, line)) {
        if (!got_p0 && line.rfind("P0:", 0) == 0) {
            std::istringstream ss(line.substr(3));
            double vals[12];
            for (int i = 0; i < 12; ++i) ss >> vals[i];
            // P0 = K * [I | 0]: vals = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
            cam.fx = vals[0];
            cam.fy = vals[5];
            cam.cx = vals[2];
            cam.cy = vals[6];
            cam.k1 = 0.0;
            cam.k2 = 0.0;
            cam.width  = 0;
            cam.height = 0;
            got_p0 = true;
        } else if (!got_p1 && line.rfind("P1:", 0) == 0) {
            std::istringstream ss(line.substr(3));
            double vals[12];
            for (int i = 0; i < 12; ++i) ss >> vals[i];
            // P1[3] = -fx·b  →  b = -P1[3] / fx
            if (cam.fx > 0.0)
                cam.baseline = -vals[3] / cam.fx;
            got_p1 = true;
        }
        if (got_p0 && got_p1) break;
    }

    if (!got_p0)
        throw std::runtime_error("P0 not found in calib file: " + calib_file);

    // ── Diagnostic: compare P0/P1 (grayscale) vs P2/P3 (color) baselines ────
    // A wrong camera choice causes scale error:
    //   KITTI seq 00 — P1 baseline ≈ 0.537 m, P3 baseline ≈ 0.469 m (13% smaller)
    // 13% smaller baseline → 13% underestimated depths → loops 13% smaller than GT.
    {
        std::ifstream f2(calib_file);
        bool got_p2 = false, got_p3 = false;
        double fx_p2 = cam.fx, p3_t = 0.0;
        std::string ln;
        while (std::getline(f2, ln)) {
            if (!got_p2 && ln.rfind("P2:", 0) == 0) {
                std::istringstream ss(ln.substr(3));
                double v[12]; for (int i = 0; i < 12; ++i) ss >> v[i];
                fx_p2 = v[0]; got_p2 = true;
            }
            if (!got_p3 && ln.rfind("P3:", 0) == 0) {
                std::istringstream ss(ln.substr(3));
                double v[12]; for (int i = 0; i < 12; ++i) ss >> v[i];
                p3_t = v[3]; got_p3 = true;
            }
            if (got_p2 && got_p3) break;
        }
        if (got_p2 && got_p3 && fx_p2 > 0.0) {
            double b_color = -p3_t / fx_p2;
            std::cout << "[Camera] Baseline P0/P1(used)=" << cam.baseline
                      << " m  P2/P3=" << b_color << " m";
            if (std::abs(cam.baseline - b_color) > 0.01)
                std::cout << "  *** MISMATCH — are you loading image_2/image_3 instead of image_0/image_1?";
            std::cout << "\n";
        }
    }

    return cam;
}

}  // namespace slam
