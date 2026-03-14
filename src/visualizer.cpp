// Rerun.io logging for SLAM state. 3D entities live under "world/", not "world/camera/".

#include "slam/visualizer.hpp"
#include "deep/semi_dense_disparity.hpp"  // deep::SemiDensePoint3D

#include <rerun.hpp>
#include <rerun/archetypes/image.hpp>
#include <rerun/archetypes/pinhole.hpp>
#include <rerun/archetypes/points2d.hpp>
#include <rerun/archetypes/points3d.hpp>
#include <rerun/archetypes/line_strips3d.hpp>
#include <rerun/archetypes/transform3d.hpp>
#include <rerun/archetypes/view_coordinates.hpp>
#include <rerun/components/color.hpp>
#include <rerun/blueprint/archetypes/viewport_blueprint.hpp>

#include <opencv2/imgproc.hpp>

#include <Eigen/Geometry>

#include <iostream>
#include <vector>

namespace slam {

// factory

Visualizer::Ptr Visualizer::create(const Config& cfg)
{
    auto v = std::shared_ptr<Visualizer>(new Visualizer());
    v->cfg_ = cfg;

    v->rec_ = std::make_unique<rerun::RecordingStream>(cfg.app_id);
    auto result = v->rec_->connect_tcp(cfg.addr);
    if (!result.is_ok()) {
        std::cerr << "[Visualizer] Warning: could not connect to Rerun at "
                  << cfg.addr << " — " << result.description << "\n"
                  << "  Start the viewer with:  rerun\n";
    } else {
        std::cout << "[Visualizer] Connected to Rerun at " << cfg.addr << "\n";
    }

    // reset viewport so cached blueprints don't block the 3D view
    {
        rerun::RecordingStream bp(cfg.app_id, "", rerun::StoreKind::Blueprint);
        bp.connect_tcp(cfg.addr);
        bp.log_static("viewport",
            rerun::blueprint::archetypes::ViewportBlueprint()
                .with_auto_layout(true)
                .with_auto_views(true)
                .with_past_viewer_recommendations({}));
    }

    // RDF = KITTI camera convention; must be on "world" so Rerun creates a 3D view
    v->rec_->log_static("world", rerun::archetypes::ViewCoordinates::RDF);

    // tiny anchor point so Rerun auto-creates the 3D view at startup before any data arrives
    v->rec_->log_static("world/origin",
        rerun::archetypes::Points3D({{0.0f, 0.0f, 0.0f}})
            .with_radii(std::vector<float>{0.001f}));

    return v;
}

Visualizer::~Visualizer() = default;

// log_pinhole

void Visualizer::log_pinhole(const Camera& cam)
{
    if (!rec_) return;
    cam_ = cam;  // store for depth projection in log_frame()

    // pinhole intrinsics — renders the camera frustum and 2D image panel
    rec_->log_static("world/camera/image",
        rerun::archetypes::Pinhole::from_focal_length_and_resolution(
            {(float)cam.fx, (float)cam.fy},
            {(float)cam.width, (float)cam.height}
        )
    );

    // identity anchor so the frustum shows before tracking starts
    rec_->log_static("world/camera/image",
        rerun::archetypes::Transform3D::from_translation_rotation(
            {0.0f, 0.0f, 0.0f},
            rerun::datatypes::Quaternion::from_wxyz(1.0f, 0.0f, 0.0f, 0.0f)));
}

// log_frame

void Visualizer::log_frame(const Frame::Ptr& frame)
{
    if (!rec_) return;

    // advance timeline so the viewer can scrub through the sequence
    rec_->set_time_seconds("time", frame->timestamp);

    // camera pose in world space — moves the 3D frustum each frame
    if (frame->num_tracked() > 0) {
        Eigen::Isometry3d  T_wc = frame->T_wc();
        Eigen::Quaterniond q(T_wc.rotation());
        Eigen::Vector3d    t = T_wc.translation();
        rec_->log("world/camera/image",
            rerun::archetypes::Transform3D::from_translation_rotation(
                {(float)t.x(), (float)t.y(), (float)t.z()},
                rerun::datatypes::Quaternion::from_wxyz(
                    (float)q.w(), (float)q.x(), (float)q.y(), (float)q.z())));
    }

    // camera image → 2D panel (static Pinhole set in log_pinhole() stays on this entity)
    if (cfg_.log_image && !frame->image_gray.empty()) {
        cv::Mat rgb;
        cv::cvtColor(frame->image_gray, rgb, cv::COLOR_GRAY2RGB);
        auto bytes = std::vector<uint8_t>(rgb.data, rgb.data + rgb.total() * 3);
        rec_->log("world/camera/image",
            rerun::archetypes::Image::from_rgb24(
                std::move(bytes), {(uint32_t)rgb.cols, (uint32_t)rgb.rows}));
    }

    // purple keypoints overlaid on the 2D panel
    if (cfg_.log_keypoints && !frame->keypoints.empty()) {
        std::vector<rerun::datatypes::Vec2D> pts;
        pts.reserve(frame->keypoints.size());
        for (auto& kp : frame->keypoints)
            pts.push_back({kp.pt.x, kp.pt.y});
        rec_->log("world/camera/image/keypoints",
            rerun::archetypes::Points2D(pts)
                .with_colors(rerun::components::Color(190, 75, 230, 220))
                .with_radii(std::vector<float>(pts.size(), 2.5f)));
    }

}

// rebuilds trajectory every call so BA corrections show up automatically.
// KFs more than 50 m apart split into a new strip.

void Visualizer::log_trajectory(const Map::Ptr& map,
                                 const Frame::Ptr& current_frame,
                                 double ts)
{
    if (!rec_) return;
    rec_->set_time_seconds("time", ts);

    // archived KFs survive map_->reset(); active KFs are the current segment
    auto archived = map->trajectory_archive();   // KFs from all prior map segments
    auto kfs      = map->all_keyframes();         // current active KFs

    constexpr double kGapSq = 50.0 * 50.0;  // 50 m gap → new strip segment

    std::vector<std::vector<rerun::datatypes::Vec3D>> segments;
    Eigen::Vector3d last_c;
    bool has_last = false;

    auto add_kf_pos = [&](const Frame::Ptr& kf) {
        Eigen::Vector3d c = kf->camera_center();
        if (!has_last || (c - last_c).squaredNorm() > kGapSq) {
            segments.emplace_back();
        }
        segments.back().push_back({(float)c.x(), (float)c.y(), (float)c.z()});
        last_c   = c;
        has_last = true;
    };

    for (auto& kf : archived) add_kf_pos(kf);
    for (auto& kf : kfs)      add_kf_pos(kf);

    // append live frame if tracked but not yet a KF — avoids duplicating the just-inserted KF
    if (current_frame && current_frame->num_tracked() > 0 && !current_frame->is_keyframe) {
        Eigen::Vector3d c = current_frame->camera_center();
        if (!has_last || (c - last_c).squaredNorm() > kGapSq) {
            segments.emplace_back();
        }
        if (segments.empty()) segments.emplace_back();
        segments.back().push_back({(float)c.x(), (float)c.y(), (float)c.z()});
    }

    if (segments.empty()) return;

    std::vector<rerun::components::LineStrip3D> strips;
    strips.reserve(segments.size());
    for (auto& seg : segments)
        strips.emplace_back(seg);

    rec_->log("world/trajectory",
        rerun::archetypes::LineStrips3D(strips)
            .with_colors({rerun::components::Color(0, 255, 128)})   // bright green
            .with_radii(std::vector<float>(strips.size(), 0.5f)));
}

// log_map

void Visualizer::log_map(const Map::Ptr& map, double timestamp)
{
    if (!rec_) return;
    rec_->set_time_seconds("time", timestamp);

    auto map_pts = map->all_map_points();
    if (map_pts.empty()) return;

    std::vector<rerun::datatypes::Vec3D> positions;
    positions.reserve(map_pts.size());

    for (auto& mp : map_pts) {
        auto& p = mp->position;
        positions.push_back({(float)p.x(), (float)p.y(), (float)p.z()});
    }

    rec_->log("world/map/points",
        rerun::archetypes::Points3D(positions)
            .with_radii(std::vector<float>(positions.size(), 0.03f))
    );
}

// log_ground_truth

void Visualizer::log_ground_truth(const std::vector<std::array<float, 3>>& centers)
{
    if (!rec_ || centers.size() < 2) return;

    std::vector<rerun::datatypes::Vec3D> pts;
    pts.reserve(centers.size());
    for (auto& c : centers)
        pts.push_back({c[0], c[1], c[2]});

    rec_->log_static("world/ground_truth/trajectory",
        rerun::archetypes::LineStrips3D(
            {rerun::components::LineStrip3D(pts)})
            .with_colors({rerun::components::Color(255, 165, 0)})  // orange
            .with_radii({0.5f}));
}

// log_semi_dense — visualization-only semi-dense point cloud
// Entity: world/map/semi_dense  (light blue)
// ISOLATION: nothing from this function enters the Map or Ceres problem.

void Visualizer::log_semi_dense(
    const std::vector<deep::SemiDensePoint3D>& pts, double ts)
{
    if (!rec_ || pts.empty()) return;

    rec_->set_time_seconds("time", ts);

    std::vector<rerun::datatypes::Vec3D> positions;
    positions.reserve(pts.size());
    for (auto& p : pts)
        positions.push_back({p.x, p.y, p.z});

    rec_->log("world/map/semi_dense",
        rerun::archetypes::Points3D(positions)
            .with_radii({0.01f})
            .with_colors({rerun::components::Color(100, 200, 255)}));
}

}  // namespace slam
