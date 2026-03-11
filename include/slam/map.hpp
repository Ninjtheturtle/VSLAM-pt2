#pragma once

#include "slam/frame.hpp"
#include "slam/map_point.hpp"
#include <deque>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <vector>

namespace slam {

/// Thread-safe container for all active map state.
///
/// Maintains:
///   • All keyframes (ordered by insertion, also indexed by id)
///   • All active 3-D map points
///   • A sliding window of the N most recent keyframes for local BA
class Map {
public:
    static constexpr int kWindowSize = 30;  // sliding window for local BA

    using Ptr = std::shared_ptr<Map>;
    static Ptr create() { return Ptr(new Map()); }

    // ── Keyframe management ──────────────────────────────────────────────────
    void insert_keyframe(Frame::Ptr kf);
    void remove_keyframe(long id);
    Frame::Ptr get_keyframe(long id) const;

    /// Ordered (oldest first) list of all keyframes
    std::vector<Frame::Ptr> all_keyframes() const;

    /// The last kWindowSize keyframes (or all if fewer) — used for BA
    std::vector<Frame::Ptr> local_window() const;

    /// The last n keyframes (or all if fewer) — used for the tracking descriptor pool
    std::vector<Frame::Ptr> local_window(int n) const;

    // ── Map point management ─────────────────────────────────────────────────
    void insert_map_point(MapPoint::Ptr mp);
    void remove_map_point(long id);
    MapPoint::Ptr get_map_point(long id) const;

    /// All active (non-bad) map points
    std::vector<MapPoint::Ptr> all_map_points() const;

    /// Remove map points flagged as bad
    void cleanup_bad_map_points();

    /// Clear all keyframes and map points (call before re-initialization).
    /// Automatically archives current keyframes into trajectory_archive_ first.
    void reset();

    /// Keyframes archived from all previous map resets — for trajectory visualization only.
    /// Never included in BA or the tracking descriptor pool.
    std::vector<Frame::Ptr> trajectory_archive() const;

    /// Count map points observed by both kf_id_a and kf_id_b (via observations map).
    /// Used by PoseGraph to detect co-visibility loop candidates.
    int count_shared_map_points(long kf_id_a, long kf_id_b) const;

    // ── Statistics ───────────────────────────────────────────────────────────
    size_t num_keyframes()  const;
    size_t num_map_points() const;

private:
    Map() = default;

    mutable std::mutex kf_mutex_;
    mutable std::mutex mp_mutex_;

    // Ordered insertion order: deque of (id, ptr)
    std::deque<Frame::Ptr>                         keyframe_order_;
    std::unordered_map<long, Frame::Ptr>            keyframes_;
    std::unordered_map<long, MapPoint::Ptr>         map_points_;

    // Trajectory archive — NOT cleared by reset(); accumulates KFs from all prior segments.
    std::vector<Frame::Ptr> trajectory_archive_;
};

}  // namespace slam
