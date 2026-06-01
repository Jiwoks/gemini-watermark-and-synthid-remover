#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <string>
#include <vector>

#include "core/types.hpp"

namespace wmr {

struct SceneInfo {
    int64_t start_frame = 0;   // inclusive
    int64_t end_frame = 0;     // exclusive (half-open interval)
    bool has_watermark = false;
    // Full detection data for this scene (populated by VideoProcessor):
    bool detected = false;
    cv::Point position;
    WatermarkSize size = WatermarkSize::Small;
    float confidence = 0.0f;
    cv::Rect region;
};

struct SceneDetectorConfig {
    double threshold = 0.4;        // Bhattacharyya distance for hard cut
    int min_scene_length = 15;     // minimum frames between cuts
};

class SceneDetector {
public:
    explicit SceneDetector(SceneDetectorConfig config = {});

    // Opens its own VideoReader internally, scans for scene boundaries.
    // Returns vector of SceneInfo with start_frame/end_frame populated.
    // has_watermark and detection fields are NOT populated here.
    std::vector<SceneInfo> detect_boundaries(const std::string& video_path);

private:
    SceneDetectorConfig config_;

    // Downsample to grayscale for histogram comparison
    cv::Mat prepare_frame(const cv::Mat& frame) const;

    // Bhattacharyya distance between two grayscale frames
    double compute_distance(const cv::Mat& prev, const cv::Mat& curr) const;

    // Merge scenes shorter than min_scene_length into predecessor
    std::vector<SceneInfo> merge_short_scenes(
        std::vector<int64_t>&& boundaries, int64_t total_frames) const;
};

} // namespace wmr
