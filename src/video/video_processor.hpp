#pragma once

#include <string>
#include <cstdint>
#include <chrono>
#include <opencv2/core.hpp>

#include "core/types.hpp"
#include "core/inpaint.hpp"
#include "video/video_writer.hpp"

namespace wmr {

struct VideoWatermarkConfig {
    VideoProfile profile = VideoProfile::GeminiDiamond;
    VideoVariant variant = VideoVariant::Auto;
    bool force = false;
    float inpaint_strength = 0.85f;
    bool scenes = false;
    double scene_threshold = 0.4;
    InpaintMethod inpaint_method = InpaintMethod::Telea;
    float denoise_sigma = 50.0f;
    int denoise_padding = 32;
    int denoise_radius = 10;
};

struct VideoResult {
    int64_t frames_processed = 0;
    int64_t frames_skipped = 0;
    float detection_confidence = 0.0f;
    double elapsed_seconds = 0.0;
    bool success = false;
    std::string message;
};

class VideoProcessor {
public:
    VideoResult process(const std::string& input_path,
                        const std::string& output_path,
                        const VideoWatermarkConfig& config,
                        const EncodeOptions& encode_opts = {});

private:
    static constexpr int kShotSampleCount = 12;
    static constexpr double kShotCoverage = 0.9;       // sample first 90% of video
    static constexpr int kPositionTolerancePx = 4;      // ±4px from shot anchor
    static constexpr float kOcclusionGateNcc = 0.35f;   // skip frames below this NCC
    static constexpr int kProgressIntervalFrames = 100;

    struct ShotDetection {
        bool found = false;
        cv::Point position;
        WatermarkSize size = WatermarkSize::Small;
        float confidence = 0.0f;
        cv::Rect region;
        WatermarkPosition geo = {72, 72, 48};
    };

    ShotDetection detect_in_shot(class VideoReader& reader,
                                 class WatermarkEngine& engine,
                                 const VideoWatermarkConfig& config,
                                 int64_t range_start = -1,
                                 int64_t range_end = -1,
                                 int max_samples = kShotSampleCount);

    WatermarkPosition resolve_geometry(
        const VideoWatermarkConfig& config, int width, int height) const;

    WatermarkSize geometry_to_size(const WatermarkPosition& geo) const;

    WatermarkPosition auto_detect_watermark_geometry(
        class VideoReader& reader,
        class WatermarkEngine& engine,
        VideoProfile profile,
        double& out_score);
};

} // namespace wmr
