#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace wmr {

enum class [[nodiscard]] ResultCode {
    Success,
    FileNotFound,
    InvalidFormat,
    ProcessingFailed,
    SaveFailed
};

enum class WatermarkSize { Small, Large };

struct ProcessResult {
    bool success = false;
    bool skipped = false;
    float confidence = 0.0f;
    std::string message;
};

struct DetectionResult {
    bool detected = false;
    float confidence = 0.0f;
    cv::Rect region;
    WatermarkSize size = WatermarkSize::Small;
    float spatial_score = 0.0f;
    float gradient_score = 0.0f;
    float variance_score = 0.0f;
};

inline WatermarkSize get_watermark_size(int width, int height) {
    return (width > 1024 && height > 1024) ? WatermarkSize::Large : WatermarkSize::Small;
}

struct WatermarkPosition {
    int margin_right;
    int margin_bottom;
    int logo_size;

    cv::Point get_position(int image_width, int image_height) const {
        return {image_width - margin_right - logo_size,
                image_height - margin_bottom - logo_size};
    }
};

inline WatermarkPosition get_watermark_config(int width, int height) {
    auto size = get_watermark_size(width, height);
    if (size == WatermarkSize::Large) {
        return {64, 64, 96};
    }
    return {32, 32, 48};
}

// Video-specific watermark geometry
// Gemini/Veo videos use different positions than still images
enum class VideoVariant {
    Auto,       // auto-detect from resolution
    P720_1,     // 720p standard (48x48, margin 72,72)
    P720_2,     // 720p compact (44x44, margin 29,40)
    P1080p,     // 1080p (96x96, margin 192,192)
};

enum class VideoProfile {
    GeminiDiamond,
    VeoLegacy,
};

inline WatermarkPosition get_video_watermark_geometry(
    VideoVariant variant, int width, int height, VideoProfile profile = VideoProfile::GeminiDiamond)
{
    // Veo legacy text watermark — different shape and position
    if (profile == VideoProfile::VeoLegacy) {
        // Reference alpha maps: 68x30 (small), 99x43 (large)
        switch (variant) {
            case VideoVariant::P1080p:
                return {17, 18, 99};  // 99x43 large Veo text
            default:
                return {17, 18, 68};  // 68x30 small Veo text
        }
    }

    // Gemini V2 diamond watermark
    switch (variant) {
        case VideoVariant::P720_1:
            return {72, 72, 48};
        case VideoVariant::P720_2:
            return {29, 40, 44};
        case VideoVariant::P1080p:
            return {192, 192, 96};
        case VideoVariant::Auto:
            break;
    }

    // Auto-detect from resolution
    int max_dim = std::max(width, height);
    if (max_dim >= 1920) {
        return {192, 192, 96};
    }
    // Default to 720p-1 for 720p content
    return {72, 72, 48};
}

} // namespace wmr
