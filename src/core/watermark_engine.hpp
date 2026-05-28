#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <memory>

#include "core/types.hpp"
#include "core/inpaint.hpp"
#include "embedded_assets.hpp"

namespace wmr {

class NccDetector;

class WatermarkEngine {
public:
    WatermarkEngine();
    ~WatermarkEngine();

    void remove_watermark(cv::Mat& image,
                          std::optional<WatermarkSize> force_size = std::nullopt);

    void remove_watermark_detected(cv::Mat& image,
                                   const DetectionResult& detection);

    DetectionResult detect_watermark(
        const cv::Mat& image,
        std::optional<WatermarkSize> force_size = std::nullopt) const;

    void inpaint_residual(cv::Mat& image,
                          const cv::Rect& region,
                          const InpaintConfig& config = {}) const;

    void add_watermark(cv::Mat& image,
                       std::optional<WatermarkSize> force_size = std::nullopt);

    const cv::Mat& get_alpha_map(WatermarkSize size) const;

private:
    cv::Mat alpha_map_small_;
    cv::Mat alpha_map_large_;
    float logo_value_ = 255.0f;

    std::unique_ptr<NccDetector> detector_;

    cv::Mat create_interpolated_alpha(int width, int height, WatermarkSize size) const;
    void init_alpha_maps();
};

} // namespace wmr
