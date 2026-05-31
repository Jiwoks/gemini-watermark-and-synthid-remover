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
                                   const DetectionResult& detection,
                                   float inpaint_strength = 0.85f,
                                   const cv::Mat* custom_alpha = nullptr);

    // Alpha blend only — no inpaint. Caller handles residual cleanup.
    void remove_watermark_alpha_only(cv::Mat& image,
                                     const DetectionResult& detection,
                                     const cv::Mat* custom_alpha = nullptr);

    DetectionResult detect_watermark(
        const cv::Mat& image,
        std::optional<WatermarkSize> force_size = std::nullopt,
        std::optional<WatermarkPosition> force_position = std::nullopt,
        const cv::Mat* custom_alpha = nullptr) const;

    void inpaint_residual(cv::Mat& image,
                          const cv::Rect& region,
                          const InpaintConfig& config = {},
                          const cv::Mat* custom_alpha = nullptr) const;

    void add_watermark(cv::Mat& image,
                       std::optional<WatermarkSize> force_size = std::nullopt);

    const cv::Mat& get_alpha_map(WatermarkSize size) const;
    const cv::Mat& get_veo_text_alpha() const { return alpha_map_veo_text_; }
    const cv::Mat& get_v2_diamond_alpha_36() const { return alpha_map_v2_diamond_36_; }
    const cv::Mat& get_v2_diamond_alpha_small() const { return alpha_map_v2_diamond_small_; }
    const cv::Mat& get_v2_diamond_alpha_large() const { return alpha_map_v2_diamond_large_; }
    const cv::Mat& get_veo_text_alpha_small() const { return alpha_map_veo_text_small_; }
    const cv::Mat& get_veo_text_alpha_large() const { return alpha_map_veo_text_large_; }

private:
    cv::Mat alpha_map_small_;
    cv::Mat alpha_map_large_;
    cv::Mat alpha_map_veo_text_;
    cv::Mat alpha_map_v2_diamond_36_;
    cv::Mat alpha_map_v2_diamond_small_;
    cv::Mat alpha_map_v2_diamond_large_;
    cv::Mat alpha_map_veo_text_small_;
    cv::Mat alpha_map_veo_text_large_;
    float logo_value_ = 255.0f;

    std::unique_ptr<NccDetector> detector_;

    cv::Mat create_interpolated_alpha(int width, int height, WatermarkSize size) const;
    void init_alpha_maps();
};

} // namespace wmr
