#pragma once

#include <opencv2/core.hpp>
#include <optional>

#include "core/types.hpp"

namespace wmr {

class NccDetector {
public:
    NccDetector(const cv::Mat& alpha_small, const cv::Mat& alpha_large,
                const cv::Mat& alpha_veo_text = {});

    DetectionResult detect(
        const cv::Mat& image,
        std::optional<WatermarkSize> force_size = std::nullopt,
        std::optional<WatermarkPosition> force_position = std::nullopt,
        const cv::Mat* custom_alpha = nullptr,
        bool enable_snap = false) const;

    const cv::Mat& get_veo_text_alpha() const { return alpha_map_veo_text_; }

private:
    cv::Mat alpha_map_small_;
    cv::Mat alpha_map_large_;
    cv::Mat alpha_map_veo_text_;

    const cv::Mat& get_alpha_map(WatermarkSize size) const;
};

} // namespace wmr
