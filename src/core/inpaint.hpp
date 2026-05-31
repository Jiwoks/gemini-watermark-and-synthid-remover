#pragma once

#include <opencv2/core.hpp>

namespace wmr {

enum class InpaintMethod {
    Gaussian,
    Telea,
    NavierStokes
};

struct InpaintConfig {
    float strength = 0.85f;
    InpaintMethod method = InpaintMethod::Gaussian;
    int radius = 10;
    int padding = 32;
    bool full_mask = false;  // use full alpha region as inpaint mask (vs gradient edges)
};

void inpaint_residual(
    cv::Mat& image,
    const cv::Rect& region,
    const cv::Mat& alpha_map,
    const InpaintConfig& config = InpaintConfig{});

} // namespace wmr
