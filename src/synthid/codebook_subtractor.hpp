#pragma once

#include <opencv2/core.hpp>
#include <string>

#include "core/fft_context.hpp"
#include "synthid/spectral_codebook.hpp"

namespace wmr {

enum class RemovalStrength {
    Gentle,
    Moderate,
    Aggressive,
    Maximum
};

struct RemovalConfig {
    RemovalStrength strength = RemovalStrength::Moderate;
    float custom_strength = -1.0f;  // Override: 0.0-1.0 if >= 0
};

class CodebookSubtractor {
public:
    CodebookSubtractor(FftContext& fft);

    void remove_synthid(cv::Mat& image,
                        const SpectralCodebook& codebook,
                        const RemovalConfig& config = {});

private:
    FftContext& fft_;

    static constexpr float kChannelWeights[3] = {0.85f, 1.0f, 0.70f};  // R, G, B

    struct StrengthParams {
        float removal;
        float cons_floor;
        float mag_cap;
        float dc_radius;
    };

    static StrengthParams get_strength_params(RemovalStrength strength);
    cv::Mat estimate_watermark_fft(
        const cv::Mat& image_fft,
        int channel,
        float removal_factor,
        float cons_floor,
        float mag_cap,
        float dc_radius,
        const SpectralProfile& profile,
        float image_luminance);
};

} // namespace wmr
