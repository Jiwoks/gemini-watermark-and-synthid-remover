#include "synthid/noise_residual_subtractor.hpp"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace wmr {

NoiseResidualSubtractor::NoiseResidualSubtractor(FftContext& fft)
    : fft_(fft) {}

auto NoiseResidualSubtractor::get_strength_params(RemovalStrength strength) -> StrengthParams {
    switch (strength) {
        case RemovalStrength::Gentle:
            return {0.60f, 0.50f, 30.0f};
        case RemovalStrength::Moderate:
            return {0.80f, 0.70f, 25.0f};
        case RemovalStrength::Aggressive:
            return {0.95f, 0.90f, 20.0f};
        case RemovalStrength::Maximum:
            return {1.00f, 0.98f, 12.0f};
    }
    return {0.80f, 0.70f, 25.0f};
}

void NoiseResidualSubtractor::remove_synthid(
    cv::Mat& image,
    const RemovalConfig& config)
{
    if (image.empty()) return;

    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    const int h = image.rows;
    const int w = image.cols;

    RemovalStrength base_strength = config.strength;
    if (config.custom_strength >= 0.0f) {
        if (config.custom_strength <= 0.25f) base_strength = RemovalStrength::Gentle;
        else if (config.custom_strength <= 0.50f) base_strength = RemovalStrength::Moderate;
        else if (config.custom_strength <= 0.75f) base_strength = RemovalStrength::Aggressive;
        else base_strength = RemovalStrength::Maximum;
    }

    struct PassSchedule {
        RemovalStrength level;
        int count;
    };

    PassSchedule schedule[4];
    int num_passes = 0;

    switch (base_strength) {
        case RemovalStrength::Gentle:
            schedule[0] = {RemovalStrength::Gentle, 1};
            num_passes = 1;
            break;
        case RemovalStrength::Moderate:
            schedule[0] = {RemovalStrength::Moderate, 1};
            schedule[1] = {RemovalStrength::Gentle, 1};
            num_passes = 2;
            break;
        case RemovalStrength::Aggressive:
            schedule[0] = {RemovalStrength::Aggressive, 1};
            schedule[1] = {RemovalStrength::Moderate, 1};
            schedule[2] = {RemovalStrength::Gentle, 1};
            num_passes = 3;
            break;
        case RemovalStrength::Maximum:
            schedule[0] = {RemovalStrength::Maximum, 1};
            schedule[1] = {RemovalStrength::Maximum, 1};
            schedule[2] = {RemovalStrength::Maximum, 1};
            schedule[3] = {RemovalStrength::Maximum, 1};
            num_passes = 4;
            break;
    }

    cv::Mat work;
    image.convertTo(work, CV_32FC3, 1.0 / 255.0);

    cv::Mat channels[3];
    cv::split(work, channels);

    // Content-aware: for content images, carrier is <0.1% of spectral energy
    cv::Scalar img_std;
    cv::meanStdDev(work, cv::Scalar(), img_std);
    float avg_std = static_cast<float>((img_std[0] + img_std[1] + img_std[2]) / 3.0);
    bool is_content_image = avg_std > 0.05f;

    if (is_content_image) {
        spdlog::info("Content image detected (std={:.4f}): carrier <0.1% of spectral energy. "
                     "Skipping carrier subtraction, applying spectral disruption only.", avg_std);
        num_passes = 0;
    }

    spdlog::info("Codebook-free SynthID removal: {}x{}, {} passes, strength={}, content={}",
                 w, h, num_passes, static_cast<int>(base_strength), is_content_image);

    for (int pass = 0; pass < num_passes; ++pass) {
        auto params = get_strength_params(schedule[pass].level);

        spdlog::debug("Pass {}/{}: removal={:.2f}, mag_cap={:.2f}, dc_radius={:.0f}",
                      pass + 1, num_passes, params.removal, params.mag_cap, params.dc_radius);

        for (int ch = 0; ch < 3; ++ch) {
            cv::Mat img_fft = fft_.forward(channels[ch]);

            cv::Mat wm_estimate = estimate_carrier_from_noise(
                img_fft, channels[ch],
                params.removal, params.mag_cap, params.dc_radius, ch);

            cv::Mat cleaned_fft;
            cv::subtract(img_fft, wm_estimate, cleaned_fft);

            channels[ch] = fft_.inverse(cleaned_fft);
        }
    }

    // Phase noise disruption: scramble any remaining carrier phase encoding
    float phase_sigma = 0.0f;
    if (is_content_image) {
        phase_sigma = 0.10f;
    } else {
        switch (base_strength) {
            case RemovalStrength::Gentle:    phase_sigma = 0.15f; break;
            case RemovalStrength::Moderate:  phase_sigma = 0.30f; break;
            case RemovalStrength::Aggressive: phase_sigma = 0.50f; break;
            case RemovalStrength::Maximum:   phase_sigma = 0.70f; break;
        }
    }

    if (phase_sigma > 0.0f) {
        cv::Mat dc_ramp(h, w, CV_32FC1);
        for (int y = 0; y < h; ++y) {
            float fy = static_cast<float>(y);
            if (fy > h / 2.0f) fy -= h;
            for (int x = 0; x < w; ++x) {
                float fx = static_cast<float>(x);
                if (fx > w / 2.0f) fx -= w;
                float dist = std::sqrt(fy * fy + fx * fx);
                dc_ramp.at<float>(y, x) = std::clamp((dist - 40.0f) / 20.0f, 0.0f, 1.0f);
            }
        }

        cv::RNG rng(42);
        for (int ch = 0; ch < 3; ++ch) {
            cv::Mat ch_fft = fft_.forward(channels[ch]);
            cv::Mat mag = FftContext::magnitude(ch_fft);
            cv::Mat pha = FftContext::phase(ch_fft);

            cv::Mat phase_noise(h, w, CV_32FC1);
            rng.fill(phase_noise, cv::RNG::NORMAL, 0.0, phase_sigma);
            phase_noise = phase_noise.mul(dc_ramp);

            cv::Mat new_phase;
            cv::add(pha, phase_noise, new_phase);

            channels[ch] = fft_.inverse(FftContext::from_polar(mag, new_phase));
        }
        spdlog::debug("Applied phase noise disruption: sigma={:.2f}", phase_sigma);
    }

    cv::Mat merged;
    cv::merge(channels, 3, merged);
    cv::GaussianBlur(merged, merged, {3, 3}, 0.4);

    merged = cv::max(merged, 0.0);
    merged = cv::min(merged, 1.0);
    merged.convertTo(image, CV_8UC3, 255.0);

    spdlog::debug("Codebook-free SynthID removal complete: {} passes", num_passes);
}

cv::Mat NoiseResidualSubtractor::compute_dc_ramp(int rows, int cols, float radius) {
    cv::Mat ramp(rows, cols, CV_32FC1);
    for (int y = 0; y < rows; ++y) {
        float fy = static_cast<float>(y);
        if (fy > rows / 2.0f) fy -= rows;
        for (int x = 0; x < cols; ++x) {
            float fx = static_cast<float>(x);
            if (fx > cols / 2.0f) fx -= cols;
            float dist = std::sqrt(fy * fy + fx * fx);
            ramp.at<float>(y, x) = std::clamp(dist / radius, 0.0f, 1.0f);
        }
    }
    return ramp;
}

cv::Mat NoiseResidualSubtractor::estimate_carrier_from_noise(
    const cv::Mat& channel_fft,
    const cv::Mat& channel_float,
    float removal_factor,
    float mag_cap,
    float dc_radius,
    int channel_idx)
{
    const int rows = channel_fft.rows;
    const int cols = channel_fft.cols;
    const float ch_weight = kChannelWeights[channel_idx];

    // Extract noise residual via bilateral filter
    cv::Mat denoised;
    cv::bilateralFilter(channel_float, denoised, 9, 75.0, 75.0);
    cv::Mat noise = channel_float - denoised;

    // FFT of noise residual → carrier estimate
    cv::Mat noise_fft = fft_.forward(noise);
    cv::Mat noise_mag = FftContext::magnitude(noise_fft);
    cv::Mat noise_phase = FftContext::phase(noise_fft);

    // DC exclusion ramp
    cv::Mat dc_ramp = compute_dc_ramp(rows, cols, dc_radius);
    noise_mag = noise_mag.mul(dc_ramp);

    // Scale by removal factor and channel weight
    noise_mag *= removal_factor * ch_weight;

    // Safety cap: never subtract more than mag_cap * |image_fft|
    cv::Mat img_mag = FftContext::magnitude(channel_fft);
    cv::Mat cap;
    img_mag.copyTo(cap);
    cap *= mag_cap;
    cv::min(noise_mag, cap, noise_mag);

    // Construct complex watermark estimate using noise residual phase
    return FftContext::from_polar(noise_mag, noise_phase);
}

} // namespace wmr
