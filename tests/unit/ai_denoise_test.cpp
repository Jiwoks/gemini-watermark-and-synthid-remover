#ifdef WMR_AI_DENOISE

#include <catch2/catch_test_macros.hpp>

#include "core/ai_denoise.hpp"
#include "core/blend_modes.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>

using namespace wmr;

namespace {

// Thin wrapper so REQUIRE_NOTHROW has a single callable to invoke.
void denoise_impl_smoke(
    NcnnDenoiser& denoiser,
    cv::Mat& image,
    const cv::Rect& region,
    const cv::Mat& alpha_map)
{
    denoiser.denoise(image, region, alpha_map, /*sigma=*/25.0f, /*strength=*/0.85f, /*padding=*/16);
}

} // namespace

// ============================================================================
// Model load / runtime init
// ============================================================================

TEST_CASE("NcnnDenoiser loads the embedded FDnCNN model", "[aidenoise]") {
    NcnnDenoiser denoiser;

    REQUIRE(denoiser.initialize());
    REQUIRE(denoiser.is_ready());

    // The runtime must report either a GPU device name or a CPU thread count.
    const std::string device = denoiser.device_name();
    INFO("device_name(): " << device);
    REQUIRE(!device.empty());

    // Log which path was taken so CI runs surface CPU-vs-GPU in the output.
    INFO("is_gpu_enabled(): " << (denoiser.is_gpu_enabled() ? "GPU" : "CPU"));
}

// ============================================================================
// Denoise smoke test
// ============================================================================

TEST_CASE("NcnnDenoiser runs inference on a forward-blended ROI", "[aidenoise]") {
    NcnnDenoiser denoiser;
    REQUIRE(denoiser.initialize());
    REQUIRE(denoiser.is_ready());

    // Synthetic image: smooth gradient so the model has real content to denoise.
    cv::Mat image(96, 96, CV_8UC3);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            const cv::Vec3b px(
                static_cast<uchar>((x * 255) / image.cols),
                static_cast<uchar>((y * 255) / image.rows),
                static_cast<uchar>(((x + y) * 255) / (image.cols + image.rows)));
            image.at<cv::Vec3b>(y, x) = px;
        }
    }

    // Known alpha map (CV_32FC1, [0,1]) over a 48x48 watermark region.
    const cv::Rect region(40, 40, 48, 48);
    cv::Mat alpha_map(region.size(), CV_32FC1);
    for (int y = 0; y < alpha_map.rows; ++y) {
        for (int x = 0; x < alpha_map.cols; ++x) {
            // Soft disk: high in the centre, falling off at the edges — this
            // produces a non-trivial gradient the denoiser's edge mask can lock
            // onto, rather than a flat alpha which has zero gradient.
            const float dx = (x - alpha_map.cols * 0.5f) / (alpha_map.cols * 0.5f);
            const float dy = (y - alpha_map.rows * 0.5f) / (alpha_map.rows * 0.5f);
            const float r = std::sqrt(dx * dx + dy * dy);
            alpha_map.at<float>(y, x) = std::max(0.0f, 0.8f * (1.0f - r));
        }
    }

    // Forward-blend a watermark onto the region (this is what reverse-blend later
    // leaves residuals on), so the denoise call has a realistic input.
    add_watermark_alpha_blend(image, alpha_map, region.tl(), 255.0f);

    cv::Mat blended = image.clone();

    // Must run to completion without throwing or crashing.
    REQUIRE_NOTHROW(denoise_impl_smoke(denoiser, blended, region, alpha_map));

    // The call modified the image in-place; just assert it remains a valid
    // BGR CV_8UC3 of the same geometry. We deliberately do not over-assert
    // pixel values — the smoke test only proves the path is wired up.
    REQUIRE(blended.type() == CV_8UC3);
    REQUIRE(blended.rows == image.rows);
    REQUIRE(blended.cols == image.cols);
}

#endif // WMR_AI_DENOISE
