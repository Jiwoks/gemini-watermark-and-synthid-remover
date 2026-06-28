#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/watermark_engine.hpp"
#include "core/types.hpp"
#include "core/blend_modes.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>

using namespace wmr;
using Catch::Matchers::WithinAbs;

static const char* kTestImage = "test-images/2400x1792-gemini.png";
static const char* kAltTestImage = "../test-images/2400x1792-gemini.png";

static cv::Mat load_test_image() {
    if (std::filesystem::exists(kTestImage)) {
        return cv::imread(kTestImage, cv::IMREAD_COLOR);
    }
    if (std::filesystem::exists(kAltTestImage)) {
        return cv::imread(kAltTestImage, cv::IMREAD_COLOR);
    }
    return {};
}

TEST_CASE("Visible watermark detection on known watermarked image", "[integration]") {
    cv::Mat image = load_test_image();
    if (image.empty()) {
        SKIP("Test image not available");
    }

    WatermarkEngine engine;
    auto result = engine.detect_watermark(image);

    REQUIRE(result.detected);
    REQUIRE(result.confidence > 0.5f);
    REQUIRE(result.region.width > 0);
    REQUIRE(result.region.height > 0);
}

TEST_CASE("Full visible pipeline: detect → remove → inpaint", "[integration]") {
    cv::Mat image = load_test_image();
    if (image.empty()) {
        SKIP("Test image not available");
    }

    cv::Mat original = image.clone();
    WatermarkEngine engine;

    auto detection = engine.detect_watermark(image);
    REQUIRE(detection.detected);

    engine.remove_watermark_detected(image, detection);

    // The bottom-right region should have changed
    cv::Rect roi = detection.region;
    cv::Mat orig_roi = original(roi);
    cv::Mat new_roi = image(roi);

    cv::Mat diff;
    cv::absdiff(orig_roi, new_roi, diff);
    double total_diff = cv::sum(diff.reshape(1))[0] + cv::sum(diff.reshape(1))[1] + cv::sum(diff.reshape(1))[2];
    REQUIRE(total_diff > 0);

    // Output should still be valid
    REQUIRE(image.rows == original.rows);
    REQUIRE(image.cols == original.cols);
    REQUIRE(image.type() == CV_8UC3);
}

TEST_CASE("Detection on clean image returns not detected", "[integration]") {
    // Create a clean gradient image (no watermark)
    cv::Mat clean(1024, 1024, CV_8UC3);
    for (int y = 0; y < 1024; ++y) {
        for (int x = 0; x < 1024; ++x) {
            clean.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>(x * 255 / 1024),
                static_cast<uchar>(y * 255 / 1024),
                128
            );
        }
    }

    WatermarkEngine engine;
    auto result = engine.detect_watermark(clean);

    REQUIRE_FALSE(result.detected);
}

// ---------------------------------------------------------------------------
// V2 (Gemini 3.5) watermark geometry — pure geometry, no fixtures.
// ---------------------------------------------------------------------------
TEST_CASE("V2 watermark geometry is variant-aware", "[v2]") {
    auto eq = [](const WatermarkPosition& p, int mr, int mb, int ls) {
        return p.margin_right == mr && p.margin_bottom == mb && p.logo_size == ls;
    };

    // V1 (legacy, pre-3.5) unchanged
    REQUIRE(eq(get_watermark_config(2400, 1792, WatermarkVariant::V1), 64, 64, 96));
    REQUIRE(eq(get_watermark_config(1024, 1024, WatermarkVariant::V1), 32, 32, 48));

    // V2 large (both dims > 1024): 192px margin, 96px logo
    REQUIRE(eq(get_watermark_config(2400, 1792, WatermarkVariant::V2), 192, 192, 96));
    REQUIRE(eq(get_watermark_config(2048, 2048, WatermarkVariant::V2), 192, 192, 96));

    // V2 small (a side <= 1024): 36px logo, aspect-aware margin.
    // 1024x1024: short=1024 >= 566 -> source 2752; margin = round(192*1024/2752) = 71
    REQUIRE(eq(get_watermark_config(1024, 1024, WatermarkVariant::V2), 71, 71, 36));
    // 1280x720: short=720 >= 566 -> source 2752; margin = round(192*1280/2752) = 89
    REQUIRE(eq(get_watermark_config(1280, 720, WatermarkVariant::V2), 89, 89, 36));

    // 2-arg overload defaults to V1 (backward compatibility)
    REQUIRE(eq(get_watermark_config(1024, 1024), 32, 32, 48));
}

// ---------------------------------------------------------------------------
// V2 (Gemini 3.5) engine path — detect/remove with the V2 alpha maps.
// Synthetic round-trips (forward-blend the V2 alpha, then detect + remove).
// ---------------------------------------------------------------------------
static cv::Mat textured(int W, int H, cv::Scalar base) {
    cv::Mat img(H, W, CV_8UC3, base);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int gx = x * 60 / W;
            int gy = y * 60 / H;
            // Moderate, varied background: a smooth gradient plus low-amplitude
            // medium-frequency variation, so the gradient/variance stages have
            // signal without drowning the spatial NCC (a pure gradient zeroes the
            // gradient stage; full-range noise drowns the spatial stage).
            int n = ((x * 5 ^ y * 3) & 0x1F) - 0x10;  // ±16
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                cv::saturate_cast<uchar>(base[0] + gx + n),
                cv::saturate_cast<uchar>(base[1] + gy + n),
                cv::saturate_cast<uchar>(base[2] + gx));
        }
    return img;
}

static double mean_abs_diff(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat d;
    cv::absdiff(a, b, d);
    cv::Scalar s = cv::mean(d);
    return (s[0] + s[1] + s[2]) / 3.0;
}

TEST_CASE("V2 large round-trip recovers original", "[v2]") {
    cv::Mat original = textured(2048, 2048, cv::Scalar(80, 100, 120));
    cv::Mat watermarked = original.clone();

    WatermarkEngine engine;
    const auto pos_cfg = get_watermark_config(2048, 2048, WatermarkVariant::V2);  // {192,192,96}
    const cv::Point pos = pos_cfg.get_position(2048, 2048);
    const cv::Mat& alpha = engine.get_v2_diamond_alpha_large();
    REQUIRE(alpha.cols == 96);  // V2 large map decoded
    add_watermark_alpha_blend(watermarked, alpha, pos, 255.0f);

    const cv::Rect roi(pos.x, pos.y, alpha.cols, alpha.rows);
    REQUIRE(mean_abs_diff(original(roi), watermarked(roi)) > 5.0);  // watermark applied

    auto det = engine.detect_watermark(watermarked, std::nullopt, std::nullopt,
                                       nullptr, WatermarkVariant::V2, /*enable_snap=*/false);
    CAPTURE(det.spatial_score, det.gradient_score, det.variance_score, det.confidence);
    REQUIRE(det.detected);
    REQUIRE(det.confidence > 0.4f);

    engine.remove_watermark_alpha_only(watermarked, det, &alpha);
    REQUIRE(mean_abs_diff(original(roi), watermarked(roi)) < 1.5);  // near-exact recovery
}

TEST_CASE("V2 small detection uses 36x36 alpha + snap", "[v2]") {
    cv::Mat original = textured(1024, 1024, cv::Scalar(60, 90, 130));
    cv::Mat watermarked = original.clone();

    WatermarkEngine engine;
    const auto pos_cfg = get_watermark_config(1024, 1024, WatermarkVariant::V2);  // {71,71,36}
    const cv::Point pos = pos_cfg.get_position(1024, 1024);
    const cv::Mat& alpha = engine.get_v2_diamond_alpha_36();
    REQUIRE(alpha.cols == 36);
    REQUIRE(alpha.rows == 36);
    add_watermark_alpha_blend(watermarked, alpha, pos, 255.0f);

    auto det = engine.detect_watermark(watermarked, std::nullopt, std::nullopt,
                                       nullptr, WatermarkVariant::V2, /*enable_snap=*/true);
    CAPTURE(det.spatial_score, det.gradient_score, det.variance_score, det.confidence);
    REQUIRE(det.detected);
    REQUIRE(det.region.width == 36);
    REQUIRE(det.region.height == 36);
}

TEST_CASE("V1 legacy path still works via default detect", "[v2]") {
    cv::Mat original = textured(1024, 1024, cv::Scalar(70, 80, 90));
    cv::Mat watermarked = original.clone();

    WatermarkEngine engine;
    const auto pos_cfg = get_watermark_config(1024, 1024, WatermarkVariant::V1);  // {32,32,48}
    const cv::Point pos = pos_cfg.get_position(1024, 1024);
    const cv::Mat& alpha = engine.get_alpha_map(WatermarkSize::Small);  // V1 48x48
    add_watermark_alpha_blend(watermarked, alpha, pos, 255.0f);

    // The 4-arg (default) detect resolves to V1 — legacy callers unaffected.
    auto det = engine.detect_watermark(watermarked);
    REQUIRE(det.detected);
    REQUIRE(det.region.width == 48);
}
