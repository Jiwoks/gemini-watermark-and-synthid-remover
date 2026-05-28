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
