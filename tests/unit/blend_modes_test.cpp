#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/blend_modes.hpp"
#include "core/types.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace wmr;
using Catch::Matchers::WithinAbs;

TEST_CASE("Alpha map calculation from white watermark template", "[blend]") {
    cv::Mat template_img(48, 48, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat alpha = calculate_alpha_map(template_img);

    REQUIRE(alpha.rows == 48);
    REQUIRE(alpha.cols == 48);
    REQUIRE(alpha.type() == CV_32FC1);

    double min_val, max_val;
    cv::minMaxLoc(alpha, &min_val, &max_val);
    REQUIRE_THAT(max_val, WithinAbs(1.0, 0.01));
}

TEST_CASE("Alpha map from black template is zero", "[blend]") {
    cv::Mat template_img(48, 48, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat alpha = calculate_alpha_map(template_img);

    double sum = cv::sum(alpha)[0];
    REQUIRE(sum < 0.01);
}

TEST_CASE("Forward then reverse alpha blend recovers background", "[blend]") {
    // Create a known background in the watermark region
    cv::Mat background(96, 96, CV_8UC3, cv::Scalar(100, 150, 200));

    // Create alpha map with mid-range values (avoid near-1.0 which amplifies error)
    cv::Mat alpha(96, 96, CV_32FC1);
    for (int y = 0; y < 96; ++y) {
        for (int x = 0; x < 96; ++x) {
            alpha.at<float>(y, x) = std::clamp(static_cast<float>(y) / 95.0f, 0.05f, 0.85f);
        }
    }

    // Save original
    cv::Mat original = background.clone();

    // Forward blend (adds white watermark using alpha)
    cv::Point pos(0, 0);
    add_watermark_alpha_blend(background, alpha, pos);

    // The background should have changed
    cv::Mat diff_forward;
    cv::absdiff(original, background, diff_forward);
    double total_forward = cv::sum(diff_forward.reshape(1))[0];
    REQUIRE(total_forward > 0);

    // Reverse blend (removes watermark using same alpha)
    remove_watermark_alpha_blend(background, alpha, pos);

    // Should recover close to original (quantization + clamp introduces some error)
    cv::Mat diff_recovered;
    cv::absdiff(original, background, diff_recovered);
    double max_diff = 0;
    cv::minMaxLoc(diff_recovered.reshape(1), nullptr, &max_diff);
    REQUIRE(max_diff < 10);
}
