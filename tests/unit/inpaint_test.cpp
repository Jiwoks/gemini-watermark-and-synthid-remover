#include <catch2/catch_test_macros.hpp>

#include "core/inpaint.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace wmr;

TEST_CASE("Gaussian inpaint produces valid output", "[inpaint]") {
    cv::Mat image(256, 256, CV_8UC3, cv::Scalar(100, 120, 140));

    // Draw a small white rectangle as "residual"
    cv::rectangle(image, {100, 100, 20, 20}, cv::Scalar(255, 255, 255), cv::FILLED);

    cv::Rect region(90, 90, 40, 40);
    cv::Mat alpha(40, 40, CV_32FC1, cv::Scalar(0.5f));

    cv::Mat result = image.clone();
    InpaintConfig config;
    config.method = InpaintMethod::Gaussian;
    config.strength = 0.85f;

    REQUIRE_NOTHROW(inpaint_residual(result, region, alpha, config));
    REQUIRE(result.rows == 256);
    REQUIRE(result.cols == 256);
    REQUIRE(result.type() == CV_8UC3);
}

TEST_CASE("Telea inpaint produces valid output", "[inpaint]") {
    cv::Mat image(256, 256, CV_8UC3, cv::Scalar(80, 100, 120));
    cv::rectangle(image, {110, 110, 10, 10}, cv::Scalar(255, 255, 255), cv::FILLED);

    cv::Rect region(100, 100, 30, 30);
    cv::Mat alpha(30, 30, CV_32FC1, cv::Scalar(0.3f));

    cv::Mat result = image.clone();
    InpaintConfig config;
    config.method = InpaintMethod::Telea;
    config.strength = 0.85f;

    REQUIRE_NOTHROW(inpaint_residual(result, region, alpha, config));
    REQUIRE(result.rows == 256);
    REQUIRE(result.cols == 256);
}

TEST_CASE("Navier-Stokes inpaint produces valid output", "[inpaint]") {
    cv::Mat image(256, 256, CV_8UC3, cv::Scalar(80, 100, 120));
    cv::rectangle(image, {110, 110, 10, 10}, cv::Scalar(255, 255, 255), cv::FILLED);

    cv::Rect region(100, 100, 30, 30);
    cv::Mat alpha(30, 30, CV_32FC1, cv::Scalar(0.3f));

    cv::Mat result = image.clone();
    InpaintConfig config;
    config.method = InpaintMethod::NavierStokes;
    config.strength = 0.85f;

    REQUIRE_NOTHROW(inpaint_residual(result, region, alpha, config));
    REQUIRE(result.rows == 256);
    REQUIRE(result.cols == 256);
}
