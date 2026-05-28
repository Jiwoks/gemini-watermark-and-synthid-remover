#include "core/inpaint.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace wmr {

void inpaint_residual(
    cv::Mat& image,
    const cv::Rect& region,
    const cv::Mat& alpha_map,
    const InpaintConfig& config)
{
    if (image.empty() || region.width < 4 || region.height < 4) return;

    const float strength = std::clamp(config.strength, 0.0f, 1.0f);
    if (strength < 0.001f) return;

    // Padded region for context
    cv::Rect padded(
        region.x - config.padding,
        region.y - config.padding,
        region.width + config.padding * 2,
        region.height + config.padding * 2);
    padded &= cv::Rect(0, 0, image.cols, image.rows);

    if (padded.width < 8 || padded.height < 8) return;

    cv::Rect inner(
        region.x - padded.x,
        region.y - padded.y,
        region.width,
        region.height);
    inner &= cv::Rect(0, 0, padded.width, padded.height);

    // Compute alpha gradient for edge detection
    cv::Mat alpha_resized;
    int interp = (region.width > alpha_map.cols)
        ? cv::INTER_LINEAR : cv::INTER_AREA;
    cv::resize(alpha_map, alpha_resized,
               cv::Size(region.width, region.height), 0, 0, interp);

    cv::Mat grad_x, grad_y, grad_mag;
    cv::Sobel(alpha_resized, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(alpha_resized, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, grad_mag);

    double grad_min, grad_max;
    cv::minMaxLoc(grad_mag, &grad_min, &grad_max);
    if (grad_max <= grad_min) {
        spdlog::debug("inpaint: flat gradient, no edges found");
        return;
    }

    if (config.method == InpaintMethod::Gaussian) {
        // --- Gaussian Soft Inpaint ---
        cv::Mat grad_norm = (grad_mag - grad_min) / (grad_max - grad_min);
        cv::Mat grad_weight;
        cv::sqrt(grad_norm, grad_weight);

        // Dilate to cover residual spread
        cv::Mat dk = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::dilate(grad_weight, grad_weight, dk);
        cv::GaussianBlur(grad_weight, grad_weight, cv::Size(0, 0), 2.0);
        grad_weight *= strength;
        cv::threshold(grad_weight, grad_weight, 1.0, 1.0, cv::THRESH_TRUNC);

        // Embed into padded coordinate system
        cv::Mat weight = cv::Mat::zeros(padded.size(), CV_32F);
        grad_weight.copyTo(weight(inner));
        cv::GaussianBlur(weight, weight, cv::Size(0, 0), 1.0);

        // Gaussian blur the image
        int ksize = config.radius * 2 + 1;
        if (ksize % 2 == 0) ksize++;
        ksize = std::max(ksize, 3);
        double sigma = config.radius * 0.8;

        cv::Mat padded_area = image(padded).clone();
        cv::Mat blurred;
        cv::GaussianBlur(padded_area, blurred, cv::Size(ksize, ksize), sigma);

        // Per-pixel weighted blend
        cv::Mat dst = image(padded);
        cv::Mat weight_3ch;
        cv::merge(std::vector<cv::Mat>{weight, weight, weight}, weight_3ch);

        cv::Mat dst_f, blurred_f, result_f;
        dst.convertTo(dst_f, CV_32FC3);
        blurred.convertTo(blurred_f, CV_32FC3);

        cv::Mat one_minus_w = cv::Scalar(1.0, 1.0, 1.0) - weight_3ch;
        cv::multiply(dst_f, one_minus_w, dst_f);
        cv::multiply(blurred_f, weight_3ch, blurred_f);
        result_f = dst_f + blurred_f;

        result_f.convertTo(dst, CV_8UC3);

        int active = cv::countNonZero(weight > 0.01f);
        spdlog::debug("inpaint: Gaussian, strength={:.0f}%, {} active pixels",
                      strength * 100.0f, active);
    } else {
        // --- TELEA / Navier-Stokes ---
        cv::Mat grad_u8;
        grad_mag.convertTo(grad_u8, CV_8U,
                           255.0 / (grad_max - grad_min),
                           -grad_min * 255.0 / (grad_max - grad_min));

        cv::Mat sparse_mask;
        cv::threshold(grad_u8, sparse_mask, 20, 255, cv::THRESH_BINARY);

        cv::Mat dilate_kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::dilate(sparse_mask, sparse_mask, dilate_kernel);

        int masked = cv::countNonZero(sparse_mask);
        if (masked == 0) {
            spdlog::debug("inpaint: no edge pixels found, skipping");
            return;
        }

        // Embed mask into padded coordinate system
        cv::Mat mask = cv::Mat::zeros(padded.size(), CV_8UC1);
        sparse_mask.copyTo(mask(inner));

        cv::Mat padded_area = image(padded).clone();
        int cv_method = (config.method == InpaintMethod::Telea)
            ? cv::INPAINT_TELEA : cv::INPAINT_NS;

        cv::Mat inpainted;
        cv::inpaint(padded_area, mask, inpainted, config.radius, cv_method);

        // Blend at masked pixels only
        cv::Mat dst = image(padded);
        cv::Mat src_inner = dst(inner);
        cv::Mat inp_inner = inpainted(inner);
        cv::Mat mask_inner = mask(inner);

        if (strength >= 0.999f) {
            inp_inner.copyTo(src_inner, mask_inner);
        } else {
            cv::Mat blended;
            cv::addWeighted(src_inner, 1.0 - strength,
                            inp_inner, strength, 0.0, blended);
            blended.copyTo(src_inner, mask_inner);
        }

        const char* name = (config.method == InpaintMethod::Telea)
            ? "TELEA" : "Navier-Stokes";
        spdlog::debug("inpaint: {}, {} pixels repaired at {:.0f}% strength",
                      name, masked, strength * 100.0f);
    }
}

} // namespace wmr
