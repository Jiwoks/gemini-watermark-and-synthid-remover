#include "cli/cli_app.hpp"
#include "core/watermark_engine.hpp"
#include "core/types.hpp"
#include "core/fft_context.hpp"
#include "synthid/spectral_codebook.hpp"
#include "synthid/codebook_subtractor.hpp"

#include <opencv2/imgcodecs.hpp>
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <filesystem>

#ifndef APP_VERSION
#define APP_VERSION "0.1.0"
#endif

#ifndef APP_NAME
#define APP_NAME "wmr"
#endif

namespace wmr {

int run_cli(int argc, char* argv[]) {
    CLI::App app{"Watermark Remover — remove Gemini visible and SynthID invisible watermarks", APP_NAME};
    app.set_version_flag("-V,--version", APP_VERSION);

    CliOptions opts;

    app.add_option("input", opts.input_path, "Input image file")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("-o,--output", opts.output_path, "Output file (default: overwrite in-place)");

    app.add_flag("-f,--force", opts.force, "Skip detection, force-remove at default position");
    app.add_flag("--force-small", opts.force_small, "Force 48x48 watermark");
    app.add_flag("--force-large", opts.force_large, "Force 96x96 watermark");
    app.add_flag("-v,--verbose", opts.verbose, "Verbose output");
    app.add_flag("--detect-only", opts.detect_only, "Report detection result without modifying");
    app.add_option("--inpaint-strength", opts.inpaint_strength,
                   "Inpaint strength 0.0-1.0 (default: 0.85)")
        ->check(CLI::Range(0.0f, 1.0f));

    app.add_flag("--synthid", opts.synthid, "Remove SynthID invisible watermark via spectral subtraction");
    app.add_option("--codebook", opts.codebook_path, "Path to spectral codebook file (.wcb)");
    app.add_option("--synthid-strength", opts.synthid_strength,
                   "SynthID removal strength 0.0-2.0 (default: 1.0)")
        ->check(CLI::Range(0.0f, 2.0f));

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    if (opts.verbose) {
        spdlog::set_level(spdlog::level::debug);
    } else {
        spdlog::set_level(spdlog::level::info);
    }

    if (opts.force_small && opts.force_large) {
        spdlog::error("Cannot use both --force-small and --force-large");
        return 1;
    }

    if (opts.synthid && opts.codebook_path.empty()) {
        spdlog::error("--synthid requires --codebook <path> to a spectral codebook file");
        return 1;
    }

    try {
        cv::Mat image = cv::imread(opts.input_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            spdlog::error("Failed to load image: {}", opts.input_path);
            return 1;
        }

        spdlog::info("Loading: {}", opts.input_path);
        spdlog::info("Image: {}x{}", image.cols, image.rows);

        // --- Detect-only mode ---
        if (opts.detect_only) {
            WatermarkEngine engine;
            auto result = engine.detect_watermark(image);
            if (result.detected) {
                spdlog::info("Watermark DETECTED (confidence: {:.1f}%)",
                             result.confidence * 100.0f);
                spdlog::info("  Region: ({}, {}) {}x{}",
                             result.region.x, result.region.y,
                             result.region.width, result.region.height);
                spdlog::info("  Scores: spatial={:.3f} grad={:.3f} var={:.3f}",
                             result.spatial_score, result.gradient_score,
                             result.variance_score);
            } else {
                spdlog::info("No watermark detected (confidence: {:.1f}%)",
                             result.confidence * 100.0f);
            }
            return result.detected ? 0 : 2;
        }

        // --- Phase 1: Visible watermark removal ---
        WatermarkEngine engine;
        std::optional<WatermarkSize> force_size;
        if (opts.force_small) {
            force_size = WatermarkSize::Small;
        } else if (opts.force_large) {
            force_size = WatermarkSize::Large;
        }

        if (!opts.force) {
            auto detection = engine.detect_watermark(image, force_size);

            if (detection.detected) {
                spdlog::info("Watermark detected ({:.1f}% confidence), removing...",
                             detection.confidence * 100.0f);
                engine.remove_watermark_detected(image, detection);
            } else {
                spdlog::warn("No watermark detected ({:.1f}% confidence). "
                             "Use --force to remove anyway.",
                             detection.confidence * 100.0f);
                // Continue to SynthID removal even if no visible watermark found
                if (!opts.synthid) {
                    return 2;
                }
            }
        } else {
            spdlog::info("Force mode: removing watermark at default position");
            engine.remove_watermark(image, force_size);
        }

        // --- Phase 2: SynthID invisible watermark removal ---
        if (opts.synthid) {
            spdlog::info("Removing SynthID watermark via spectral subtraction...");

            FftContext fft;
            SpectralCodebook codebook;
            codebook.load(opts.codebook_path);

            CodebookSubtractor subtractor(fft);
            RemovalConfig config;
            config.custom_strength = opts.synthid_strength;

            subtractor.remove_synthid(image, codebook, config);
            spdlog::info("SynthID removal complete");
        }

        // --- Save output ---
        std::string output = opts.output_path.empty() ? opts.input_path : opts.output_path;
        std::filesystem::path out_path(output);
        if (!out_path.parent_path().empty() && !std::filesystem::exists(out_path.parent_path())) {
            std::filesystem::create_directories(out_path.parent_path());
        }

        std::vector<int> params;
        std::string ext = out_path.extension().string();
        if (ext == ".jpg" || ext == ".jpeg") {
            params = {cv::IMWRITE_JPEG_QUALITY, 100};
        } else if (ext == ".png") {
            params = {cv::IMWRITE_PNG_COMPRESSION, 6};
        } else if (ext == ".webp") {
            params = {cv::IMWRITE_WEBP_QUALITY, 101};
        }

        if (!cv::imwrite(output, image, params)) {
            spdlog::error("Failed to save: {}", output);
            return 1;
        }

        spdlog::info("Saved: {}", output);
        return 0;

    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }
}

} // namespace wmr
