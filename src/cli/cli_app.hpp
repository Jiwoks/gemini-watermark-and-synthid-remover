#pragma once

#include <string>
#include <optional>
#include <utility>

#include "core/types.hpp"

namespace wmr {

enum class CliMode {
    AutoRemove,
    Detect,
    VisibleOnly,
    SynthidOnly,
    BuildCodebook,
    Video,
};

struct CliOptions {
    CliMode mode = CliMode::AutoRemove;
    std::string input_path;
    std::string output_path;
    bool force = false;
    bool force_small = false;
    bool force_large = false;
    bool verbose = false;
    bool detect_only = false;
    float inpaint_strength = 0.85f;
    bool synthid = false;
    std::string codebook_path;
    float synthid_strength = 0.50f;
    bool recursive = false;
    bool codebook_free = false;
    bool phase_adaptive = false;
    bool legacy_profile = false;       // video: Veo legacy text profile
    bool still_legacy = false;         // still images: pin legacy V1 (Gemini pre-3.5)
    bool still_no_legacy = false;      // still images: pin current V2, disable V2→V1 fallback
    std::string video_variant_str;
    bool scenes = false;
    double scene_threshold = 0.30;
    int video_crf = 14;
    std::string video_preset = "slow";
    std::string video_codec = "libx264";
};

// Resolve the still-image profile variant from CLI flags.
// Returns {force_variant, try_v1_fallback}:
//   --legacy      → {V1, false}
//   --no-legacy   → {V2, false}
//   (neither)     → {nullopt, true}  (default V2 with auto V2→V1 fallback)
std::pair<std::optional<WatermarkVariant>, bool>
resolve_still_variant(const CliOptions& opts);

int run_cli(int argc, char* argv[]);

} // namespace wmr
