#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <cstdint>

extern "C" {
struct AVFormatContext;
struct AVCodecContext;
struct SwsContext;
struct AVStream;
struct AVPacket;
struct AVFrame;
}

namespace wmr {

class VideoReader;

struct EncodeOptions {
    std::string codec = "libx264";
    int crf = 14;
    std::string preset = "slow";
    std::string profile = "high";
};

class VideoWriter {
public:
    VideoWriter();
    ~VideoWriter();

    // Non-copyable, non-movable
    VideoWriter(const VideoWriter&) = delete;
    VideoWriter& operator=(const VideoWriter&) = delete;

    bool open(const std::string& path, int width, int height, double fps,
              const EncodeOptions& opts = {});
    bool write_frame(const cv::Mat& frame);
    bool copy_audio_from(VideoReader& reader);
    void close();

private:
    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    AVStream* video_stream_ = nullptr;
    AVPacket* packet_ = nullptr;
    AVFrame* frame_ = nullptr;
    int64_t pts_counter_ = 0;
    double fps_ = 0.0;
    bool header_written_ = false;
    bool audio_copied_ = false;
};

} // namespace wmr
