#include "ShaderReloader.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <thread>

namespace {
constexpr auto kPollInterval = std::chrono::milliseconds(250);
} // namespace

ShaderReloader::ShaderReloader(MTL::Device* device, std::string sourcePath)
    : device_(device), path_(std::move(sourcePath)), lastCheck_(std::chrono::steady_clock::now()) {
    std::error_code error;
    lastWriteTime_ = std::filesystem::last_write_time(path_, error);
    enabled_ = !error;
    if (!enabled_) fprintf(stderr, "Shader hot reloading off: cannot read %s\n", path_.c_str());
}

ShaderReloader::~ShaderReloader() {
    // The compile's completion handler points back at this object, so let a running compile finish first.
    while (compiling_) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (pendingLibrary_) pendingLibrary_->release();
}

ShaderReloader::Status ShaderReloader::poll(MTL::Library*& library, std::string& message) {
    if (!enabled_) return Status::Nothing;

    auto now = std::chrono::steady_clock::now();
    if (!compiling_ && now - lastCheck_ >= kPollInterval) {
        lastCheck_ = now;
        std::error_code error;
        auto writeTime = std::filesystem::last_write_time(path_, error);
        // The time is remembered before the file is read, so a save that lands while this compile runs is
        // seen as a further change and compiled next.
        if (!error && writeTime != lastWriteTime_) {
            lastWriteTime_ = writeTime;
            std::ifstream file(path_);
            std::stringstream source;
            source << file.rdbuf();

            compiling_ = true;
            NS::String* text = NS::String::string(source.str().c_str(), NS::UTF8StringEncoding);
            MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
            device_->newLibrary(text, options, [this](MTL::Library* compiled, NS::Error* compileError) {
                std::lock_guard<std::mutex> lock(resultMutex_);
                if (compiled) {
                    compiled->retain();
                    if (pendingLibrary_) pendingLibrary_->release(); // an unread earlier result is stale
                    pendingLibrary_ = compiled;
                    pendingStatus_ = Status::Compiled;
                    pendingMessage_.clear();
                } else {
                    pendingStatus_ = Status::Failed;
                    pendingMessage_ = compileError ? compileError->localizedDescription()->utf8String() : "unknown compile error";
                }
                compiling_ = false;
            });
            options->release();
        }
    }

    std::lock_guard<std::mutex> lock(resultMutex_);
    Status status = pendingStatus_;
    if (status == Status::Compiled) {
        library = pendingLibrary_;
        pendingLibrary_ = nullptr; // ownership passes to the caller
    } else if (status == Status::Failed) {
        message = pendingMessage_;
    }
    pendingStatus_ = Status::Nothing;
    return status;
}
