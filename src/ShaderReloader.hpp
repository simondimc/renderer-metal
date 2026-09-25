#pragma once
#include <Metal/Metal.hpp>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <string>

// Hot reloading for the Metal shader source. The build compiles Shader.metal into default.metallib once; this
// watches the source file itself and, whenever it is saved, compiles it again at run time (on a Metal
// worker thread, so the frame loop does not stall while the compiler runs). Main.cpp then rebuilds its pipeline
// states from the new library, and keeps drawing with the old ones until that has worked - a typo in the shader
// shows up as an error in the overlay and the terminal, and the last good shaders stay in use.
class ShaderReloader {
public:
    // sourcePath is relative to the working directory (build/, like the other asset paths). A missing file just
    // leaves the reloader disabled.
    ShaderReloader(MTL::Device* device, std::string sourcePath);
    ~ShaderReloader();
    ShaderReloader(const ShaderReloader&) = delete;
    ShaderReloader& operator=(const ShaderReloader&) = delete;

    enum class Status { Nothing, Compiled, Failed };

    // Call once per frame from the main thread. Checks the file a few times a second and starts a compile when it
    // has changed. Returns Compiled (library holds a new library - the caller owns it) or Failed (message holds the
    // compiler's output) once a compile has finished, and Nothing otherwise.
    Status poll(MTL::Library*& library, std::string& message);

private:
    MTL::Device* device_;
    std::string path_;
    std::filesystem::file_time_type lastWriteTime_;
    bool enabled_ = false;
    std::chrono::steady_clock::time_point lastCheck_;

    std::atomic<bool> compiling_{false};
    std::mutex resultMutex_;
    Status pendingStatus_ = Status::Nothing;
    MTL::Library* pendingLibrary_ = nullptr;
    std::string pendingMessage_;
};
