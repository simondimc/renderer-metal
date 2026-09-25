#pragma once
#include <Metal/Metal.hpp>
#include <mutex>
#include <string>
#include <vector>

// Measures how long the GPU spends on each labelled pass of a frame, plus the whole frame's GPU time.
//
// Every render/blit pass is handed a pair of timestamp samples (start of its vertex work, end of its
// fragment work - the "stage boundary" sampling Apple GPUs support) through its pass descriptor. Once
// the frame's command buffer has finished, the samples are read back, converted to milliseconds and
// summed per label (so the six faces of a point light's shadow pass show up as one "Shadow" line).
// Displayed values are smoothed over a few frames. The frame total is the time the GPU actually spent on
// the frame: from when the previous frame let go of it (frames overlap when the GPU is the bottleneck)
// to this frame's last pass. Without pass sampling it falls back to the command buffer's own GPU
// start/end times, which also count any time spent waiting behind the previous frame.
//
// Apple GPUs are tile-based and overlap passes (one pass's vertex work starts long before the previous
// pass's fragment work ends), so a pass's own start-to-end span says little. Each pass is instead charged
// for the time between the previous pass finishing and itself finishing, which makes the per-pass numbers
// add up to the frame's GPU time. That is an attribution rather than an isolated measurement: good for
// finding what is expensive, and a pass that runs concurrently with another is charged to whichever
// finishes last.
class GpuTimer {
public:
    GpuTimer(MTL::Device* device, int framesInFlight);
    ~GpuTimer();
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    // True if per-pass timing works on this GPU (the frame total works regardless).
    bool passTimingSupported() const { return supported_; }

    // Call once per frame before any pass is created, with the frame-in-flight slot the frame uses.
    void beginFrame(int slot);

    // Make this render pass record timestamps, grouped under label. Call before creating its encoder.
    void attach(MTL::RenderPassDescriptor* descriptor, const char* label);

    // A blit pass descriptor that records timestamps under label, or nullptr if this GPU can't sample
    // blit passes (then use the plain blitCommandEncoder() and the pass simply isn't listed).
    MTL::BlitPassDescriptor* blitDescriptor(const char* label);

    // Call from the frame's command buffer completion handler, before the frame's slot is freed for
    // reuse: reads the samples back and updates the smoothed results. Runs on a Metal thread.
    void endFrame(int slot, MTL::CommandBuffer* commandBuffer);

    struct Entry {
        std::string label;
        float milliseconds;
    };
    // Smoothed results of the most recent frames: per pass (in no particular order) and the frame total.
    void snapshot(std::vector<Entry>& passes, float& totalMilliseconds) const;

private:
    struct PassRecord {
        std::string label;
        NS::UInteger firstSample; // the pass's start sample; its end sample is the next index
    };

    // Returns the pair's first sample index, or -1 when the sample buffer is full.
    int allocatePass(const char* label);

    static constexpr NS::UInteger kMaxPassesPerFrame = 128;

    MTL::Device* device_;
    bool supported_ = false;
    bool blitSupported_ = false;
    std::vector<MTL::CounterSampleBuffer*> sampleBuffers_; // one per frame in flight
    std::vector<std::vector<PassRecord>> passes_;          // likewise
    int currentSlot_ = 0;

    // The last GPU timestamp of the previous frame, in ticks - when this frame really got the GPU (0 = none yet).
    uint64_t previousFrameEndTick_ = 0;

    mutable std::mutex resultsMutex_;
    std::vector<Entry> smoothedPasses_;
    float smoothedTotal_ = 0.0f;
};
