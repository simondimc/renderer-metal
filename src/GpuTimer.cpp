#include "GpuTimer.hpp"
#include <algorithm>
#include <cstdint>

namespace {
// Fraction of each new frame blended into the displayed numbers: ~0.3 s of history at 60 fps.
constexpr float kSmoothing = 0.05f;
} // namespace

GpuTimer::GpuTimer(MTL::Device* device, int framesInFlight) : device_(device) {
    passes_.resize(framesInFlight);

    // Timestamps sampled at the start/end of a pass need the GPU to support stage-boundary sampling and
    // to offer the timestamp counter set. Not every GPU does; the frame total still works without.
    if (device_->supportsCounterSampling(MTL::CounterSamplingPointAtStageBoundary)) {
        MTL::CounterSet* timestampSet = nullptr;
        NS::Array* counterSets = device_->counterSets();
        for (NS::UInteger i = 0; counterSets && i < counterSets->count(); i++) {
            auto* candidate = static_cast<MTL::CounterSet*>(counterSets->object(i));
            if (candidate->name()->isEqualToString(MTL::CommonCounterSetTimestamp)) {
                timestampSet = candidate;
                break;
            }
        }
        if (timestampSet) {
            MTL::CounterSampleBufferDescriptor* descriptor = MTL::CounterSampleBufferDescriptor::alloc()->init();
            descriptor->setCounterSet(timestampSet);
            descriptor->setStorageMode(MTL::StorageModeShared);
            descriptor->setSampleCount(2 * kMaxPassesPerFrame);
            supported_ = true;
            for (int i = 0; i < framesInFlight; i++) {
                NS::Error* error = nullptr;
                MTL::CounterSampleBuffer* buffer = device_->newCounterSampleBuffer(descriptor, &error);
                if (!buffer) supported_ = false;
                sampleBuffers_.push_back(buffer);
            }
            descriptor->release();
        }
    }
    blitSupported_ = supported_ && device_->supportsCounterSampling(MTL::CounterSamplingPointAtBlitBoundary);
}

GpuTimer::~GpuTimer() {
    for (MTL::CounterSampleBuffer* buffer : sampleBuffers_) {
        if (buffer) buffer->release();
    }
}

void GpuTimer::beginFrame(int slot) {
    currentSlot_ = slot;
    passes_[slot].clear();
}

int GpuTimer::allocatePass(const char* label) {
    auto& records = passes_[currentSlot_];
    if (!supported_ || records.size() >= kMaxPassesPerFrame) return -1;
    NS::UInteger first = 2 * records.size();
    records.push_back({label, first});
    return (int)first;
}

void GpuTimer::attach(MTL::RenderPassDescriptor* descriptor, const char* label) {
    int first = allocatePass(label);
    if (first < 0) return;
    auto* attachment = descriptor->sampleBufferAttachments()->object(0);
    attachment->setSampleBuffer(sampleBuffers_[currentSlot_]);
    attachment->setStartOfVertexSampleIndex(first);
    attachment->setEndOfFragmentSampleIndex(first + 1);
    // Leave the other two stage boundaries unsampled (they must be told so explicitly).
    attachment->setStartOfFragmentSampleIndex(MTL::CounterDontSample);
    attachment->setEndOfVertexSampleIndex(MTL::CounterDontSample);
}

MTL::BlitPassDescriptor* GpuTimer::blitDescriptor(const char* label) {
    if (!blitSupported_) return nullptr;
    int first = allocatePass(label);
    if (first < 0) return nullptr;
    MTL::BlitPassDescriptor* descriptor = MTL::BlitPassDescriptor::blitPassDescriptor();
    auto* attachment = descriptor->sampleBufferAttachments()->object(0);
    attachment->setSampleBuffer(sampleBuffers_[currentSlot_]);
    attachment->setStartOfEncoderSampleIndex(first);
    attachment->setEndOfEncoderSampleIndex(first + 1);
    return descriptor;
}

void GpuTimer::endFrame(int slot, MTL::CommandBuffer* commandBuffer) {
    float totalMilliseconds = (float)((commandBuffer->GPUEndTime() - commandBuffer->GPUStartTime()) * 1000.0);

    std::vector<Entry> frame;
    const auto& records = passes_[slot];
    if (supported_ && !records.empty()) {
        NS::Data* data = sampleBuffers_[slot]->resolveCounterRange(NS::Range(0, 2 * records.size()));
        if (data && data->length() >= 2 * records.size() * sizeof(MTL::CounterResultTimestamp)) {
            const auto* stamps = static_cast<const MTL::CounterResultTimestamp*>(data->bytes());

            // The samples are in GPU ticks of an unknown, not-quite-constant length (the GPU clock can
            // pause while idle and change with power state), so a fixed tick-to-time factor would drift.
            // Instead the length is calibrated on every frame against the command buffer's own measured
            // GPU time: the first sample to the last one spans (almost exactly) the whole command buffer.
            uint64_t firstTick = UINT64_MAX, lastTick = 0;
            for (const PassRecord& record : records) {
                uint64_t start = stamps[record.firstSample].timestamp;
                uint64_t end = stamps[record.firstSample + 1].timestamp;
                if (start == MTL::CounterErrorValue || end == MTL::CounterErrorValue || end < start) continue;
                firstTick = std::min(firstTick, start);
                lastTick = std::max(lastTick, end);
            }
            double millisecondsPerTick = lastTick > firstTick ? (double)totalMilliseconds / (double)(lastTick - firstTick) : 0.0;

            // A pass's own start-to-end is not its cost: on a tile-based GPU its vertex work starts early,
            // long before the previous pass's fragment work is done, so those spans overlap heavily (their
            // sum can be several times the frame). Passes finish in the order they were encoded, though,
            // so each one is charged for the time from when the GPU was free (the previous pass's end, or
            // its own start if it was idle in between) to its own end - which adds up to the frame's span.
            uint64_t previousEnd = 0;
            for (const PassRecord& record : records) {
                uint64_t start = stamps[record.firstSample].timestamp;
                uint64_t end = stamps[record.firstSample + 1].timestamp;
                // A pass that recorded nothing (or was skipped by the GPU) reads back as an error value.
                if (start == MTL::CounterErrorValue || end == MTL::CounterErrorValue || end < start) continue;
                uint64_t chargedFrom = std::max(start, previousEnd);
                float milliseconds = end > chargedFrom ? (float)((double)(end - chargedFrom) * millisecondsPerTick) : 0.0f;
                previousEnd = std::max(previousEnd, end);
                bool merged = false;
                for (Entry& entry : frame) {
                    if (entry.label == record.label) {
                        entry.milliseconds += milliseconds;
                        merged = true;
                        break;
                    }
                }
                if (!merged) frame.push_back({record.label, milliseconds});
            }
        }
    }

    std::lock_guard<std::mutex> lock(resultsMutex_);
    smoothedTotal_ += (totalMilliseconds - smoothedTotal_) * kSmoothing;
    // Labels missing from this frame (a disabled effect) fade toward zero instead of holding a stale value.
    for (Entry& shown : smoothedPasses_) {
        float target = 0.0f;
        for (const Entry& entry : frame) {
            if (entry.label == shown.label) target = entry.milliseconds;
        }
        shown.milliseconds += (target - shown.milliseconds) * kSmoothing;
    }
    for (const Entry& entry : frame) {
        bool known = false;
        for (const Entry& shown : smoothedPasses_) {
            if (shown.label == entry.label) known = true;
        }
        if (!known) smoothedPasses_.push_back(entry);
    }
}

void GpuTimer::snapshot(std::vector<Entry>& passes, float& totalMilliseconds) const {
    std::lock_guard<std::mutex> lock(resultsMutex_);
    passes = smoothedPasses_;
    totalMilliseconds = smoothedTotal_;
}
