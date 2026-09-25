#include "LightCulling.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {
// Radiance below which a light is treated as not reaching a point (linear HDR, before exposure).
constexpr float kLightCutoff = 0.005f;
// No light reaches further than this: past the camera's far plane there is nothing to light anyway.
constexpr float kMaxLightRadius = 150.0f;
constexpr float kCameraNear = 0.1f;   // must match CAMERA_NEAR_PLANE / computeProjection
constexpr float kCameraFar = 100.0f;  // must match CAMERA_FAR_PLANE / computeProjection
constexpr float kPi = 3.14159265358979323846f;

// The scene shader's falloff (see fragmentMain).
float falloff(float distance) {
    return 1.0f / (1.0f + 0.09f * distance + 0.032f * distance * distance);
}

// Radius where a light with `peak` radiance falls to kLightCutoff, i.e. falloff(r) = kLightCutoff / peak,
// solved for r in 0.032 r^2 + 0.09 r + 1 - 1/falloff = 0.
float radiusForPeak(float peak) {
    float target = kLightCutoff / peak;
    if (target >= 1.0f) return 0.0f;
    float radius = (-0.09f + sqrtf(0.0081f + 0.128f * (1.0f / target - 1.0f))) / 0.064f;
    return std::min(radius, kMaxLightRadius);
}
} // namespace

float lightCullRadius(const SceneLight& light) {
    float peak = kPi * light.intensity * std::max({light.color.x, light.color.y, light.color.z});
    return peak > 0.0f ? radiusForPeak(peak) : 0.0f;
}

LightCuller::LightCuller(MTL::Device* device, MTL::Library* library, int framesInFlight) : device_(device) {
    MTL::Function* function = library->newFunction(NS::String::string("lightCullKernel", NS::UTF8StringEncoding));
    NS::Error* error = nullptr;
    cullPipeline_ = device_->newComputePipelineState(function, &error);
    if (!cullPipeline_) fprintf(stderr, "Failed to create the light culling pipeline\n");
    function->release();

    for (int i = 0; i < framesInFlight; i++) {
        lightBuffers_.push_back(device_->newBuffer(kMaxClusteredLights * sizeof(GPULight), MTL::ResourceStorageModeShared));
    }
}

LightCuller::~LightCuller() {
    if (cullPipeline_) cullPipeline_->release();
    for (MTL::Buffer* buffer : lightBuffers_) buffer->release();
    if (clusterCounts_) clusterCounts_->release();
    if (clusterIndices_) clusterIndices_->release();
}

void LightCuller::ensureClusterBuffers(uint32_t clusters) {
    if (clusters <= clusterCapacity_) return;
    if (clusterCounts_) clusterCounts_->release();
    if (clusterIndices_) clusterIndices_->release();
    clusterCounts_ = device_->newBuffer(clusters * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    clusterIndices_ = device_->newBuffer((size_t)clusters * kMaxLightsPerCluster * sizeof(uint16_t), MTL::ResourceStorageModePrivate);
    clusterCapacity_ = clusters;
}

void LightCuller::prepare(int slot, const SceneLight* lights, int lightCount, const Camera& camera, int width, int height) {
    slot_ = slot;

    // Directional lights first (they are shaded everywhere, no culling), then every other light. Each keeps the
    // shadow-map slot it has by scene order, which the shader uses to find its shadow map.
    GPULight* gpuLights = static_cast<GPULight*>(lightBuffers_[slot]->contents());
    uint32_t directionalCount = 0, clusteredCount = 0;
    std::vector<GPULight> clustered;
    for (int i = 0; i < lightCount; i++) {
        const SceneLight& light = lights[i];
        float peak = kPi * light.intensity * std::max({light.color.x, light.color.y, light.color.z});
        if (peak <= 0.0f) continue; // a black light adds nothing

        GPULight gpu;
        gpu.directionType = simd_make_float4(light.direction.x, light.direction.y, light.direction.z, (float)(int)light.type);
        gpu.colorIntensity = simd_make_float4(light.color.x, light.color.y, light.color.z, light.intensity);
        gpu.params = simd_make_float4(light.spotCosInner, light.spotCosOuter, light.areaHalfSize.x, light.areaHalfSize.y);
        bool castsShadow = i < (int)kMaxLights && light.type != LightType::Area;
        gpu.right = simd_make_float4(light.right.x, light.right.y, light.right.z, castsShadow ? (float)i : -1.0f);
        gpu.up = simd_make_float4(light.up.x, light.up.y, light.up.z, 0.0f);

        if (light.type == LightType::Directional) {
            gpu.positionRange = simd_make_float4(light.position.x, light.position.y, light.position.z, 0.0f);
            if (directionalCount + clusteredCount < kMaxClusteredLights) {
                gpuLights[directionalCount++] = gpu;
            }
        } else {
            float radius = radiusForPeak(peak);
            gpu.up.w = falloff(radius);
            // An area light lights everything within `radius` of its rectangle: a sphere around its centre needs
            // the rectangle's half-diagonal on top.
            float sphere = radius + (light.type == LightType::Area ? simd_length(light.areaHalfSize) : 0.0f);
            gpu.positionRange = simd_make_float4(light.position.x, light.position.y, light.position.z, sphere);
            clustered.push_back(gpu);
        }
    }
    for (const GPULight& gpu : clustered) {
        if (directionalCount + clusteredCount >= kMaxClusteredLights) break;
        gpuLights[directionalCount + clusteredCount++] = gpu;
    }

    float sliceScale = (float)kClusterSlices / logf(kCameraFar / kCameraNear);
    simd::float4x4 projection = computeProjection(width, height);
    params_.view = viewMatrix(camera);
    params_.projection = simd_make_float4(projection.columns[0][0], projection.columns[1][1], kCameraNear, kCameraFar);
    params_.grid = simd_make_uint4((width + kClusterTilePixels - 1) / kClusterTilePixels,
                                   (height + kClusterTilePixels - 1) / kClusterTilePixels,
                                   kClusterSlices, kClusterTilePixels);
    params_.lights = simd_make_uint4(directionalCount, clusteredCount, 0, 0);
    params_.screen = simd_make_float4((float)width, (float)height, sliceScale, -logf(kCameraNear) * sliceScale);
    ensureClusterBuffers(clusterCount());
}

void LightCuller::encodeCulling(MTL::CommandBuffer* commandBuffer, GpuTimer& gpuTimer) {
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder(gpuTimer.computeDescriptor("Light culling"));
    encoder->setComputePipelineState(cullPipeline_);
    encoder->setBytes(&params_, sizeof(ClusterParams), 0);
    encoder->setBuffer(lightBuffers_[slot_], 0, 1);
    encoder->setBuffer(clusterCounts_, 0, 2);
    encoder->setBuffer(clusterIndices_, 0, 3);
    encoder->dispatchThreads(MTL::Size::Make(params_.grid.x, params_.grid.y, params_.grid.z), MTL::Size::Make(8, 8, 1));
    encoder->endEncoding();
}

void LightCuller::bind(MTL::RenderCommandEncoder* encoder) const {
    encoder->setFragmentBytes(&params_, sizeof(ClusterParams), 4);
    encoder->setFragmentBuffer(lightBuffers_[slot_], 0, 5);
    encoder->setFragmentBuffer(clusterCounts_, 0, 6);
    encoder->setFragmentBuffer(clusterIndices_, 0, 7);
}
