#pragma once
#include <Metal/Metal.hpp>
#include <simd/simd.h>
#include <vector>
#include "Camera.hpp"
#include "GpuTimer.hpp"
#include "Uniforms.hpp" // SceneLight

// Clustered forward ("Forward+") lighting. Shading every light at every pixel costs pixels x lights, so the
// camera's view volume is instead cut into a 3D grid of clusters - screen tiles times depth slices, spaced
// exponentially so near clusters stay small - and a compute pass (lightCullKernel in Shader.metal) records,
// per cluster, which lights can reach it. The scene's fragment shader finds its own cluster from its pixel
// position and view depth and shades only that short list, so the cost follows how many lights touch a
// pixel rather than how many exist. Directional lights reach everything, so they are not clustered: the
// shader loops over them separately.
//
// Materials, glass and blending are untouched - this is still forward shading, just with a per-pixel
// light list.

constexpr uint32_t kClusterTilePixels = 64;     // screen-space tile size
constexpr uint32_t kClusterSlices = 24;         // depth slices, from the near to the far plane
constexpr uint32_t kMaxLightsPerCluster = 64;   // a cluster touched by more lights than this drops the extras

// One light as the GPU reads it. Must stay layout-compatible with GPULight in Shader.metal (all float4).
struct GPULight {
    simd::float4 positionRange;  // xyz = world position (Point/Spot/Area), w = cull radius (its reach)
    simd::float4 directionType;  // xyz = normalized emission direction, w = LightType
    simd::float4 colorIntensity; // rgb = color, a = intensity
    simd::float4 params;         // x = spotCosInner, y = spotCosOuter, z = areaHalfWidth, w = areaHalfHeight
    simd::float4 right;          // xyz = area light local right axis, w = shadow-map slot (-1 = casts no shadow)
    simd::float4 up;             // xyz = area light local up axis, w = attenuation at the cull radius (see below)
};

// The grid's geometry and the light counts. Must stay layout-compatible with ClusterParams in Shader.metal.
struct ClusterParams {
    simd::float4x4 view;         // world -> view space
    simd::float4 projection;     // x = proj[0][0], y = proj[1][1] (un-jittered), z = near, w = far
    simd::uint4 grid;            // x, y = tile counts, z = depth slices, w = tile size in pixels
    simd::uint4 lights;          // x = directional lights (first in the buffer), y = clustered lights after them
    simd::float4 screen;         // x, y = width, height in pixels, z = slice scale, w = slice bias (see sliceOf)
};

// Distance beyond which a light of this color/intensity contributes less than ~0.005 (in linear HDR radiance),
// found from the falloff 1 / (1 + 0.09 d + 0.032 d^2) the scene shader uses. That falloff never reaches zero,
// so the shader subtracts its value at this radius (renormalizing), which makes the light end smoothly at the
// radius it is culled at instead of being cut off - the change to the lit result is below the cutoff.
float lightCullRadius(const SceneLight& light);

class LightCuller {
public:
    LightCuller(MTL::Device* device, MTL::Library* library, int framesInFlight);
    ~LightCuller();
    LightCuller(const LightCuller&) = delete;
    LightCuller& operator=(const LightCuller&) = delete;

    // Rebuilds the culling kernel from a freshly compiled library (shader hot reloading). Returns false, keeping
    // the current kernel, if it does not build.
    bool reload(MTL::Library* library);

    // Uploads this frame's lights (in scene order; the first kMaxLights are the ones with shadow maps) into the
    // frame slot's buffer and works out the grid for the camera. Call once per frame before encodeCulling.
    void prepare(int slot, const SceneLight* lights, int lightCount, const Camera& camera, int width, int height);

    // Compute pass that fills the cluster light lists. Must be encoded before any pass that shades with them.
    void encodeCulling(MTL::CommandBuffer* commandBuffer, GpuTimer& gpuTimer);

    // Binds everything the scene fragment shader reads (buffers 4-7) to the encoder.
    void bind(MTL::RenderCommandEncoder* encoder) const;

    uint32_t clusterCount() const { return params_.grid.x * params_.grid.y * params_.grid.z; }
    // Lights that ended up on the GPU this frame (directional + clustered), for the overlay.
    uint32_t lightCount() const { return params_.lights.x + params_.lights.y; }

private:
    void ensureClusterBuffers(uint32_t clusters);

    MTL::ComputePipelineState* makeCullPipeline(MTL::Library* library);

    MTL::Device* device_;
    MTL::ComputePipelineState* cullPipeline_ = nullptr;
    std::vector<MTL::Buffer*> lightBuffers_; // one per frame in flight (CPU-written every frame)
    MTL::Buffer* clusterCounts_ = nullptr;   // per cluster: how many lights touch it
    MTL::Buffer* clusterIndices_ = nullptr;  // per cluster: kMaxLightsPerCluster light indices
    uint32_t clusterCapacity_ = 0;
    int slot_ = 0;
    ClusterParams params_ = {};
};
