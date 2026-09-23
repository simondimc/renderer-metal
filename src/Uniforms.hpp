#pragma once
#include <simd/simd.h>
#include "Camera.hpp"

// Must match MAX_LIGHTS in Shader.metal
constexpr size_t kMaxLights = 4;

struct PointLight {
    simd::float3 position;
    simd::float3 color;
    float intensity;
};

// Must stay layout-compatible with the Uniforms struct in Shader.metal
struct Uniforms {
    simd::float4x4 mvpMatrix;
    simd::float4x4 modelMatrix;
    simd::float4 lightPositions[kMaxLights]; // world space, xyz used
    simd::float4 lightColors[kMaxLights];    // rgb = color, a = intensity
    simd::int4 lightMeta;                    // x = active light count
    simd::float4 cameraPosition;
};

// Builds the per-object Uniforms: projects/views objectModel through the camera, lit by up to
// kMaxLights point lights (extras beyond that are ignored). Call once per object per frame
// (objectModel = objectModelMatrix(obj)). Per-light shadow-map matrices are handled separately in
// Main.cpp (see Shadow.hpp) - they change per light, not per object, so they don't belong here.
Uniforms computeUniforms(const Camera& cam, const simd::float4x4& objectModel,
                          const PointLight* lights, int lightCount,
                          int width, int height);
