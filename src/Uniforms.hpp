#pragma once
#include <simd/simd.h>
#include "Camera.hpp"
#include "Scene.hpp" // LightType

// Must match MAX_LIGHTS in Shader.metal
constexpr size_t kMaxLights = 4;

// CPU-side description of one active light, gathered from a SceneObject each frame (see Main.cpp).
// Fields not used by a given type are harmless zero/default values, not special-cased away, so the
// GPU side can stay branch-light.
struct SceneLight {
    LightType type = LightType::Point;
    simd::float3 position = {0.0f, 0.0f, 0.0f};    // Point/Spot/Area (world space)
    simd::float3 direction = {0.0f, 0.0f, -1.0f};  // Directional/Spot/Area emission direction (normalized)
    simd::float3 right = {1.0f, 0.0f, 0.0f};       // Area only: local right axis (normalized, world space)
    simd::float3 up = {0.0f, 1.0f, 0.0f};          // Area only: local up axis (normalized, world space)
    simd::float3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    float spotCosInner = 0.965f; // cos(~15deg)
    float spotCosOuter = 0.906f; // cos(~25deg)
    simd::float2 areaHalfSize = {0.5f, 0.5f};
    // Directional/Spot only: light-space view-projection for the single-frustum shadow map (see
    // Shadow.hpp's computeDirectionalShadowMatrix/computeSpotShadowMatrix). Unused by Point (which
    // uses the cube shadow maps instead) and Area (no shadows yet).
    simd::float4x4 shadowViewProj = matrix_identity_float4x4;
};

// Must stay layout-compatible with the Uniforms struct in Shader.metal, field for field.
struct Uniforms {
    simd::float4x4 mvpMatrix;
    simd::float4x4 modelMatrix;
    simd::float4x4 lightViewProj[kMaxLights];  // Directional/Spot shadow-map view-projection
    simd::float4 lightPositions[kMaxLights];  // xyz = position (Point/Spot/Area)
    simd::float4 lightDirections[kMaxLights]; // xyz = normalized emission direction (Directional/Spot/Area)
    simd::float4 lightRight[kMaxLights];      // xyz = area light local right axis
    simd::float4 lightUp[kMaxLights];         // xyz = area light local up axis
    simd::float4 lightColors[kMaxLights];     // rgb = color, a = intensity
    simd::float4 lightParams[kMaxLights];     // x=spotCosInner, y=spotCosOuter, z=areaHalfWidth, w=areaHalfHeight
    simd::int4 lightTypes[kMaxLights];        // x = LightType of that light
    simd::int4 lightMeta;                     // x = active light count
    simd::float4 cameraPosition;
    simd::float4 materialAlbedo;              // rgb = albedo tint
    simd::float4 materialParams;              // x = metallic, y = roughness, z = ao, w = useTextures (0/1)
};

// glTF's alphaMode. Opaque ignores alpha entirely; Mask keeps or discards each pixel against a
// cutoff (foliage, chain-link); Blend mixes the surface over what's behind it (glass, smoke) and
// so is drawn in a separate, back-to-front sorted pass. Must match ALPHA_MODE_* in Shader.metal.
enum class AlphaMode { Opaque = 0, Mask = 1, Blend = 2 };

// Per-draw glTF material factors (one per submesh - see MeshLoader.hpp), bound at fragment buffer 2
// and multiplied with the Scene Editor's per-object Material. Must stay layout-compatible with
// MaterialParams in Shader.metal. The defaults are neutral, which is what non-glTF meshes use.
struct MaterialParams {
    simd::float4 baseColorFactor = {1.0f, 1.0f, 1.0f, 1.0f};
    simd::float4 factors = {1.0f, 1.0f, 0.0f, 0.0f}; // x = metallic, y = roughness, z = occlusion strength
    simd::float4 emissiveFactor = {0.0f, 0.0f, 0.0f, 0.0f}; // rgb = emissive color * emissive strength (HDR, may exceed 1)
    simd::float4 alphaParams = {0.0f, 0.5f, 0.0f, 0.0f};    // x = AlphaMode, y = Mask cutoff
};

// Camera-only view-projection (no object model matrix) - see its own comment in Uniforms.cpp.
// Main.cpp uses this directly for motion blur's frame-to-frame reprojection.
simd::float4x4 computeViewProj(const Camera& cam, int width, int height);

// Builds the per-object Uniforms: projects/views objectModel through the camera, lit by up to
// kMaxLights lights of any type (extras beyond that are ignored). Call once per object per frame
// (objectModel = objectModelMatrix(obj)). Per-light shadow-map matrices are handled separately in
// Main.cpp (see Shadow.hpp) - they change per light, not per object, so they don't belong here.
// material may be null (gizmo/markers, which don't use it) - a neutral default is written then.
Uniforms computeUniforms(const Camera& cam, const simd::float4x4& objectModel,
                          const SceneLight* lights, int lightCount,
                          int width, int height, const Material* material = nullptr);
