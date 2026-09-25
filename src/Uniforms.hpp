#pragma once
#include <simd/simd.h>
#include "Camera.hpp"
#include "Scene.hpp" // LightType

// How many lights get a shadow map: the first kMaxLights lights of the scene (in scene order). Must match
// MAX_LIGHTS in Shader.metal. Lights past these still shine - unshadowed - see kMaxClusteredLights.
constexpr size_t kMaxLights = 4;

// How many lights the renderer shades in total. All of them are handed to the GPU each frame and sorted
// into the camera's cluster grid (see LightCulling.hpp); extras beyond this are ignored.
constexpr size_t kMaxClusteredLights = 512;

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
    simd::float4x4 lightViewProj[kMaxLights];  // Directional/Spot shadow-map view-projection (the other light data lives in
                                               // the shared GPULight buffer, see LightCulling.hpp)
    simd::float4 cameraPosition;
    simd::float4 materialAlbedo;              // rgb = albedo tint
    simd::float4 materialParams;              // x = metallic, y = roughness, z = ao, w = useTextures (0/1)
    simd::float4x4 viewProjMatrix;            // camera only, no model - transmission projects refracted points to the screen
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
    // KHR_materials_transmission / _volume / _ior. Transmission 0 = an ordinary surface. Thickness is
    // in mesh units (scaled by the object's model matrix at draw time); 1e30 = no absorption.
    simd::float4 transmissionParams = {0.0f, 0.0f, 1e30f, 1.5f}; // x = transmission, y = thickness, z = attenuation distance, w = IOR
    simd::float4 attenuationColor = {1.0f, 1.0f, 1.0f, 0.0f};    // rgb = color the volume tints light toward over attenuation distance
};

// The camera's perspective projection alone (no view, no model) - see computeViewProj. Main.cpp reads
// its [0][0]/[1][1] scales to reconstruct view-space positions from depth for ambient occlusion.
// jitterNDC shifts the whole image by that much in normalized device coordinates (2 units = the full
// screen) - temporal anti-aliasing's per-frame sub-pixel offset. Zero = the plain, unshifted projection.
simd::float4x4 computeProjection(int width, int height, simd::float2 jitterNDC = {0.0f, 0.0f});

// Camera-only view-projection (no object model matrix) - see its own comment in Uniforms.cpp.
// Main.cpp uses this directly for motion blur's frame-to-frame reprojection.
simd::float4x4 computeViewProj(const Camera& cam, int width, int height, simd::float2 jitterNDC = {0.0f, 0.0f});

// Builds the per-object Uniforms: projects/views objectModel through the camera and copies in the
// Directional/Spot shadow-map matrices of the first kMaxLights lights, which the fragment shader needs
// to look shadows up. The lights themselves (position, color, ...) are not per object - they go to the
// GPU once per frame, see LightCulling.hpp. Call once per object per frame
// (objectModel = objectModelMatrix(obj)).
// material may be null (gizmo/markers, which don't use it) - a neutral default is written then.
Uniforms computeUniforms(const Camera& cam, const simd::float4x4& objectModel,
                          const SceneLight* lights, int lightCount,
                          int width, int height, const Material* material = nullptr,
                          simd::float2 jitterNDC = {0.0f, 0.0f});
