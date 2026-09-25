#pragma once
#include <simd/simd.h>

// Cube shadow map face resolution (per face, so total memory per light is 6x this squared).
constexpr int kShadowMapSize = 2048;

constexpr float kShadowNearPlane = 0.05f;
constexpr float kShadowFarPlane = 50.0f;

constexpr int kCubeFaceCount = 6;

// Single-frustum shadow map resolution for Directional/Spot lights (one 2D depth texture each,
// unlike the point light's 6-face cube).
constexpr int kShadow2DMapSize = 4096;

// A directional light has no true position, so its shadow volume is centered on the light
// object's `position` field instead (reused in the editor as a movable anchor - see
// SceneEditorPanel). kDirectionalShadowHalfExtent is the half-width/height of that box in world
// units; kDirectionalShadowDistance is how far behind the anchor the virtual shadow camera sits
// (also used as the far-plane distance, so the anchor falls roughly at mid-depth).
constexpr float kDirectionalShadowHalfExtent = 15.0f;
constexpr float kDirectionalShadowDistance = 25.0f;
constexpr float kDirectionalShadowNearPlane = 0.05f;
constexpr float kDirectionalShadowFarPlane = 2.0f * kDirectionalShadowDistance;

// Builds the 6 face view-projection matrices for one point light's cube shadow map, in the same
// slice order Metal uses for a MTLTextureType::TypeCube texture: +X, -X, +Y, -Y, +Z, -Z. Each
// face uses an exact 90-degree FOV so the six together tile the full sphere around the light with
// no gaps or overlap - unlike the single-frustum approach this replaces, every direction around
// the light is covered, so any object anywhere around it can both cast and receive its shadow.
void computeCubeShadowMatrices(simd::float3 lightPos, float nearPlane, float farPlane,
                                simd::float4x4 outMatrices[kCubeFaceCount]);

// Builds an orthographic light-space view-projection matrix for a directional light's shadow map.
// `right`/`up` must be normalized and mutually orthogonal to `direction` (true for the axes
// objectRight/objectUp/objectForward derive from a single rotation, see Scene.cpp).
simd::float4x4 computeDirectionalShadowMatrix(simd::float3 center, simd::float3 direction,
                                               simd::float3 right, simd::float3 up);

// Builds a perspective light-space view-projection matrix for a spot light's shadow map, with a
// field of view of 2 * outerAngleDegrees so the cone exactly fills the frustum.
simd::float4x4 computeSpotShadowMatrix(simd::float3 position, simd::float3 direction,
                                        simd::float3 right, simd::float3 up,
                                        float outerAngleDegrees, float nearPlane, float farPlane);
