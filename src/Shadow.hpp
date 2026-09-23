#pragma once
#include <simd/simd.h>

// Cube shadow map face resolution (per face, so total memory per light is 6x this squared).
constexpr int kShadowMapSize = 1024;

constexpr float kShadowNearPlane = 0.05f;
constexpr float kShadowFarPlane = 50.0f;

constexpr int kCubeFaceCount = 6;

// Builds the 6 face view-projection matrices for one point light's cube shadow map, in the same
// slice order Metal uses for a MTLTextureType::TypeCube texture: +X, -X, +Y, -Y, +Z, -Z. Each
// face uses an exact 90-degree FOV so the six together tile the full sphere around the light with
// no gaps or overlap - unlike the single-frustum approach this replaces, every direction around
// the light is covered, so any object anywhere around it can both cast and receive its shadow.
void computeCubeShadowMatrices(simd::float3 lightPos, float nearPlane, float farPlane,
                                simd::float4x4 outMatrices[kCubeFaceCount]);
