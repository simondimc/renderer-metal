#pragma once
#include <simd/simd.h>
#include "Camera.hpp"

// Must stay layout-compatible with the Uniforms struct in Shader.metal
struct Uniforms {
    simd::float4x4 mvpMatrix;
    simd::float4x4 modelMatrix;
    simd::float4 lightDirection;
    simd::float4 cameraPosition;
};

// Builds the per-frame Uniforms: projection, cube spin (via cubeTime), camera view/position, light
Uniforms computeUniforms(const Camera& cam, float cubeTime, int width, int height);
