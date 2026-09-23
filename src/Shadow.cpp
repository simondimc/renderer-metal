#include "Shadow.hpp"
#include <cmath>

namespace {
struct CubeFace {
    simd::float3 look;
    simd::float3 up;
};

// Cube-map face look/up directions, matching Metal's hardware direction-to-face/uv convention
// (same "positive u / positive v per face" table as D3D/Vulkan): each face's `up` here is the
// world direction pointing toward that face's v=0 edge. This is the vertical mirror of the
// up-vectors many OpenGL cube-shadow tutorials use, since OpenGL's texture origin is bottom-left
// while Metal's is top-left - reusing OpenGL's up-vectors as-is renders each face upside down
// relative to how Metal's texturecube::sample(direction) reads it back.
constexpr CubeFace kCubeFaces[kCubeFaceCount] = {
    {{ 1.0f,  0.0f,  0.0f}, {0.0f,  1.0f,  0.0f}}, // +X
    {{-1.0f,  0.0f,  0.0f}, {0.0f,  1.0f,  0.0f}}, // -X
    {{ 0.0f,  1.0f,  0.0f}, {0.0f,  0.0f, -1.0f}}, // +Y
    {{ 0.0f, -1.0f,  0.0f}, {0.0f,  0.0f,  1.0f}}, // -Y
    {{ 0.0f,  0.0f,  1.0f}, {0.0f,  1.0f,  0.0f}}, // +Z
    {{ 0.0f,  0.0f, -1.0f}, {0.0f,  1.0f,  0.0f}}, // -Z
};
} // namespace

void computeCubeShadowMatrices(simd::float3 lightPos, float nearPlane, float farPlane,
                                simd::float4x4 outMatrices[kCubeFaceCount]) {
    // A 90-degree FOV perspective, shared by every face - exactly matches a cube face's angular
    // extent, so f = 1/tan(45 deg) = 1 and aspect is always 1 (square shadow map faces).
    float f = 1.0f;
    float zRange = farPlane - nearPlane;
    simd::float4x4 proj = simd_matrix(
        simd_make_float4(f,    0.0f, 0.0f,                              0.0f),
        simd_make_float4(0.0f, f,    0.0f,                              0.0f),
        simd_make_float4(0.0f, 0.0f, farPlane / -zRange,               -1.0f),
        simd_make_float4(0.0f, 0.0f, -(farPlane * nearPlane) / zRange,  0.0f)
    );

    for (int i = 0; i < kCubeFaceCount; i++) {
        simd::float3 front = kCubeFaces[i].look;
        simd::float3 up = kCubeFaces[i].up;
        // cross(up, front) here, not cross(front, up) - the argument order that matches Metal's
        // per-face u-direction (verified against the hardware table for all 6 faces); swapping it
        // is the other half of the up-vector fix above, not an arbitrary choice.
        simd::float3 right = simd_normalize(simd_cross(up, front));

        // Same right-handed lookAt construction as Camera.cpp's viewMatrix
        simd::float4x4 view = simd_matrix(
            simd_make_float4(right.x, up.x, -front.x, 0.0f),
            simd_make_float4(right.y, up.y, -front.y, 0.0f),
            simd_make_float4(right.z, up.z, -front.z, 0.0f),
            simd_make_float4(-simd_dot(right, lightPos), -simd_dot(up, lightPos), simd_dot(front, lightPos), 1.0f)
        );

        outMatrices[i] = proj * view;
    }
}
