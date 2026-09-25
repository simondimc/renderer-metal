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

namespace {
// Shared by computeDirectionalShadowMatrix/computeSpotShadowMatrix - same right-handed lookAt
// construction as the cube shadow matrices and Camera.cpp's viewMatrix (view space looks down -Z).
simd::float4x4 lookAtMatrix(simd::float3 eye, simd::float3 front, simd::float3 right, simd::float3 up) {
    return simd_matrix(
        simd_make_float4(right.x, up.x, -front.x, 0.0f),
        simd_make_float4(right.y, up.y, -front.y, 0.0f),
        simd_make_float4(right.z, up.z, -front.z, 0.0f),
        simd_make_float4(-simd_dot(right, eye), -simd_dot(up, eye), simd_dot(front, eye), 1.0f)
    );
}
} // namespace

simd::float4x4 computeDirectionalShadowMatrix(simd::float3 center, simd::float3 direction,
                                               simd::float3 right, simd::float3 up) {
    simd::float3 eye = center - direction * kDirectionalShadowDistance;
    simd::float4x4 view = lookAtMatrix(eye, direction, right, up);

    // Orthographic projection (Metal depth 0..1): no perspective divide, so unlike the point/spot
    // projections f/aspect don't apply - width/height are set directly from the half-extent.
    // Z column is -1/zRange, not +1/zRange: lookAtMatrix (like Camera.cpp's viewMatrix) puts
    // view-space Z negative in front of the eye, and every perspective projection in this codebase
    // negates it back (see the farPlane/-zRange terms below and in Uniforms.cpp) - an orthographic
    // projection needs that same flip, just without a perspective divide to hide a missing one.
    // Getting this wrong sends NDC.z negative for the whole scene, which Metal's clip test rejects
    // outright, so nothing ever reaches the shadow map.
    float r = kDirectionalShadowHalfExtent;
    float nearPlane = kDirectionalShadowNearPlane, farPlane = kDirectionalShadowFarPlane;
    float zRange = farPlane - nearPlane;
    simd::float4x4 proj = simd_matrix(
        simd_make_float4(1.0f / r, 0.0f,     0.0f,                 0.0f),
        simd_make_float4(0.0f,     1.0f / r, 0.0f,                 0.0f),
        simd_make_float4(0.0f,     0.0f,     -1.0f / zRange,       0.0f),
        simd_make_float4(0.0f,     0.0f,     -nearPlane / zRange,  1.0f)
    );

    return proj * view;
}

simd::float4x4 computeSpotShadowMatrix(simd::float3 position, simd::float3 direction,
                                        simd::float3 right, simd::float3 up,
                                        float outerAngleDegrees, float nearPlane, float farPlane) {
    simd::float4x4 view = lookAtMatrix(position, direction, right, up);

    // Same projection formula as the main camera (Uniforms.cpp) / cube shadow matrices above, but
    // with the FOV set from the cone's outer angle instead of the camera's fixed 60 degrees / a
    // fixed 90 degrees, so the frustum exactly covers the cone.
    constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
    float fov = fmin(2.0f * outerAngleDegrees * kDegToRad, 3.05f); // clamp shy of 180 deg
    float f = 1.0f / tanf(fov * 0.5f);
    float zRange = farPlane - nearPlane;
    simd::float4x4 proj = simd_matrix(
        simd_make_float4(f,    0.0f, 0.0f,                              0.0f),
        simd_make_float4(0.0f, f,    0.0f,                              0.0f),
        simd_make_float4(0.0f, 0.0f, farPlane / -zRange,               -1.0f),
        simd_make_float4(0.0f, 0.0f, -(farPlane * nearPlane) / zRange,  0.0f)
    );

    return proj * view;
}

uint64_t hashBytes(uint64_t hash, const void* data, size_t size) {
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}
