#include "Uniforms.hpp"
#include <cmath>

Uniforms computeUniforms(const Camera& cam, float cubeTime, int width, int height) {
    Uniforms u;

    float fov = 60.0f * (M_PI / 180.0f);
    float aspect = (float)width / (float)height;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;

    float f = 1.0f / tanf(fov / 2.0f);
    float zRange = farPlane - nearPlane;

    // STANDARD RIGHT-HANDED PROJECTION FOR METAL (Depth: 0 to 1)
    simd::float4x4 proj = simd_matrix(
        simd_make_float4(f / aspect, 0.0f,  0.0f,                             0.0f),  // Col 0
        simd_make_float4(0.0f,       f,     0.0f,                             0.0f),  // Col 1
        simd_make_float4(0.0f,       0.0f,  farPlane / -zRange,              -1.0f),  // Col 2 (Negated for RH)
        simd_make_float4(0.0f,       0.0f,  -(farPlane * nearPlane) / zRange, 0.0f)   // Col 3
    );

    // Standard Right-Handed Rotations (Original math works perfectly here!)
    float cosX = cosf(0), sinX = sinf(0);
    simd::float4x4 rotX = simd_matrix(
        simd_make_float4(1.0f, 0.0f,  0.0f,  0.0f), // Col 0
        simd_make_float4(0.0f, cosX,  sinX,  0.0f), // Col 1
        simd_make_float4(0.0f, -sinX, cosX,  0.0f), // Col 2
        simd_make_float4(0.0f, 0.0f,  0.0f,  1.0f)  // Col 3
    );

    float cosY = cosf(cubeTime * 0.8f), sinY = sinf(cubeTime * 0.8f);
    simd::float4x4 rotY = simd_matrix(
        simd_make_float4(cosY,  0.0f, -sinY, 0.0f), // Col 0
        simd_make_float4(0.0f,  1.0f, 0.0f,  0.0f), // Col 1
        simd_make_float4(sinY,  0.0f, cosY,  0.0f), // Col 2
        simd_make_float4(0.0f,  0.0f, 0.0f,  1.0f)  // Col 3
    );

    // Cube stays at the origin and only spins in place; the camera now handles scene distance/movement
    simd::float4x4 model = rotY * rotX;
    simd::float4x4 view = viewMatrix(cam);
    u.modelMatrix = model;
    u.mvpMatrix = proj * view * model;
    u.cameraPosition = simd_make_float4(cam.position.x, cam.position.y, cam.position.z, 1.0f);

    u.lightDirection = simd_make_float4(0.4f, 0.8f, 0.5f, 0.0f);

    return u;
}
