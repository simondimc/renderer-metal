#include "Uniforms.hpp"
#include <algorithm>
#include <cmath>

Uniforms computeUniforms(const Camera& cam, const simd::float4x4& objectModel,
                          const SceneLight* lights, int lightCount,
                          int width, int height) {
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

    simd::float4x4 view = viewMatrix(cam);
    u.modelMatrix = objectModel;
    u.mvpMatrix = proj * view * objectModel;
    u.cameraPosition = simd_make_float4(cam.position.x, cam.position.y, cam.position.z, 1.0f);

    int count = std::min(lightCount, (int)kMaxLights);
    for (int i = 0; i < count; i++) {
        const SceneLight& light = lights[i];
        u.lightPositions[i] = simd_make_float4(light.position.x, light.position.y, light.position.z, 1.0f);
        u.lightDirections[i] = simd_make_float4(light.direction.x, light.direction.y, light.direction.z, 0.0f);
        u.lightRight[i] = simd_make_float4(light.right.x, light.right.y, light.right.z, 0.0f);
        u.lightUp[i] = simd_make_float4(light.up.x, light.up.y, light.up.z, 0.0f);
        u.lightColors[i] = simd_make_float4(light.color.x, light.color.y, light.color.z, light.intensity);
        u.lightParams[i] = simd_make_float4(light.spotCosInner, light.spotCosOuter,
                                             light.areaHalfSize.x, light.areaHalfSize.y);
        u.lightTypes[i] = simd_make_int4((int)light.type, 0, 0, 0);
    }
    u.lightMeta = simd_make_int4(count, 0, 0, 0);

    return u;
}
