#include "Uniforms.hpp"
#include <algorithm>
#include <cmath>

// Pulled out of computeUniforms so Main.cpp can also get the bare camera view-projection (no
// per-object model matrix) for motion blur's frame-to-frame reprojection, without duplicating the
// projection setup (and risking it drifting out of sync with near/far here - see
// CAMERA_NEAR_PLANE/FAR_PLANE's comment in Shader.metal, which must match these).
simd::float4x4 computeProjection(int width, int height) {
    float fov = 60.0f * (M_PI / 180.0f);
    float aspect = (float)width / (float)height;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;

    float f = 1.0f / tanf(fov / 2.0f);
    float zRange = farPlane - nearPlane;

    // STANDARD RIGHT-HANDED PROJECTION FOR METAL (Depth: 0 to 1)
    return simd_matrix(
        simd_make_float4(f / aspect, 0.0f,  0.0f,                             0.0f),  // Col 0
        simd_make_float4(0.0f,       f,     0.0f,                             0.0f),  // Col 1
        simd_make_float4(0.0f,       0.0f,  farPlane / -zRange,              -1.0f),  // Col 2 (Negated for RH)
        simd_make_float4(0.0f,       0.0f,  -(farPlane * nearPlane) / zRange, 0.0f)   // Col 3
    );
}

simd::float4x4 computeViewProj(const Camera& cam, int width, int height) {
    return computeProjection(width, height) * viewMatrix(cam);
}

Uniforms computeUniforms(const Camera& cam, const simd::float4x4& objectModel,
                          const SceneLight* lights, int lightCount,
                          int width, int height, const Material* material) {
    Uniforms u;

    simd::float4x4 viewProj = computeViewProj(cam, width, height);
    u.modelMatrix = objectModel;
    u.mvpMatrix = viewProj * objectModel;
    u.viewProjMatrix = viewProj;
    u.cameraPosition = simd_make_float4(cam.position.x, cam.position.y, cam.position.z, 1.0f);

    int count = std::min(lightCount, (int)kMaxLights);
    for (int i = 0; i < count; i++) {
        const SceneLight& light = lights[i];
        u.lightViewProj[i] = light.shadowViewProj;
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

    Material fallback;
    const Material& m = material ? *material : fallback;
    u.materialAlbedo = simd_make_float4(m.albedo[0], m.albedo[1], m.albedo[2], 1.0f);
    u.materialParams = simd_make_float4(m.metallic, m.roughness, m.ao, m.useTextures ? 1.0f : 0.0f);

    return u;
}
