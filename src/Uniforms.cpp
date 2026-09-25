#include "Uniforms.hpp"
#include <algorithm>
#include <cmath>

// Pulled out of computeUniforms so Main.cpp can also get the bare camera view-projection (no
// per-object model matrix) for motion blur's frame-to-frame reprojection, without duplicating the
// projection setup (and risking it drifting out of sync with near/far here - see
// CAMERA_NEAR_PLANE/FAR_PLANE's comment in Shader.metal, which must match these).
simd::float4x4 computeProjection(int width, int height, simd::float2 jitterNDC) {
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
        // Col 2: the -jitter terms add jitterNDC to x/w and y/w after the divide (w = -z_view), i.e. shift
        // the whole rendered image by a sub-pixel amount without touching depth.
        simd_make_float4(-jitterNDC.x, -jitterNDC.y, farPlane / -zRange,     -1.0f),  // Col 2 (Negated for RH)
        simd_make_float4(0.0f,       0.0f,  -(farPlane * nearPlane) / zRange, 0.0f)   // Col 3
    );
}

simd::float4x4 computeViewProj(const Camera& cam, int width, int height, simd::float2 jitterNDC) {
    return computeProjection(width, height, jitterNDC) * viewMatrix(cam);
}

FrameUniforms computeFrameUniforms(const Camera& cam, const SceneLight* lights, int lightCount,
                                    int width, int height, simd::float2 jitterNDC) {
    FrameUniforms u = {};
    u.viewProj = computeViewProj(cam, width, height, jitterNDC);
    u.cameraPosition = simd_make_float4(cam.position.x, cam.position.y, cam.position.z, 1.0f);
    int count = std::min(lightCount, (int)kMaxLights);
    for (int i = 0; i < count; i++) {
        u.lightViewProj[i] = lights[i].shadowViewProj;
    }
    return u;
}

InstanceData computeInstanceData(const simd::float4x4& objectModel, const Material* material) {
    Material fallback;
    const Material& m = material ? *material : fallback;
    InstanceData data;
    data.model = objectModel;
    data.materialAlbedo = simd_make_float4(m.albedo[0], m.albedo[1], m.albedo[2], 1.0f);
    data.materialParams = simd_make_float4(m.metallic, m.roughness, m.ao, m.useTextures ? 1.0f : 0.0f);
    data.materialAniso = simd_make_float4(m.anisotropy, m.anisotropyRotation * ((float)M_PI / 180.0f), 0.0f, 0.0f);
    return data;
}

Uniforms computeUniforms(const Camera& cam, const simd::float4x4& objectModel, int width, int height) {
    Uniforms u;
    u.mvpMatrix = computeViewProj(cam, width, height) * objectModel;
    return u;
}
