#include <metal_stdlib>
using namespace metal;

#define MAX_LIGHTS 4 // must match kMaxLights in Uniforms.hpp

struct VertexInput {
    float3 position [[attribute(0)]];
    float2 uv       [[attribute(1)]];
    float3 normal   [[attribute(2)]];
    float3 tangent  [[attribute(3)]];
};

struct RasterData {
    float4 position [[position]];
    float2 uv;
    float3 worldPosition;
    float3 worldNormal;
    float3 worldTangent;
};

struct Uniforms {
    float4x4 mvpMatrix;
    float4x4 modelMatrix;
    float4 lightPositions[MAX_LIGHTS]; // world space, xyz used
    float4 lightColors[MAX_LIGHTS];    // rgb = color, a = intensity
    int4 lightMeta;                    // x = active light count
    float4 cameraPosition;
};

// Vertex Shader
vertex RasterData vertexMain(VertexInput in [[stage_in]],
                             constant Uniforms& uniforms [[buffer(1)]]) {
    RasterData out;
    out.position = uniforms.mvpMatrix * float4(in.position, 1.0);
    out.uv = in.uv;
    out.worldPosition = (uniforms.modelMatrix * float4(in.position, 1.0)).xyz;
    out.worldNormal = (uniforms.modelMatrix * float4(in.normal, 0.0)).xyz;
    out.worldTangent = (uniforms.modelMatrix * float4(in.tangent, 0.0)).xyz;
    return out;
}

// Cube shadow pass: rendered 6 times per light (once per cube face, see Shadow.hpp's
// computeCubeShadowMatrices and Main.cpp's per-light/per-face shadow pass loop). Rather than
// hardware depth-compare (which would need to know, at sampling time, which of the 6 faces and
// which face-local UV a world position falls into), this stores plain world-space distance from
// the light as a color value - so shading can later sample the cube by direction alone and get a
// distance to compare against, with Metal's hardware cube-map fetch (and its automatic
// cross-face filtering) handling face selection for free.
struct CubeShadowRasterData {
    float4 position [[position]];
    float3 worldPosition;
};

vertex CubeShadowRasterData cubeShadowVertexMain(VertexInput in [[stage_in]],
                                                 constant Uniforms& uniforms [[buffer(1)]],
                                                 constant float4x4& lightViewProj [[buffer(2)]]) {
    CubeShadowRasterData out;
    float4 worldPos = uniforms.modelMatrix * float4(in.position, 1.0);
    out.position = lightViewProj * worldPos;
    out.worldPosition = worldPos.xyz;
    return out;
}

fragment float cubeShadowFragmentMain(CubeShadowRasterData in [[stage_in]],
                                      constant float3& lightPosition [[buffer(3)]]) {
    return length(in.worldPosition - lightPosition);
}

// 1.0 = fully lit, 0.0 = fully in shadow. Sampling by direction (rather than a light-space
// projected UV) means every direction around the light is valid - no "outside the frustum" case
// to special-case, unlike a single-frustum shadow map.
static float sampleCubeShadow(float3 worldPosition, float3 lightPosition,
                              texturecube<float> shadowCube, sampler shadowSampler) {
    float3 fragToLight = worldPosition - lightPosition;
    float currentDistance = length(fragToLight);
    float closestDistance = shadowCube.sample(shadowSampler, fragToLight).r;

    constexpr float bias = 0.05; // world-space units; mitigates shadow acne from limited face resolution
    return (currentDistance - bias > closestDistance) ? 0.0 : 1.0;
}

// Fragment Shader
fragment float4 fragmentMain(RasterData in [[stage_in]],
                             constant Uniforms& uniforms [[buffer(1)]],
                             texture2d<float> tex [[texture(0)]],
                             texture2d<float> normalMap [[texture(1)]],
                             array<texturecube<float>, MAX_LIGHTS> shadowCubes [[texture(2)]],
                             sampler smp [[sampler(0)]],
                             sampler shadowSampler [[sampler(1)]]) {
    constexpr float ambientStrength = 0.15;
    constexpr float specularStrength = 0.5;
    constexpr float shininess = 32.0;

    // Build the TBN basis and use it to rotate the tangent-space normal map sample into world space
    float3 N = normalize(in.worldNormal);
    float3 T = normalize(in.worldTangent);
    float3 B = cross(N, T);
    float3x3 TBN = float3x3(T, B, N);

    float3 tangentNormal = normalMap.sample(smp, in.uv).rgb * 2.0 - 1.0;
    float3 normal = normalize(TBN * tangentNormal);

    float3 viewDir = normalize(uniforms.cameraPosition.xyz - in.worldPosition);
    float4 texColor = tex.sample(smp, in.uv);
    float3 litColor = texColor.rgb * ambientStrength;

    int lightCount = uniforms.lightMeta.x;
    for (int i = 0; i < lightCount; i++) {
        float3 toLight = uniforms.lightPositions[i].xyz - in.worldPosition;
        float lightDist = length(toLight);
        float3 lightDir = toLight / max(lightDist, 1e-4);
        float3 halfVector = normalize(lightDir + viewDir);

        // Standard point-light attenuation (constant/linear/quadratic falloff)
        float attenuation = 1.0 / (1.0 + 0.09 * lightDist + 0.032 * lightDist * lightDist);
        float3 radiance = uniforms.lightColors[i].rgb * uniforms.lightColors[i].a * attenuation;

        float diffuse = max(dot(normal, lightDir), 0.0);
        float specular = powr(max(dot(normal, halfVector), 0.0), shininess) * specularStrength;

        float shadow = sampleCubeShadow(in.worldPosition, uniforms.lightPositions[i].xyz, shadowCubes[i], shadowSampler);

        litColor += shadow * (texColor.rgb * diffuse + specular) * radiance;
    }

    return float4(litColor, texColor.a);
}

// --- Axis gizmo: flat-colored lines, no lighting/texturing ---

struct AxisVertexInput {
    float3 position [[attribute(0)]];
    float3 color    [[attribute(1)]];
};

struct AxisRasterData {
    float4 position [[position]];
    float3 color;
};

vertex AxisRasterData axisVertexMain(AxisVertexInput in [[stage_in]],
                                     constant Uniforms& uniforms [[buffer(1)]]) {
    AxisRasterData out;
    out.position = uniforms.mvpMatrix * float4(in.position, 1.0);
    out.color = in.color;
    return out;
}

fragment float4 axisFragmentMain(AxisRasterData in [[stage_in]]) {
    return float4(in.color, 1.0);
}
