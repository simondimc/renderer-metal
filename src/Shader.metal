#include <metal_stdlib>
using namespace metal;

struct VertexInput {
    float3 position [[attribute(0)]];
    float2 uv       [[attribute(1)]];
    float3 normal   [[attribute(2)]];
};

struct RasterData {
    float4 position [[position]];
    float2 uv;
    float3 worldPosition;
    float3 worldNormal;
};

struct Uniforms {
    float4x4 mvpMatrix;
    float4x4 modelMatrix;
    float4 lightDirection;
};

// Vertex Shader
vertex RasterData vertexMain(VertexInput in [[stage_in]],
                             constant Uniforms& uniforms [[buffer(1)]]) {
    RasterData out;
    out.position = uniforms.mvpMatrix * float4(in.position, 1.0);
    out.uv = in.uv;
    out.worldPosition = (uniforms.modelMatrix * float4(in.position, 1.0)).xyz;
    out.worldNormal = (uniforms.modelMatrix * float4(in.normal, 0.0)).xyz;
    return out;
}

// Fragment Shader
fragment float4 fragmentMain(RasterData in [[stage_in]],
                             constant Uniforms& uniforms [[buffer(1)]],
                             texture2d<float> tex [[texture(0)]],
                             sampler smp [[sampler(0)]]) {
    constexpr float ambientStrength = 0.15;
    constexpr float specularStrength = 0.5;
    constexpr float shininess = 32.0;

    // No separate view matrix in this renderer - the model matrix already places
    // objects relative to a camera fixed at the world origin (see perspectiveProjectionRightHanded).
    constexpr float3 cameraPosition = float3(0.0, 0.0, 0.0);

    float3 normal = normalize(in.worldNormal);
    float3 lightDir = normalize(uniforms.lightDirection.xyz);
    float3 viewDir = normalize(cameraPosition - in.worldPosition);
    float3 halfVector = normalize(lightDir + viewDir);

    float diffuse = max(dot(normal, lightDir), 0.0);
    float specular = powr(max(dot(normal, halfVector), 0.0), shininess) * specularStrength;

    float4 texColor = tex.sample(smp, in.uv);
    float3 litColor = texColor.rgb * (ambientStrength + (1.0 - ambientStrength) * diffuse) + specular;
    return float4(litColor, texColor.a);
}
