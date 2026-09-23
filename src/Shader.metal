#include <metal_stdlib>
using namespace metal;

struct VertexInput {
    float3 position [[attribute(0)]];
    float2 uv       [[attribute(1)]];
};

struct RasterData {
    float4 position [[position]];
    float2 uv;
};

struct Uniforms {
    float4x4 modelViewProjectionMatrix;
};

// Vertex Shader
vertex RasterData vertexMain(VertexInput in [[stage_in]],
                             constant Uniforms& uniforms [[buffer(1)]]) {
    RasterData out;
    out.position = uniforms.modelViewProjectionMatrix * float4(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

// Fragment Shader
fragment float4 fragmentMain(RasterData in [[stage_in]],
                             texture2d<float> tex [[texture(0)]],
                             sampler smp [[sampler(0)]]) {
    return tex.sample(smp, in.uv);
}
