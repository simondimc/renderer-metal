#include <metal_stdlib>
using namespace metal;

struct VertexInput {
    float3 position [[attribute(0)]];
    float3 color    [[attribute(1)]];
};

struct RasterData {
    float4 position [[position]];
    float3 color;
};

struct Uniforms {
    float4x4 modelViewProjectionMatrix;
};

// Vertex Shader
vertex RasterData vertexMain(VertexInput in [[stage_in]],
                             constant Uniforms& uniforms [[buffer(1)]]) {
    RasterData out;
    out.position = uniforms.modelViewProjectionMatrix * float4(in.position, 1.0);
    out.color = in.color;
    return out;
}

// Fragment Shader
fragment float4 fragmentMain(RasterData in [[stage_in]]) {
    return float4(in.color, 1.0);
}
