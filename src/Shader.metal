#include <metal_stdlib>
using namespace metal;

struct VertexInput {
    float2 position [[attribute(0)]];
    float3 color    [[attribute(1)]];
};

struct RasterData {
    float4 position [[position]];
    float3 color;
};

// Vertex Shader
vertex RasterData vertexMain(VertexInput in [[stage_in]]) {
    RasterData out;
    out.position = float4(in.position, 0.0, 1.0);
    out.color = in.color;
    return out;
}

// Fragment Shader
fragment float4 fragmentMain(RasterData in [[stage_in]]) {
    return float4(in.color, 1.0); // Output interpolated RGB color
}
