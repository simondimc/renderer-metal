#pragma once
#include <Metal/Metal.hpp>

// Small "jack" cross drawn at each point light's position in the scene editor, so lights are
// visible even though they have no mesh of their own. Reuses the axis gizmo's unlit
// position+color pipeline (axisVertexMain/axisFragmentMain in Shader.metal), just with its own
// small vertex buffer and pipeline object.
struct LightMarker {
    MTL::Buffer* vertexBuffer = nullptr;
    MTL::RenderPipelineState* pipelineState = nullptr;
    NS::UInteger vertexCount = 0;
};

LightMarker createLightMarker(MTL::Device* device, MTL::Library* library);

// uniformBuffer/uniformOffset must point at a slot holding an mvpMatrix that places the marker
// at the light's world position (translation only, no rotation/scale needed for a point marker).
void drawLightMarker(const LightMarker& marker, MTL::RenderCommandEncoder* encoder,
                      MTL::Buffer* uniformBuffer, NS::UInteger uniformOffset);

void releaseLightMarker(LightMarker& marker);
