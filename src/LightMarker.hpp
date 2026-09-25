#pragma once
#include <Metal/Metal.hpp>

// Small "jack" cross drawn at each light's position in the scene editor, so lights are visible
// even though they have no mesh of their own, plus a direction ray (drawn for Directional/Spot/
// Area lights only) showing which way they point. Both reuse the axis gizmo's unlit
// position+color pipeline (axisVertexMain/axisFragmentMain in Shader.metal), just with their own
// vertex buffers.
struct LightMarker {
    MTL::Buffer* vertexBuffer = nullptr;
    MTL::Buffer* directionRayVertexBuffer = nullptr;
    MTL::RenderPipelineState* pipelineState = nullptr;
    NS::UInteger vertexCount = 0;
    NS::UInteger directionRayVertexCount = 0;
};

LightMarker createLightMarker(MTL::Device* device, MTL::Library* library);

// uniformBuffer/uniformOffset must point at a slot holding an mvpMatrix that places the marker
// at the light's world position (translation only, no rotation/scale needed for a point marker).
void drawLightMarker(const LightMarker& marker, MTL::RenderCommandEncoder* encoder,
                      MTL::Buffer* uniformBuffer, NS::UInteger uniformOffset);

// uniformBuffer/uniformOffset must point at a slot holding an mvpMatrix built from the light
// object's full rotation + translation (e.g. objectModelMatrix(obj)), so the ray points along the
// light's actual direction.
void drawLightDirectionRay(const LightMarker& marker, MTL::RenderCommandEncoder* encoder,
                            MTL::Buffer* uniformBuffer, NS::UInteger uniformOffset);

void releaseLightMarker(LightMarker& marker);
