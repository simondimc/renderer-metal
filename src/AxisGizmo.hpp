#pragma once
#include <Metal/Metal.hpp>

// X/Y/Z reference axes through the world origin (X=red, Y=green, Z=blue), shown while the
// scene editor's UI mode is active. Uses its own tiny unlit pipeline (see axisVertexMain/
// axisFragmentMain in Shader.metal) since it's a different vertex layout (position+color)
// than the textured/lit cube mesh.
struct AxisGizmo {
    MTL::Buffer* vertexBuffer = nullptr;
    MTL::RenderPipelineState* pipelineState = nullptr;
    NS::UInteger vertexCount = 0;
};

AxisGizmo createAxisGizmo(MTL::Device* device, MTL::Library* library);

// Draws the gizmo into the currently active encoder. uniformBuffer/uniformOffset must point at a
// slot already holding an mvpMatrix for the identity model (the gizmo lives at the world origin).
void drawAxisGizmo(const AxisGizmo& gizmo, MTL::RenderCommandEncoder* encoder,
                    MTL::Buffer* uniformBuffer, NS::UInteger uniformOffset);

void releaseAxisGizmo(AxisGizmo& gizmo);
