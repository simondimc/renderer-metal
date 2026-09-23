#include "LightMarker.hpp"

namespace {
constexpr float kRadius = 0.2f;
constexpr float kR = 1.0f, kG = 0.85f, kB = 0.3f; // warm yellow, distinct from the RGB axis gizmo

// position(3) + color(3) per vertex, 2 vertices per ray (drawn as MTL::PrimitiveTypeLine).
// A small 3D "jack" (3 perpendicular rays) centered on the light's position.
constexpr float kMarkerVertices[] = {
    -kRadius, 0.0f, 0.0f,   kR, kG, kB,
    kRadius,  0.0f, 0.0f,   kR, kG, kB,
    0.0f, -kRadius, 0.0f,   kR, kG, kB,
    0.0f,  kRadius, 0.0f,   kR, kG, kB,
    0.0f, 0.0f, -kRadius,   kR, kG, kB,
    0.0f, 0.0f,  kRadius,   kR, kG, kB,
};
} // namespace

LightMarker createLightMarker(MTL::Device* device, MTL::Library* library) {
    LightMarker marker;
    marker.vertexCount = sizeof(kMarkerVertices) / (6 * sizeof(float));
    marker.vertexBuffer = device->newBuffer(kMarkerVertices, sizeof(kMarkerVertices), MTL::ResourceStorageModeShared);

    MTL::Function* vertFunc = library->newFunction(NS::String::string("axisVertexMain", NS::UTF8StringEncoding));
    MTL::Function* fragFunc = library->newFunction(NS::String::string("axisFragmentMain", NS::UTF8StringEncoding));

    MTL::VertexDescriptor* vertexDesc = MTL::VertexDescriptor::vertexDescriptor();
    vertexDesc->attributes()->object(0)->setFormat(MTL::VertexFormatFloat3);
    vertexDesc->attributes()->object(0)->setOffset(0);
    vertexDesc->attributes()->object(0)->setBufferIndex(0);
    vertexDesc->attributes()->object(1)->setFormat(MTL::VertexFormatFloat3);
    vertexDesc->attributes()->object(1)->setOffset(3 * sizeof(float));
    vertexDesc->attributes()->object(1)->setBufferIndex(0);
    vertexDesc->layouts()->object(0)->setStride(6 * sizeof(float));

    MTL::RenderPipelineDescriptor* pipeDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pipeDesc->setVertexFunction(vertFunc);
    pipeDesc->setFragmentFunction(fragFunc);
    pipeDesc->setVertexDescriptor(vertexDesc);
    pipeDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    pipeDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);

    NS::Error* error = nullptr;
    marker.pipelineState = device->newRenderPipelineState(pipeDesc, &error);

    vertFunc->release();
    fragFunc->release();
    pipeDesc->release();

    return marker;
}

void drawLightMarker(const LightMarker& marker, MTL::RenderCommandEncoder* encoder,
                      MTL::Buffer* uniformBuffer, NS::UInteger uniformOffset) {
    encoder->setRenderPipelineState(marker.pipelineState);
    encoder->setVertexBuffer(marker.vertexBuffer, 0, 0);
    encoder->setVertexBuffer(uniformBuffer, uniformOffset, 1);
    encoder->drawPrimitives(MTL::PrimitiveTypeLine, (NS::UInteger)0, marker.vertexCount);
}

void releaseLightMarker(LightMarker& marker) {
    marker.vertexBuffer->release();
    marker.pipelineState->release();
}
