#include "AxisGizmo.hpp"

namespace {
constexpr float kAxisLength = 5.0f;

// position(3) + color(3) per vertex, 2 vertices per axis (drawn as MTL::PrimitiveTypeLine)
constexpr float kAxisVertices[] = {
    -kAxisLength, 0.0f, 0.0f,   1.0f, 0.0f, 0.0f, // X axis (red)
    kAxisLength,  0.0f, 0.0f,   1.0f, 0.0f, 0.0f,
    0.0f, -kAxisLength, 0.0f,   0.0f, 1.0f, 0.0f, // Y axis (green)
    0.0f,  kAxisLength, 0.0f,   0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, -kAxisLength,   0.0f, 0.0f, 1.0f, // Z axis (blue)
    0.0f, 0.0f,  kAxisLength,   0.0f, 0.0f, 1.0f,
};
} // namespace

AxisGizmo createAxisGizmo(MTL::Device* device, MTL::Library* library) {
    AxisGizmo gizmo;
    gizmo.vertexCount = sizeof(kAxisVertices) / (6 * sizeof(float));
    gizmo.vertexBuffer = device->newBuffer(kAxisVertices, sizeof(kAxisVertices), MTL::ResourceStorageModeShared);

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
    gizmo.pipelineState = device->newRenderPipelineState(pipeDesc, &error);

    vertFunc->release();
    fragFunc->release();
    pipeDesc->release();

    return gizmo;
}

void drawAxisGizmo(const AxisGizmo& gizmo, MTL::RenderCommandEncoder* encoder,
                    MTL::Buffer* uniformBuffer, NS::UInteger uniformOffset) {
    encoder->setRenderPipelineState(gizmo.pipelineState);
    encoder->setVertexBuffer(gizmo.vertexBuffer, 0, 0);
    encoder->setVertexBuffer(uniformBuffer, uniformOffset, 1);
    encoder->drawPrimitives(MTL::PrimitiveTypeLine, (NS::UInteger)0, gizmo.vertexCount);
}

void releaseAxisGizmo(AxisGizmo& gizmo) {
    gizmo.vertexBuffer->release();
    gizmo.pipelineState->release();
}
