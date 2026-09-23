#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include "Bridge.hpp"
#include "Camera.hpp"
#include "CubeMesh.hpp"
#include "Texture.hpp"
#include "Uniforms.hpp"

int main() {
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    int width = 800;
    int height = 600;
    GLFWwindow* window = glfwCreateWindow(width, height, "Renderer", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    // Mouse-look camera: capture and hide the cursor, route movement through mouseCallback
    Camera camera;
    glfwSetWindowUserPointer(window, &camera);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, mouseCallback);

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MTL::CommandQueue* cmdQueue = device->newCommandQueue();

    // Call our isolated bridge to bind the window with a layer
    setupMetalLayerForWindow(window, device);

    // Safely grab the newly attached layer back as a pure C++ pointer
    // We cast it via an inline __bridge simulation since we are in pure C++
    void* nativeWinPtr = glfwGetCocoaWindow(window);

    // Objective-C runtime trick to get the layer in pure C++:
    // This sends a 'contentView' message and a 'layer' message to the window pointer
    typedef id (*IdMessageSend)(id, SEL);
    IdMessageSend sendMsg = (IdMessageSend)objc_msgSend;
    id contentView = sendMsg((id)nativeWinPtr, sel_registerName("contentView"));
    CA::MetalLayer* cppMetalLayer = (CA::MetalLayer*)sendMsg(contentView, sel_registerName("layer"));

    // Allocate Depth Buffer based on initial window framebuffer sizing
    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatDepth32Float, (NS::UInteger)width, (NS::UInteger)height, false
    );
    desc->setStorageMode(MTL::StorageModePrivate);
    desc->setUsage(MTL::TextureUsageRenderTarget);
    MTL::Texture* depthTexture = device->newTexture(desc);

    // Create GPU buffers
    MTL::Buffer* vertexBuffer = device->newBuffer(CubeMesh::vertices, sizeof(CubeMesh::vertices), MTL::ResourceStorageModeShared);
    MTL::Buffer* indexBuffer = device->newBuffer(CubeMesh::indices, sizeof(CubeMesh::indices), MTL::ResourceStorageModeShared);
    constexpr NS::UInteger indexCount = sizeof(CubeMesh::indices) / sizeof(CubeMesh::indices[0]);

    // Create a buffer to hold the 4x4 matrix (Uniforms)
    MTL::Buffer* uniformBuffer = device->newBuffer(sizeof(Uniforms), MTL::ResourceStorageModeShared);

    // Compile the shader library
    NS::Error* error = nullptr;
    MTL::Library* library = device->newDefaultLibrary();

    MTL::Function* vertFunc = library->newFunction(NS::String::string("vertexMain", NS::UTF8StringEncoding));
    MTL::Function* fragFunc = library->newFunction(NS::String::string("fragmentMain", NS::UTF8StringEncoding));

    // Create the Vertex Descriptor layout
    MTL::VertexDescriptor* vertexDesc = MTL::VertexDescriptor::vertexDescriptor();
    // Position attribute (Float3 for X, Y, Z layout)
    vertexDesc->attributes()->object(0)->setFormat(MTL::VertexFormatFloat3);
    vertexDesc->attributes()->object(0)->setOffset(0);
    vertexDesc->attributes()->object(0)->setBufferIndex(0);
    // UV attribute (Offset matches 3 floats of position data)
    vertexDesc->attributes()->object(1)->setFormat(MTL::VertexFormatFloat2);
    vertexDesc->attributes()->object(1)->setOffset(3 * sizeof(float));
    vertexDesc->attributes()->object(1)->setBufferIndex(0);
    // Normal attribute (Offset matches 3 floats of position + 2 floats of UV)
    vertexDesc->attributes()->object(2)->setFormat(MTL::VertexFormatFloat3);
    vertexDesc->attributes()->object(2)->setOffset(5 * sizeof(float));
    vertexDesc->attributes()->object(2)->setBufferIndex(0);
    // Tangent attribute (Offset matches 3 Position + 2 UV + 3 Normal floats)
    vertexDesc->attributes()->object(3)->setFormat(MTL::VertexFormatFloat3);
    vertexDesc->attributes()->object(3)->setOffset(8 * sizeof(float));
    vertexDesc->attributes()->object(3)->setBufferIndex(0);
    vertexDesc->layouts()->object(0)->setStride(CubeMesh::vertexStrideFloats * sizeof(float));

    // Build the Pipeline State Object (PSO)
    MTL::RenderPipelineDescriptor* pipeDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pipeDesc->setVertexFunction(vertFunc);
    pipeDesc->setFragmentFunction(fragFunc);
    pipeDesc->setVertexDescriptor(vertexDesc);
    pipeDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    pipeDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);

    MTL::RenderPipelineState* pipelineState = device->newRenderPipelineState(pipeDesc, &error);

    MTL::DepthStencilDescriptor* depthDesc = MTL::DepthStencilDescriptor::alloc()->init();
    depthDesc->setDepthCompareFunction(MTL::CompareFunctionLess);
    depthDesc->setDepthWriteEnabled(true);
    MTL::DepthStencilState* depthState = device->newDepthStencilState(depthDesc);

    // run.sh/clean_run.sh launch the binary with the build/ directory as cwd
    MTL::Texture* colorTexture = loadTexture(device, "../texture/metal_plate_4k/textures/metal_plate_diff_4k.jpg");
    // Converted offline from the source EXR (DWAA compression, unsupported by stb_image) via ffmpeg
    MTL::Texture* normalTexture = loadTexture(device, "../texture/metal_plate_4k/textures/metal_plate_nor_gl_4k.png");
    if (!colorTexture || !normalTexture) return -1;

    MTL::SamplerDescriptor* samplerDesc = MTL::SamplerDescriptor::alloc()->init();
    samplerDesc->setMinFilter(MTL::SamplerMinMagFilterLinear);
    samplerDesc->setMagFilter(MTL::SamplerMinMagFilterLinear);
    samplerDesc->setMipFilter(MTL::SamplerMipFilterLinear);
    samplerDesc->setSAddressMode(MTL::SamplerAddressModeRepeat);
    samplerDesc->setTAddressMode(MTL::SamplerAddressModeRepeat);
    MTL::SamplerState* samplerState = device->newSamplerState(samplerDesc);

    float lastFrameTime = (float)glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        float currentTime = (float)glfwGetTime();
        float deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;
        processCameraInput(window, camera, deltaTime);

        // Create the frame memory pool
        NS::AutoreleasePool* framePool = NS::AutoreleasePool::alloc()->init();

        // Fetch the canvas
        CA::MetalDrawable* drawable = cppMetalLayer->nextDrawable();

        if (drawable) {
            int liveWidth, liveHeight;
            glfwGetFramebufferSize(window, &liveWidth, &liveHeight);

            // Regenerate depth attachment allocations if user scaling structural size bounds shifts
            if (liveWidth != width || liveHeight != height) {
                depthTexture->release();
                width = liveWidth;
                height = liveHeight;

                MTL::TextureDescriptor* newDesc = MTL::TextureDescriptor::texture2DDescriptor(
                    MTL::PixelFormatDepth32Float, (NS::UInteger)width, (NS::UInteger)height, false
                );
                newDesc->setStorageMode(MTL::StorageModePrivate);
                newDesc->setUsage(MTL::TextureUsageRenderTarget);
                depthTexture = device->newTexture(newDesc);
            }

            // Configure the render pass
            MTL::RenderPassDescriptor* rpd = MTL::RenderPassDescriptor::renderPassDescriptor();
            auto colorAttachment = rpd->colorAttachments()->object(0);

            colorAttachment->setClearColor({0.15, 0.15, 0.15, 1.0});
            colorAttachment->setLoadAction(MTL::LoadActionClear);
            colorAttachment->setStoreAction(MTL::StoreActionStore);
            colorAttachment->setTexture(drawable->texture());

            // Bind depth texture
            rpd->depthAttachment()->setTexture(depthTexture);
            rpd->depthAttachment()->setLoadAction(MTL::LoadActionClear);
            rpd->depthAttachment()->setClearDepth(1.0);
            rpd->depthAttachment()->setStoreAction(MTL::StoreActionDontCare);

            Uniforms uniforms = computeUniforms(camera, currentTime, liveWidth, liveHeight);
            memcpy(uniformBuffer->contents(), &uniforms, sizeof(Uniforms));

            // Request a command buffer from the queue
            MTL::CommandBuffer* cmdBuffer = cmdQueue->commandBuffer();

            // Create the encoder
            MTL::RenderCommandEncoder* encoder = cmdBuffer->renderCommandEncoder(rpd);

            // Issue drawing commands
            encoder->setRenderPipelineState(pipelineState);
            encoder->setDepthStencilState(depthState);
            encoder->setVertexBuffer(vertexBuffer, 0, 0);
            encoder->setVertexBuffer(uniformBuffer, 0, 1);
            encoder->setFragmentBuffer(uniformBuffer, 0, 1);
            encoder->setFragmentTexture(colorTexture, 0);
            encoder->setFragmentTexture(normalTexture, 1);
            encoder->setFragmentSamplerState(samplerState, 0);

            encoder->drawIndexedPrimitives(
                MTL::PrimitiveTypeTriangle,
                indexCount,
                MTL::IndexTypeUInt16,
                indexBuffer,
                0
            );

            // Present the texture onto the screen and submit to the GPU
            encoder->endEncoding();
            cmdBuffer->presentDrawable(drawable);
            cmdBuffer->commit();
        }

        // Drain the temporary memory pool for this frame
        framePool->release();
    }

    // Explicit GPU Cleanups
    samplerState->release();
    samplerDesc->release();
    colorTexture->release();
    normalTexture->release();
    depthTexture->release();
    depthState->release();
    depthDesc->release();
    pipelineState->release();
    pipeDesc->release();
    vertFunc->release();
    fragFunc->release();
    library->release();
    vertexBuffer->release();
    indexBuffer->release();
    uniformBuffer->release();
    cmdQueue->release();
    device->release();
    pool->release();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
