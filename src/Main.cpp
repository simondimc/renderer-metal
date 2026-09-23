#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <simd/simd.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "Bridge.hpp"

struct Uniforms {
    simd::float4x4 mvpMatrix;
    simd::float4x4 modelMatrix;
    simd::float4 lightDirection;
};

Uniforms perspectiveProjectionRightHanded(float time, int width, int height) {
    Uniforms u;

    float fov = 60.0f * (M_PI / 180.0f);
    float aspect = (float)width / (float)height;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;
    
    float f = 1.0f / tanf(fov / 2.0f);
    float zRange = farPlane - nearPlane;

    // STANDARD RIGHT-HANDED PROJECTION FOR METAL (Depth: 0 to 1)
    simd::float4x4 proj = simd_matrix(
        simd_make_float4(f / aspect, 0.0f,  0.0f,                             0.0f),  // Col 0
        simd_make_float4(0.0f,       f,     0.0f,                             0.0f),  // Col 1
        simd_make_float4(0.0f,       0.0f,  farPlane / -zRange,              -1.0f),  // Col 2 (Negated for RH)
        simd_make_float4(0.0f,       0.0f,  -(farPlane * nearPlane) / zRange, 0.0f)   // Col 3
    );

    // Standard Right-Handed Rotations (Original math works perfectly here!)
    float cosX = cosf(0), sinX = sinf(0);
    simd::float4x4 rotX = simd_matrix(
        simd_make_float4(1.0f, 0.0f,  0.0f,  0.0f), // Col 0
        simd_make_float4(0.0f, cosX,  sinX,  0.0f), // Col 1
        simd_make_float4(0.0f, -sinX, cosX,  0.0f), // Col 2
        simd_make_float4(0.0f, 0.0f,  0.0f,  1.0f)  // Col 3
    );

    float cosY = cosf(time * 0.8f), sinY = sinf(time * 0.8f);
    simd::float4x4 rotY = simd_matrix(
        simd_make_float4(cosY,  0.0f, -sinY, 0.0f), // Col 0
        simd_make_float4(0.0f,  1.0f, 0.0f,  0.0f), // Col 1
        simd_make_float4(sinY,  0.0f, cosY,  0.0f), // Col 2
        simd_make_float4(0.0f,  0.0f, 0.0f,  1.0f)  // Col 3
    );

    // Camera looking down -Z means pushing the object forward requires a NEGATIVE Z translation
    simd::float4x4 trans = simd_matrix(
        simd_make_float4(1.0f, 0.0f, 0.0f,   0.0f), // Col 0
        simd_make_float4(0.0f, 1.0f, 0.0f,   0.0f), // Col 1
        simd_make_float4(0.0f, 0.0f, 1.0f,   0.0f), // Col 2
        simd_make_float4(0.0f, 0.0f, -2.5f,  1.0f)  // Col 3 (Move cube into the scene)
    );

    simd::float4x4 model = trans * rotY * rotX;
    u.modelMatrix = model;
    u.mvpMatrix = proj * model;

    // Light orbits around the Y axis, independent of the cube's own rotation
    float lightAngle = time * 0.5f;
    u.lightDirection = simd_make_float4(cosf(lightAngle) * 0.8f, 0.6f, sinf(lightAngle) * 0.8f, 0.0f);

    return u;
}

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

    // Standard Right-Handed Cube: +Z is Front (toward viewer), -Z is Back (away)
    // Each face gets its own 4 vertices (24 total) so every face can have its own 0..1 UV range.
    float cubeVertices[] = {
        // Front (Z = 0.5)               normal
        -0.5f,  0.5f,  0.5f,   0.0f, 1.0f,   0.0f, 0.0f, 1.0f,
        0.5f,  0.5f,  0.5f,   1.0f, 1.0f,   0.0f, 0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,   0.0f, 0.0f,   0.0f, 0.0f, 1.0f,
        0.5f, -0.5f,  0.5f,   1.0f, 0.0f,   0.0f, 0.0f, 1.0f,
        // Back (Z = -0.5)
        0.5f,  0.5f, -0.5f,   0.0f, 1.0f,   0.0f, 0.0f, -1.0f,
        -0.5f,  0.5f, -0.5f,   1.0f, 1.0f,   0.0f, 0.0f, -1.0f,
        0.5f, -0.5f, -0.5f,   0.0f, 0.0f,   0.0f, 0.0f, -1.0f,
        -0.5f, -0.5f, -0.5f,   1.0f, 0.0f,   0.0f, 0.0f, -1.0f,
        // Top (Y = 0.5)
        -0.5f,  0.5f, -0.5f,   0.0f, 1.0f,   0.0f, 1.0f, 0.0f,
        0.5f,  0.5f, -0.5f,   1.0f, 1.0f,   0.0f, 1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,   0.0f, 0.0f,   0.0f, 1.0f, 0.0f,
        0.5f,  0.5f,  0.5f,   1.0f, 0.0f,   0.0f, 1.0f, 0.0f,
        // Bottom (Y = -0.5)
        -0.5f, -0.5f,  0.5f,   0.0f, 1.0f,   0.0f, -1.0f, 0.0f,
        0.5f, -0.5f,  0.5f,   1.0f, 1.0f,   0.0f, -1.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,   0.0f, 0.0f,   0.0f, -1.0f, 0.0f,
        0.5f, -0.5f, -0.5f,   1.0f, 0.0f,   0.0f, -1.0f, 0.0f,
        // Left (X = -0.5)
        -0.5f,  0.5f, -0.5f,   0.0f, 1.0f,   -1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,   1.0f, 1.0f,   -1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,   0.0f, 0.0f,   -1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,   1.0f, 0.0f,   -1.0f, 0.0f, 0.0f,
        // Right (X = 0.5)
        0.5f,  0.5f,  0.5f,   0.0f, 1.0f,   1.0f, 0.0f, 0.0f,
        0.5f,  0.5f, -0.5f,   1.0f, 1.0f,   1.0f, 0.0f, 0.0f,
        0.5f, -0.5f,  0.5f,   0.0f, 0.0f,   1.0f, 0.0f, 0.0f,
        0.5f, -0.5f, -0.5f,   1.0f, 0.0f,   1.0f, 0.0f, 0.0f,
    };

    // 12 triangles mapped Counter-Clockwise (CCW). Each face block is 4 verts: a,b,c,d -> (a,c,d) + (a,d,b)
    uint16_t cubeIndices[] = {
        0, 2, 3,   0, 3, 1,     // Front
        4, 6, 7,   4, 7, 5,     // Back
        8, 10, 11, 8, 11, 9,    // Top
        12, 14, 15, 12, 15, 13, // Bottom
        16, 18, 19, 16, 19, 17, // Left
        20, 22, 23, 20, 23, 21  // Right
    };

    // Create GPU buffers
    MTL::Buffer* vertexBuffer = device->newBuffer(cubeVertices, sizeof(cubeVertices), MTL::ResourceStorageModeShared);
    MTL::Buffer* indexBuffer = device->newBuffer(cubeIndices, sizeof(cubeIndices), MTL::ResourceStorageModeShared);

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
    // Layout stride (8 floats: 3 Position + 2 UV + 3 Normal)
    vertexDesc->layouts()->object(0)->setStride(8 * sizeof(float));

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

    // Load the diffuse texture from disk and upload it into a MTL::Texture
    int texWidth, texHeight, texChannels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* texPixels = stbi_load(
        "../texture/metal_plate_4k/textures/metal_plate_diff_4k.jpg",
        &texWidth, &texHeight, &texChannels, STBI_rgb_alpha
    );
    if (!texPixels) {
        fprintf(stderr, "Failed to load texture: %s\n", stbi_failure_reason());
        return -1;
    }

    MTL::TextureDescriptor* texDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRGBA8Unorm, (NS::UInteger)texWidth, (NS::UInteger)texHeight, false
    );
    texDesc->setStorageMode(MTL::StorageModeShared);
    texDesc->setUsage(MTL::TextureUsageShaderRead);
    MTL::Texture* colorTexture = device->newTexture(texDesc);
    colorTexture->replaceRegion(
        MTL::Region(0, 0, (NS::UInteger)texWidth, (NS::UInteger)texHeight),
        0, texPixels, (NS::UInteger)texWidth * 4
    );
    stbi_image_free(texPixels);

    MTL::SamplerDescriptor* samplerDesc = MTL::SamplerDescriptor::alloc()->init();
    samplerDesc->setMinFilter(MTL::SamplerMinMagFilterLinear);
    samplerDesc->setMagFilter(MTL::SamplerMinMagFilterLinear);
    samplerDesc->setMipFilter(MTL::SamplerMipFilterLinear);
    samplerDesc->setSAddressMode(MTL::SamplerAddressModeRepeat);
    samplerDesc->setTAddressMode(MTL::SamplerAddressModeRepeat);
    MTL::SamplerState* samplerState = device->newSamplerState(samplerDesc);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

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

            // Perspective Projection Matrix
            float currentTime = (float)glfwGetTime();
            Uniforms ppm = perspectiveProjectionRightHanded(currentTime, liveWidth, liveHeight);
            memcpy(uniformBuffer->contents(), &ppm, sizeof(Uniforms));

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
            encoder->setFragmentSamplerState(samplerState, 0);

            // 36 indices total (12 triangles * 3 vertices)
            encoder->drawIndexedPrimitives(
                MTL::PrimitiveTypeTriangle, 
                36, 
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
