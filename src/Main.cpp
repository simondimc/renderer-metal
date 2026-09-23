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
    simd::float4 cameraPosition;
};

struct Camera {
    simd::float3 position = simd_make_float3(0.0f, 0.0f, 2.5f);
    float yaw = 0.0f;   // radians, 0 = looking down -Z
    float pitch = 0.0f; // radians, clamped to avoid gimbal flip
    float moveSpeed = 2.5f;
    float mouseSensitivity = 0.0025f;
    bool firstMouse = true;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
};

// Forward direction derived from yaw/pitch; 0,0 points down -Z to match the rest of the renderer
simd::float3 cameraFront(const Camera& cam) {
    return simd_normalize(simd_make_float3(
        sinf(cam.yaw) * cosf(cam.pitch),
        sinf(cam.pitch),
        -cosf(cam.yaw) * cosf(cam.pitch)
    ));
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    Camera* cam = static_cast<Camera*>(glfwGetWindowUserPointer(window));
    if (cam->firstMouse) {
        cam->lastMouseX = xpos;
        cam->lastMouseY = ypos;
        cam->firstMouse = false;
    }

    double dx = xpos - cam->lastMouseX;
    double dy = cam->lastMouseY - ypos; // reversed: screen Y grows downward
    cam->lastMouseX = xpos;
    cam->lastMouseY = ypos;

    cam->yaw += (float)dx * cam->mouseSensitivity;
    cam->pitch += (float)dy * cam->mouseSensitivity;

    const float maxPitch = 1.5533f; // ~89 degrees
    cam->pitch = fmaxf(-maxPitch, fminf(maxPitch, cam->pitch));
}

void processCameraInput(GLFWwindow* window, Camera& cam, float deltaTime) {
    simd::float3 front = cameraFront(cam);
    simd::float3 worldUp = simd_make_float3(0.0f, 1.0f, 0.0f);
    simd::float3 right = simd_normalize(simd_cross(front, worldUp));

    float velocity = cam.moveSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) cam.position += front * velocity;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cam.position -= front * velocity;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) cam.position -= right * velocity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) cam.position += right * velocity;
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) cam.position += worldUp * velocity;
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) cam.position -= worldUp * velocity;
}

// Standard lookAt view matrix built from the camera's own right/up/front basis
simd::float4x4 viewMatrix(const Camera& cam) {
    simd::float3 front = cameraFront(cam);
    simd::float3 worldUp = simd_make_float3(0.0f, 1.0f, 0.0f);
    simd::float3 right = simd_normalize(simd_cross(front, worldUp));
    simd::float3 up = simd_cross(right, front);

    return simd_matrix(
        simd_make_float4(right.x, up.x, -front.x, 0.0f),
        simd_make_float4(right.y, up.y, -front.y, 0.0f),
        simd_make_float4(right.z, up.z, -front.z, 0.0f),
        simd_make_float4(-simd_dot(right, cam.position), -simd_dot(up, cam.position), simd_dot(front, cam.position), 1.0f)
    );
}

Uniforms computeUniforms(const Camera& cam, float cubeTime, int width, int height) {
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

    float cosY = cosf(cubeTime * 0.8f), sinY = sinf(cubeTime * 0.8f);
    simd::float4x4 rotY = simd_matrix(
        simd_make_float4(cosY,  0.0f, -sinY, 0.0f), // Col 0
        simd_make_float4(0.0f,  1.0f, 0.0f,  0.0f), // Col 1
        simd_make_float4(sinY,  0.0f, cosY,  0.0f), // Col 2
        simd_make_float4(0.0f,  0.0f, 0.0f,  1.0f)  // Col 3
    );

    // Cube stays at the origin and only spins in place; the camera now handles scene distance/movement
    simd::float4x4 model = rotY * rotX;
    simd::float4x4 view = viewMatrix(cam);
    u.modelMatrix = model;
    u.mvpMatrix = proj * view * model;
    u.cameraPosition = simd_make_float4(cam.position.x, cam.position.y, cam.position.z, 1.0f);

    u.lightDirection = simd_make_float4(0.4f, 0.8f, 0.5f, 0.0f);

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

    // Standard Right-Handed Cube: +Z is Front (toward viewer), -Z is Back (away)
    // Each face gets its own 4 vertices (24 total) so every face can have its own 0..1 UV range.
    // Tangent = world-space direction of increasing U (needed to build the TBN basis for normal mapping)
    float cubeVertices[] = {
        // Front (Z = 0.5)               normal              tangent
        -0.5f,  0.5f,  0.5f,   0.0f, 1.0f,   0.0f, 0.0f, 1.0f,   1.0f, 0.0f, 0.0f,
        0.5f,  0.5f,  0.5f,   1.0f, 1.0f,   0.0f, 0.0f, 1.0f,   1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,   0.0f, 0.0f,   0.0f, 0.0f, 1.0f,   1.0f, 0.0f, 0.0f,
        0.5f, -0.5f,  0.5f,   1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   1.0f, 0.0f, 0.0f,
        // Back (Z = -0.5)
        0.5f,  0.5f, -0.5f,   0.0f, 1.0f,   0.0f, 0.0f, -1.0f,  -1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,   1.0f, 1.0f,   0.0f, 0.0f, -1.0f,  -1.0f, 0.0f, 0.0f,
        0.5f, -0.5f, -0.5f,   0.0f, 0.0f,   0.0f, 0.0f, -1.0f,  -1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,   1.0f, 0.0f,   0.0f, 0.0f, -1.0f,  -1.0f, 0.0f, 0.0f,
        // Top (Y = 0.5)
        -0.5f,  0.5f, -0.5f,   0.0f, 1.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, 0.0f,
        0.5f,  0.5f, -0.5f,   1.0f, 1.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,   0.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, 0.0f,
        0.5f,  0.5f,  0.5f,   1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, 0.0f,
        // Bottom (Y = -0.5)
        -0.5f, -0.5f,  0.5f,   0.0f, 1.0f,   0.0f, -1.0f, 0.0f,  1.0f, 0.0f, 0.0f,
        0.5f, -0.5f,  0.5f,   1.0f, 1.0f,   0.0f, -1.0f, 0.0f,  1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,   0.0f, 0.0f,   0.0f, -1.0f, 0.0f,  1.0f, 0.0f, 0.0f,
        0.5f, -0.5f, -0.5f,   1.0f, 0.0f,   0.0f, -1.0f, 0.0f,  1.0f, 0.0f, 0.0f,
        // Left (X = -0.5)
        -0.5f,  0.5f, -0.5f,   0.0f, 1.0f,   -1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,   1.0f, 1.0f,   -1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,   0.0f, 0.0f,   -1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,   1.0f, 0.0f,   -1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f,
        // Right (X = 0.5)
        0.5f,  0.5f,  0.5f,   0.0f, 1.0f,   1.0f, 0.0f, 0.0f,   0.0f, 0.0f, -1.0f,
        0.5f,  0.5f, -0.5f,   1.0f, 1.0f,   1.0f, 0.0f, 0.0f,   0.0f, 0.0f, -1.0f,
        0.5f, -0.5f,  0.5f,   0.0f, 0.0f,   1.0f, 0.0f, 0.0f,   0.0f, 0.0f, -1.0f,
        0.5f, -0.5f, -0.5f,   1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   0.0f, 0.0f, -1.0f,
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
    // Tangent attribute (Offset matches 3 Position + 2 UV + 3 Normal floats)
    vertexDesc->attributes()->object(3)->setFormat(MTL::VertexFormatFloat3);
    vertexDesc->attributes()->object(3)->setOffset(8 * sizeof(float));
    vertexDesc->attributes()->object(3)->setBufferIndex(0);
    // Layout stride (11 floats: 3 Position + 2 UV + 3 Normal + 3 Tangent)
    vertexDesc->layouts()->object(0)->setStride(11 * sizeof(float));

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

    // Loads an image from disk (via stb_image) and uploads it into a MTL::Texture
    stbi_set_flip_vertically_on_load(true);
    auto loadTexture = [device](const char* path) -> MTL::Texture* {
        int w, h, channels;
        unsigned char* pixels = stbi_load(path, &w, &h, &channels, STBI_rgb_alpha);
        if (!pixels) {
            fprintf(stderr, "Failed to load texture %s: %s\n", path, stbi_failure_reason());
            return nullptr;
        }
        MTL::TextureDescriptor* desc = MTL::TextureDescriptor::texture2DDescriptor(
            MTL::PixelFormatRGBA8Unorm, (NS::UInteger)w, (NS::UInteger)h, false
        );
        desc->setStorageMode(MTL::StorageModeShared);
        desc->setUsage(MTL::TextureUsageShaderRead);
        MTL::Texture* texture = device->newTexture(desc);
        texture->replaceRegion(MTL::Region(0, 0, (NS::UInteger)w, (NS::UInteger)h), 0, pixels, (NS::UInteger)w * 4);
        stbi_image_free(pixels);
        return texture;
    };

    // run.sh/clean_run.sh launch the binary with the build/ directory as cwd
    MTL::Texture* colorTexture = loadTexture("../texture/metal_plate_4k/textures/metal_plate_diff_4k.jpg");
    // Converted offline from the source EXR (DWAA compression, unsupported by stb_image) via ffmpeg
    MTL::Texture* normalTexture = loadTexture("../texture/metal_plate_4k/textures/metal_plate_nor_gl_4k.png");
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
