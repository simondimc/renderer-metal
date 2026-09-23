#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include "AxisGizmo.hpp"
#include "Bridge.hpp"
#include "Camera.hpp"
#include "CubeMesh.hpp"
#include "LightMarker.hpp"
#include "Scene.hpp"
#include "SceneEditorPanel.hpp"
#include "Texture.hpp"
#include "UI.hpp"
#include "Uniforms.hpp"

int main() {
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    int width = 1920;
    int height = 1080;
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
    // camera.uiMode (Tab toggles it) switches between fly-camera and scene-editor mouse focus

    // Scene editor state: one starter cube + one starter light so the viewport isn't empty/unlit
    Scene scene;
    scene.objects.push_back(SceneObject{});
    SceneObject defaultLight;
    defaultLight.type = SceneObjectType::Light;
    defaultLight.name = "Light 1";
    defaultLight.position[0] = 2.0f;
    defaultLight.position[1] = 4.0f;
    defaultLight.position[2] = 2.0f;
    scene.objects.push_back(defaultLight);
    int selectedObjectIndex = 0;

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MTL::CommandQueue* cmdQueue = device->newCommandQueue();
    initUI(window, device);

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

    // One Uniforms slot per scene object per frame, plus one for the axis gizmo and one per
    // possible light marker. 384-byte stride is Metal's safe alignment for per-draw buffer
    // offsets, rounded up from sizeof(Uniforms) (which now holds up to kMaxLights lights).
    constexpr NS::UInteger kUniformStride = 384;
    static_assert(sizeof(Uniforms) <= kUniformStride, "Uniforms grew past the reserved per-object stride");
    constexpr NS::UInteger kGizmoUniformOffset = kMaxSceneObjects * kUniformStride;
    constexpr NS::UInteger kLightMarkerUniformOffset = kGizmoUniformOffset + kUniformStride;
    constexpr NS::UInteger kTotalUniformSlots = kMaxSceneObjects + 1 + kMaxLights;
    MTL::Buffer* uniformBuffer = device->newBuffer(kTotalUniformSlots * kUniformStride, MTL::ResourceStorageModeShared);

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

    AxisGizmo axisGizmo = createAxisGizmo(device, library);
    LightMarker lightMarker = createLightMarker(device, library);

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
    bool toggleKeyWasPressed = false;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        float currentTime = (float)glfwGetTime();
        float deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;

        // Cmd+S toggles edit mode. Not Tab: ImGui reserves Tab for keyboard navigation between
        // widgets (e.g. cycling the 3 fields of a DragFloat3), so Tab can't double as our toggle.
        bool cmdHeld = glfwGetKey(window, GLFW_KEY_LEFT_SUPER) == GLFW_PRESS
                     || glfwGetKey(window, GLFW_KEY_RIGHT_SUPER) == GLFW_PRESS;
        bool toggleKeyIsPressed = cmdHeld && glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
        if (toggleKeyIsPressed && !toggleKeyWasPressed) {
            camera.uiMode = !camera.uiMode;
            glfwSetInputMode(window, GLFW_CURSOR, camera.uiMode ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
            camera.firstMouse = true; // avoid a big look-jump when re-capturing the cursor
            if (camera.uiMode) {
                // GLFW_CURSOR_DISABLED reports an unbounded virtual position (for FPS look) that's
                // decoupled from the real (hidden, frozen) OS cursor - re-center both on the way out
                // so the visible cursor and ImGui's idea of its position agree again.
                int winWidth, winHeight;
                glfwGetWindowSize(window, &winWidth, &winHeight);
                glfwSetCursorPos(window, winWidth / 2.0, winHeight / 2.0);
            }
        }
        toggleKeyWasPressed = toggleKeyIsPressed;

        if (!camera.uiMode) {
            processCameraInput(window, camera, deltaTime);
        }

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

            beginUIFrame(rpd);
            if (camera.uiMode) {
                drawSceneEditorPanel(scene, selectedObjectIndex);
            }

            // Gather up to kMaxLights point lights from the scene. If there are none, fall back to
            // a single default so the scene isn't unlit.
            PointLight lights[kMaxLights];
            int lightCount = 0;
            for (const auto& obj : scene.objects) {
                if (obj.type != SceneObjectType::Light) continue;
                if (lightCount >= (int)kMaxLights) break;
                lights[lightCount].position = simd_make_float3(obj.position[0], obj.position[1], obj.position[2]);
                lights[lightCount].color = simd_make_float3(obj.color[0], obj.color[1], obj.color[2]);
                lights[lightCount].intensity = obj.intensity;
                lightCount++;
            }
            if (lightCount == 0) {
                lights[0] = {simd_make_float3(2.0f, 4.0f, 2.0f), simd_make_float3(1.0f, 1.0f, 1.0f), 3.0f};
                lightCount = 1;
            }

            // Write every drawn cube's Uniforms into its own aligned slot before any draw call
            // touches the buffer - drawIndexedPrimitives only records GPU work, it doesn't execute
            // it yet, so overwriting the same slot before commit would corrupt earlier draws' data.
            NS::UInteger cubeCount = 0;
            for (const auto& obj : scene.objects) {
                if (obj.type != SceneObjectType::Cube) continue;
                if (cubeCount >= kMaxSceneObjects) break;
                Uniforms uniforms = computeUniforms(camera, objectModelMatrix(obj),
                                                     lights, lightCount, liveWidth, liveHeight);
                memcpy((uint8_t*)uniformBuffer->contents() + cubeCount * kUniformStride, &uniforms, sizeof(Uniforms));
                cubeCount++;
            }
            if (camera.uiMode) {
                Uniforms gizmoUniforms = computeUniforms(camera, matrix_identity_float4x4,
                                                          lights, lightCount, liveWidth, liveHeight);
                memcpy((uint8_t*)uniformBuffer->contents() + kGizmoUniformOffset, &gizmoUniforms, sizeof(Uniforms));

                // Light markers: a small translate-only model matrix places the marker at each light
                for (int i = 0; i < lightCount; i++) {
                    simd::float4x4 markerModel = simd_matrix(
                        simd_make_float4(1.0f, 0.0f, 0.0f, 0.0f),
                        simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f),
                        simd_make_float4(0.0f, 0.0f, 1.0f, 0.0f),
                        simd_make_float4(lights[i].position.x, lights[i].position.y, lights[i].position.z, 1.0f)
                    );
                    Uniforms markerUniforms = computeUniforms(camera, markerModel, lights, lightCount, liveWidth, liveHeight);
                    memcpy((uint8_t*)uniformBuffer->contents() + kLightMarkerUniformOffset + i * kUniformStride,
                           &markerUniforms, sizeof(Uniforms));
                }
            }

            // Request a command buffer from the queue
            MTL::CommandBuffer* cmdBuffer = cmdQueue->commandBuffer();

            // Create the encoder
            MTL::RenderCommandEncoder* encoder = cmdBuffer->renderCommandEncoder(rpd);

            // Issue drawing commands
            encoder->setDepthStencilState(depthState);

            if (camera.uiMode) {
                drawAxisGizmo(axisGizmo, encoder, uniformBuffer, kGizmoUniformOffset);
                for (int i = 0; i < lightCount; i++) {
                    drawLightMarker(lightMarker, encoder, uniformBuffer, kLightMarkerUniformOffset + i * kUniformStride);
                }
            }

            encoder->setRenderPipelineState(pipelineState);
            encoder->setVertexBuffer(vertexBuffer, 0, 0);
            encoder->setFragmentTexture(colorTexture, 0);
            encoder->setFragmentTexture(normalTexture, 1);
            encoder->setFragmentSamplerState(samplerState, 0);

            for (NS::UInteger i = 0; i < cubeCount; i++) {
                NS::UInteger offset = i * kUniformStride;
                encoder->setVertexBuffer(uniformBuffer, offset, 1);
                encoder->setFragmentBuffer(uniformBuffer, offset, 1);
                encoder->drawIndexedPrimitives(
                    MTL::PrimitiveTypeTriangle,
                    indexCount,
                    MTL::IndexTypeUInt16,
                    indexBuffer,
                    0
                );
            }

            endUIFrame(cmdBuffer, encoder);

            // Present the texture onto the screen and submit to the GPU
            encoder->endEncoding();
            cmdBuffer->presentDrawable(drawable);
            cmdBuffer->commit();
        }

        // Drain the temporary memory pool for this frame
        framePool->release();
    }

    shutdownUI();

    // Explicit GPU Cleanups
    releaseAxisGizmo(axisGizmo);
    releaseLightMarker(lightMarker);
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
