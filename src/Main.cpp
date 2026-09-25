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
#include "MeshLoader.hpp"
#include "Scene.hpp"
#include "SceneEditorPanel.hpp"
#include "Shadow.hpp"
#include "Texture.hpp"
#include "UI.hpp"
#include "Uniforms.hpp"

#include <dispatch/dispatch.h>
#include <unordered_map>
#include <vector>

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

    // Scene editor state: load the last-saved scene (see SceneEditorPanel's Save/Load buttons);
    // fall back to one starter cube + one starter light so the viewport isn't empty/unlit if
    // scene.txt doesn't exist yet (e.g. a fresh checkout).
    Scene scene;
    if (!loadScene(scene, kSceneFilePath)) {
        scene.objects.push_back(SceneObject{});
        SceneObject defaultLight;
        defaultLight.type = SceneObjectType::Light;
        defaultLight.name = "Light 1";
        defaultLight.position[0] = 2.0f;
        defaultLight.position[1] = 4.0f;
        defaultLight.position[2] = 2.0f;
        scene.objects.push_back(defaultLight);
    }
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

    // A CAMetalLayer's drawableSize does NOT automatically track the layer's frame as the window
    // resizes (unlike MTKView, which this app doesn't use) - it must be set explicitly, both here
    // and again on every resize below, or nextDrawable() keeps handing back a texture at the old
    // size forever. The window server then stretches that stale-sized surface to fit the window's
    // actual (correctly resized) frame, which is what made the rendered content - and, since
    // ImGui's cursor-to-widget mapping assumes the drawable matches the current window size, the
    // mouse - visibly drift out of sync with the real window size after a resize.
    // glfwGetFramebufferSize (not the width/height window was created with, which are in screen
    // points) gives the real pixel size, matching what a Retina/HiDPI display actually needs.
    glfwGetFramebufferSize(window, &width, &height);
    cppMetalLayer->setDrawableSize(CGSizeMake(width, height));

    // Allocate Depth Buffer based on initial window framebuffer sizing
    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatDepth32Float, (NS::UInteger)width, (NS::UInteger)height, false
    );
    desc->setStorageMode(MTL::StorageModePrivate);
    desc->setUsage(MTL::TextureUsageRenderTarget);
    MTL::Texture* depthTexture = device->newTexture(desc);

    // Shadow cube maps: one 6-face cube texture per light slot, storing that light's distance to
    // the nearest occluder in every direction (see Shader.metal's cubeShadowFragmentMain). All
    // kMaxLights are allocated up front since the shader's fixed-size texture array (see
    // fragmentMain) needs every slot bound, even the unused ones. R32Float (not a depth format)
    // because each face stores a plain world-space distance, sampled by direction later.
    MTL::TextureDescriptor* shadowDesc = MTL::TextureDescriptor::textureCubeDescriptor(
        MTL::PixelFormatR32Float, (NS::UInteger)kShadowMapSize, false
    );
    shadowDesc->setStorageMode(MTL::StorageModePrivate);
    shadowDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
    MTL::Texture* shadowCubeMaps[kMaxLights];
    for (size_t i = 0; i < kMaxLights; i++) {
        shadowCubeMaps[i] = device->newTexture(shadowDesc);
    }

    // Scratch depth buffer for the cube shadow pass's own hidden-surface removal - never sampled
    // afterward, so one shared 2D texture is reused across every face of every light each frame.
    MTL::TextureDescriptor* shadowDepthDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatDepth32Float, (NS::UInteger)kShadowMapSize, (NS::UInteger)kShadowMapSize, false
    );
    shadowDepthDesc->setStorageMode(MTL::StorageModePrivate);
    shadowDepthDesc->setUsage(MTL::TextureUsageRenderTarget);
    MTL::Texture* shadowScratchDepth = device->newTexture(shadowDepthDesc);

    // Single-frustum shadow maps for Directional/Spot lights: one real (sampled) texture per light
    // slot, unlike the cube shadow's scratch depth buffer below - this one *is* what gets sampled
    // back in fragmentMain (see sampleProjectedShadow in Shader.metal), so it can't be shared/
    // reused across lights within a frame. R32Float world-space distance, same scheme as the cube
    // shadow maps and for the same reason - see sampleProjectedShadow's comment. Allocated for all
    // kMaxLights slots for the same fixed-size-texture-array reason as shadowCubeMaps; unused by
    // Point/Area lights.
    MTL::TextureDescriptor* shadow2DDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatR32Float, (NS::UInteger)kShadow2DMapSize, (NS::UInteger)kShadow2DMapSize, false
    );
    shadow2DDesc->setStorageMode(MTL::StorageModePrivate);
    shadow2DDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
    MTL::Texture* shadow2DMaps[kMaxLights];
    for (size_t i = 0; i < kMaxLights; i++) {
        shadow2DMaps[i] = device->newTexture(shadow2DDesc);
    }

    // Scratch depth buffer for the 2D shadow pass's own hidden-surface removal, sized to match
    // shadow2DMaps - never sampled afterward, same role as shadowScratchDepth above but a
    // different size so it can't just reuse that one.
    MTL::TextureDescriptor* shadow2DScratchDepthDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatDepth32Float, (NS::UInteger)kShadow2DMapSize, (NS::UInteger)kShadow2DMapSize, false
    );
    shadow2DScratchDepthDesc->setStorageMode(MTL::StorageModePrivate);
    shadow2DScratchDepthDesc->setUsage(MTL::TextureUsageRenderTarget);
    MTL::Texture* shadow2DScratchDepth = device->newTexture(shadow2DScratchDepthDesc);

    // Create GPU buffers
    MTL::Buffer* vertexBuffer = device->newBuffer(CubeMesh::vertices, sizeof(CubeMesh::vertices), MTL::ResourceStorageModeShared);
    MTL::Buffer* indexBuffer = device->newBuffer(CubeMesh::indices, sizeof(CubeMesh::indices), MTL::ResourceStorageModeShared);
    constexpr NS::UInteger indexCount = sizeof(CubeMesh::indices) / sizeof(CubeMesh::indices[0]);

    // Loaded Mesh-type objects' GPU buffers, keyed by scene file path - populated lazily each
    // frame below as new paths show up in the scene (see the scene editor's "Add Mesh" button).
    std::unordered_map<std::string, MeshData> meshCache;

    // One Uniforms slot per scene object per frame, plus one for the axis gizmo, one per possible
    // light marker, and one per possible light direction ray. 1024-byte stride is Metal's safe
    // alignment for per-draw buffer offsets, rounded up from sizeof(Uniforms) (which now holds up
    // to kMaxLights lights, each with type/direction/area-axis/shadow-matrix data on top of
    // position/color).
    constexpr NS::UInteger kUniformStride = 1024;
    static_assert(sizeof(Uniforms) <= kUniformStride, "Uniforms grew past the reserved per-object stride");
    constexpr NS::UInteger kGizmoUniformOffset = kMaxSceneObjects * kUniformStride;
    constexpr NS::UInteger kLightMarkerUniformOffset = kGizmoUniformOffset + kUniformStride;
    constexpr NS::UInteger kLightRayUniformOffset = kLightMarkerUniformOffset + kMaxLights * kUniformStride;
    constexpr NS::UInteger kTotalUniformSlots = kMaxSceneObjects + 1 + kMaxLights + kMaxLights;

    // The CPU writes this frame's Uniforms straight into mapped memory (ResourceStorageModeShared)
    // while the GPU may still be reading last frame's - or the frame before that's - draw calls out
    // of the same buffer, since commit() below doesn't block the CPU. One buffer per in-flight frame
    // (rotated by frameIndex) plus frameBoundarySemaphore (signaled from each command buffer's
    // completion handler, waited on before reusing that slot) keeps the CPU from overwriting data
    // the GPU hasn't finished consuming yet.
    constexpr int kMaxFramesInFlight = 3;
    MTL::Buffer* uniformBuffers[kMaxFramesInFlight];
    for (int i = 0; i < kMaxFramesInFlight; i++) {
        uniformBuffers[i] = device->newBuffer(kTotalUniformSlots * kUniformStride, MTL::ResourceStorageModeShared);
    }
    dispatch_semaphore_t frameBoundarySemaphore = dispatch_semaphore_create(kMaxFramesInFlight);
    int frameIndex = 0;

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

    // Cube shadow pass PSO: outputs world-space distance-to-light as a color value (see
    // Shader.metal's cubeShadowFragmentMain) plus a scratch depth attachment for hidden-surface
    // removal within the pass.
    MTL::Function* cubeShadowVertFunc = library->newFunction(NS::String::string("cubeShadowVertexMain", NS::UTF8StringEncoding));
    MTL::Function* cubeShadowFragFunc = library->newFunction(NS::String::string("cubeShadowFragmentMain", NS::UTF8StringEncoding));
    MTL::RenderPipelineDescriptor* shadowPipeDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    shadowPipeDesc->setVertexFunction(cubeShadowVertFunc);
    shadowPipeDesc->setFragmentFunction(cubeShadowFragFunc);
    shadowPipeDesc->setVertexDescriptor(vertexDesc);
    shadowPipeDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatR32Float);
    shadowPipeDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
    MTL::RenderPipelineState* shadowPipelineState = device->newRenderPipelineState(shadowPipeDesc, &error);
    // Directional/Spot lights' single-frustum shadow pass reuses this same PSO (cubeShadowVertexMain/
    // cubeShadowFragmentMain don't care whether the target is one cube face or a plain 2D texture -
    // see Shader.metal's comment on sampleProjectedShadow for why it stays a distance encoding).

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

    // Shadow cube sampler: nearest + clamp, since linearly filtering raw (uncompared) distance
    // values would blend distances instead of blending shadow/lit results, giving wrong edges.
    MTL::SamplerDescriptor* shadowSamplerDesc = MTL::SamplerDescriptor::alloc()->init();
    shadowSamplerDesc->setMinFilter(MTL::SamplerMinMagFilterNearest);
    shadowSamplerDesc->setMagFilter(MTL::SamplerMinMagFilterNearest);
    shadowSamplerDesc->setSAddressMode(MTL::SamplerAddressModeClampToEdge);
    shadowSamplerDesc->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
    MTL::SamplerState* shadowSamplerState = device->newSamplerState(shadowSamplerDesc);

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

        // Resize before fetching the drawable below, so nextDrawable() hands back a texture
        // already sized to match - the layer's drawableSize needs explicit updates on resize, see
        // the comment where it's first set, above the render loop.
        int liveWidth, liveHeight;
        glfwGetFramebufferSize(window, &liveWidth, &liveHeight);
        if (liveWidth != width || liveHeight != height) {
            depthTexture->release();
            width = liveWidth;
            height = liveHeight;
            cppMetalLayer->setDrawableSize(CGSizeMake(width, height));

            MTL::TextureDescriptor* newDesc = MTL::TextureDescriptor::texture2DDescriptor(
                MTL::PixelFormatDepth32Float, (NS::UInteger)width, (NS::UInteger)height, false
            );
            newDesc->setStorageMode(MTL::StorageModePrivate);
            newDesc->setUsage(MTL::TextureUsageRenderTarget);
            depthTexture = device->newTexture(newDesc);
        }

        // Fetch the canvas
        CA::MetalDrawable* drawable = cppMetalLayer->nextDrawable();

        if (drawable) {
            // Block here (not before nextDrawable() above) so waiting for a free uniform-buffer
            // slot doesn't also hold up drawable acquisition - the two are independent resources.
            // Released by this frame's command buffer completion handler, below.
            dispatch_semaphore_wait(frameBoundarySemaphore, DISPATCH_TIME_FOREVER);
            MTL::Buffer* uniformBuffer = uniformBuffers[frameIndex];

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

            // Lazily load any newly-referenced Mesh asset - editing the path field or adding a
            // Mesh object in the scene editor just works next frame, no explicit reload plumbing
            // needed between the UI and the renderer.
            for (const auto& obj : scene.objects) {
                if (obj.type != SceneObjectType::Mesh || obj.meshPath.empty()) continue;
                if (meshCache.find(obj.meshPath) == meshCache.end()) {
                    meshCache[obj.meshPath] = loadMesh(device, obj.meshPath);
                }
            }

            // Gather up to kMaxLights lights (of any type) from the scene. lightObjects[i] keeps
            // the originating SceneObject alongside lights[i] (same index) so the marker/ray pass
            // below can read its rotation - only a nullptr for the synthetic fallback light.
            // If there are no lights at all, fall back to a single default so the scene isn't unlit.
            constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
            SceneLight lights[kMaxLights];
            const SceneObject* lightObjects[kMaxLights] = {};
            int lightCount = 0;
            for (const auto& obj : scene.objects) {
                if (obj.type != SceneObjectType::Light) continue;
                if (lightCount >= (int)kMaxLights) break;
                SceneLight& light = lights[lightCount];
                light.type = obj.lightType;
                light.position = simd_make_float3(obj.position[0], obj.position[1], obj.position[2]);
                light.direction = objectForward(obj);
                light.right = objectRight(obj);
                light.up = objectUp(obj);
                light.color = simd_make_float3(obj.color[0], obj.color[1], obj.color[2]);
                light.intensity = obj.intensity;
                light.spotCosInner = cosf(obj.spotInnerDegrees * kDegToRad);
                light.spotCosOuter = cosf(obj.spotOuterDegrees * kDegToRad);
                light.areaHalfSize = simd_make_float2(obj.areaSize[0] * 0.5f, obj.areaSize[1] * 0.5f);
                if (light.type == LightType::Directional) {
                    light.shadowViewProj = computeDirectionalShadowMatrix(light.position, light.direction,
                                                                           light.right, light.up);
                } else if (light.type == LightType::Spot) {
                    light.shadowViewProj = computeSpotShadowMatrix(light.position, light.direction,
                                                                     light.right, light.up, obj.spotOuterDegrees,
                                                                     kShadowNearPlane, kShadowFarPlane);
                }
                lightObjects[lightCount] = &obj;
                lightCount++;
            }
            if (lightCount == 0) {
                lights[0] = SceneLight{};
                lights[0].position = simd_make_float3(2.0f, 4.0f, 2.0f);
                lights[0].color = simd_make_float3(1.0f, 1.0f, 1.0f);
                lights[0].intensity = 3.0f;
                lightCount = 1;
            }

            // Every active Point light gets its own 6-face cube shadow map (see Shadow.hpp).
            // Other light types don't cast shadows yet (see Shader.metal's fragmentMain).
            simd::float4x4 cubeFaceMatrices[kMaxLights][kCubeFaceCount];
            for (int i = 0; i < lightCount; i++) {
                if (lights[i].type != LightType::Point) continue;
                computeCubeShadowMatrices(lights[i].position, kShadowNearPlane, kShadowFarPlane, cubeFaceMatrices[i]);
            }

            // One entry per drawable (non-Light) scene object this frame - Cube always, Mesh only
            // once its file has successfully loaded into meshCache. Built once and reused by the
            // uniform-fill loop below and by every draw pass further down so their indexing (and
            // thus each object's uniform-buffer slot) always lines up.
            struct RenderableObject {
                const SceneObject* obj;
                MTL::Buffer* vertexBuffer;
                MTL::Buffer* indexBuffer;
                NS::UInteger indexCount;
                MTL::IndexType indexType;
            };
            std::vector<RenderableObject> renderables;
            for (const auto& obj : scene.objects) {
                if (renderables.size() >= kMaxSceneObjects) break;
                if (obj.type == SceneObjectType::Cube) {
                    renderables.push_back({&obj, vertexBuffer, indexBuffer, indexCount, MTL::IndexTypeUInt16});
                } else if (obj.type == SceneObjectType::Mesh && !obj.meshPath.empty()) {
                    auto it = meshCache.find(obj.meshPath);
                    if (it != meshCache.end() && it->second.indexCount > 0) {
                        const MeshData& m = it->second;
                        renderables.push_back({&obj, m.vertexBuffer, m.indexBuffer, m.indexCount, m.indexType});
                    }
                }
            }

            // Write every drawn object's Uniforms into its own aligned slot before any draw call
            // touches the buffer - drawIndexedPrimitives only records GPU work, it doesn't execute
            // it yet, so overwriting the same slot before commit would corrupt earlier draws' data.
            for (NS::UInteger i = 0; i < renderables.size(); i++) {
                Uniforms uniforms = computeUniforms(camera, objectModelMatrix(*renderables[i].obj), lights, lightCount,
                                                     liveWidth, liveHeight);
                memcpy((uint8_t*)uniformBuffer->contents() + i * kUniformStride, &uniforms, sizeof(Uniforms));
            }
            if (camera.uiMode) {
                Uniforms gizmoUniforms = computeUniforms(camera, matrix_identity_float4x4, lights, lightCount,
                                                          liveWidth, liveHeight);
                memcpy((uint8_t*)uniformBuffer->contents() + kGizmoUniformOffset, &gizmoUniforms, sizeof(Uniforms));

                // Light markers: a small translate-only model matrix places the marker at each light.
                // Non-Point lights also get a direction ray, built from the object's full rotation +
                // translation so it points along the light's actual orientation.
                for (int i = 0; i < lightCount; i++) {
                    simd::float4x4 markerModel = simd_matrix(
                        simd_make_float4(1.0f, 0.0f, 0.0f, 0.0f),
                        simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f),
                        simd_make_float4(0.0f, 0.0f, 1.0f, 0.0f),
                        simd_make_float4(lights[i].position.x, lights[i].position.y, lights[i].position.z, 1.0f)
                    );
                    Uniforms markerUniforms = computeUniforms(camera, markerModel, lights, lightCount,
                                                               liveWidth, liveHeight);
                    memcpy((uint8_t*)uniformBuffer->contents() + kLightMarkerUniformOffset + i * kUniformStride,
                           &markerUniforms, sizeof(Uniforms));

                    if (lightObjects[i] && lights[i].type != LightType::Point) {
                        Uniforms rayUniforms = computeUniforms(camera, objectModelMatrix(*lightObjects[i]),
                                                                lights, lightCount, liveWidth, liveHeight);
                        memcpy((uint8_t*)uniformBuffer->contents() + kLightRayUniformOffset + i * kUniformStride,
                               &rayUniforms, sizeof(Uniforms));
                    }
                }
            }

            // Request a command buffer from the queue
            MTL::CommandBuffer* cmdBuffer = cmdQueue->commandBuffer();

            // Shadow pass: render every cube's depth/distance-from-light into each active light's
            // shadow map(s), before the main color pass that will sample them all. Point lights
            // get 6 encoders (one per cube face, see the earlier conversation on instanced/layered
            // rendering for how a production renderer collapses this to one encoder per light;
            // this stays the simple, unoptimized version for now); Directional/Spot get one
            // encoder each into their single-frustum depth map; Area lights don't cast shadows yet.
            for (int lightIndex = 0; lightIndex < lightCount; lightIndex++) {
                LightType lightType = lights[lightIndex].type;
                if (lightType == LightType::Point) {
                    for (int face = 0; face < kCubeFaceCount; face++) {
                        MTL::RenderPassDescriptor* shadowRPD = MTL::RenderPassDescriptor::renderPassDescriptor();
                        auto shadowColor = shadowRPD->colorAttachments()->object(0);
                        shadowColor->setTexture(shadowCubeMaps[lightIndex]);
                        shadowColor->setSlice(face);
                        shadowColor->setLoadAction(MTL::LoadActionClear);
                        shadowColor->setClearColor({(double)kShadowFarPlane, (double)kShadowFarPlane, (double)kShadowFarPlane, 1.0});
                        shadowColor->setStoreAction(MTL::StoreActionStore);
                        shadowRPD->depthAttachment()->setTexture(shadowScratchDepth);
                        shadowRPD->depthAttachment()->setLoadAction(MTL::LoadActionClear);
                        shadowRPD->depthAttachment()->setClearDepth(1.0);
                        shadowRPD->depthAttachment()->setStoreAction(MTL::StoreActionDontCare);

                        MTL::RenderCommandEncoder* shadowEncoder = cmdBuffer->renderCommandEncoder(shadowRPD);
                        shadowEncoder->setDepthStencilState(depthState);
                        shadowEncoder->setRenderPipelineState(shadowPipelineState);
                        shadowEncoder->setVertexBytes(&cubeFaceMatrices[lightIndex][face], sizeof(simd::float4x4), 2);
                        shadowEncoder->setFragmentBytes(&lights[lightIndex].position, sizeof(simd::float3), 3);
                        for (NS::UInteger i = 0; i < renderables.size(); i++) {
                            const auto& r = renderables[i];
                            shadowEncoder->setVertexBuffer(r.vertexBuffer, 0, 0);
                            shadowEncoder->setVertexBuffer(uniformBuffer, i * kUniformStride, 1);
                            shadowEncoder->drawIndexedPrimitives(
                                MTL::PrimitiveTypeTriangle, r.indexCount, r.indexType, r.indexBuffer, 0
                            );
                        }
                        shadowEncoder->endEncoding();
                    }
                } else if (lightType == LightType::Directional || lightType == LightType::Spot) {
                    // The distance reference point the shadow pass renders from - must match what
                    // sampleProjectedShadow in Shader.metal uses at sampling time. Spot has a true
                    // position; Directional's virtual eye is rederived there too (see its comment).
                    simd::float3 shadowEye = (lightType == LightType::Directional)
                        ? lights[lightIndex].position - lights[lightIndex].direction * kDirectionalShadowDistance
                        : lights[lightIndex].position;

                    MTL::RenderPassDescriptor* shadow2DRPD = MTL::RenderPassDescriptor::renderPassDescriptor();
                    auto shadow2DColor = shadow2DRPD->colorAttachments()->object(0);
                    shadow2DColor->setTexture(shadow2DMaps[lightIndex]);
                    shadow2DColor->setLoadAction(MTL::LoadActionClear);
                    shadow2DColor->setClearColor({1000.0, 1000.0, 1000.0, 1.0});
                    shadow2DColor->setStoreAction(MTL::StoreActionStore);
                    shadow2DRPD->depthAttachment()->setTexture(shadow2DScratchDepth);
                    shadow2DRPD->depthAttachment()->setLoadAction(MTL::LoadActionClear);
                    shadow2DRPD->depthAttachment()->setClearDepth(1.0);
                    shadow2DRPD->depthAttachment()->setStoreAction(MTL::StoreActionDontCare);

                    MTL::RenderCommandEncoder* shadow2DEncoder = cmdBuffer->renderCommandEncoder(shadow2DRPD);
                    shadow2DEncoder->setDepthStencilState(depthState);
                    shadow2DEncoder->setRenderPipelineState(shadowPipelineState);
                    shadow2DEncoder->setVertexBytes(&lights[lightIndex].shadowViewProj, sizeof(simd::float4x4), 2);
                    shadow2DEncoder->setFragmentBytes(&shadowEye, sizeof(simd::float3), 3);
                    for (NS::UInteger i = 0; i < renderables.size(); i++) {
                        const auto& r = renderables[i];
                        shadow2DEncoder->setVertexBuffer(r.vertexBuffer, 0, 0);
                        shadow2DEncoder->setVertexBuffer(uniformBuffer, i * kUniformStride, 1);
                        shadow2DEncoder->drawIndexedPrimitives(
                            MTL::PrimitiveTypeTriangle, r.indexCount, r.indexType, r.indexBuffer, 0
                        );
                    }
                    shadow2DEncoder->endEncoding();
                }
            }

            // Create the encoder
            MTL::RenderCommandEncoder* encoder = cmdBuffer->renderCommandEncoder(rpd);

            // Issue drawing commands
            encoder->setDepthStencilState(depthState);

            if (camera.uiMode) {
                drawAxisGizmo(axisGizmo, encoder, uniformBuffer, kGizmoUniformOffset);
                for (int i = 0; i < lightCount; i++) {
                    drawLightMarker(lightMarker, encoder, uniformBuffer, kLightMarkerUniformOffset + i * kUniformStride);
                    if (lightObjects[i] && lights[i].type != LightType::Point) {
                        drawLightDirectionRay(lightMarker, encoder, uniformBuffer, kLightRayUniformOffset + i * kUniformStride);
                    }
                }
            }

            encoder->setRenderPipelineState(pipelineState);
            encoder->setFragmentTexture(colorTexture, 0);
            encoder->setFragmentTexture(normalTexture, 1);
            for (size_t i = 0; i < kMaxLights; i++) {
                encoder->setFragmentTexture(shadowCubeMaps[i], 2 + i);
            }
            for (size_t i = 0; i < kMaxLights; i++) {
                encoder->setFragmentTexture(shadow2DMaps[i], 2 + kMaxLights + i);
            }
            encoder->setFragmentSamplerState(samplerState, 0);
            encoder->setFragmentSamplerState(shadowSamplerState, 1);

            for (NS::UInteger i = 0; i < renderables.size(); i++) {
                const auto& r = renderables[i];
                NS::UInteger offset = i * kUniformStride;
                encoder->setVertexBuffer(r.vertexBuffer, 0, 0);
                encoder->setVertexBuffer(uniformBuffer, offset, 1);
                encoder->setFragmentBuffer(uniformBuffer, offset, 1);
                encoder->drawIndexedPrimitives(
                    MTL::PrimitiveTypeTriangle,
                    r.indexCount,
                    r.indexType,
                    r.indexBuffer,
                    0
                );
            }

            endUIFrame(cmdBuffer, encoder);

            // Free this frame's uniform-buffer slot for reuse once the GPU actually finishes
            // reading it, kMaxFramesInFlight frames from now.
            cmdBuffer->addCompletedHandler([frameBoundarySemaphore](MTL::CommandBuffer*) {
                dispatch_semaphore_signal(frameBoundarySemaphore);
            });

            // Present the texture onto the screen and submit to the GPU
            encoder->endEncoding();
            cmdBuffer->presentDrawable(drawable);
            cmdBuffer->commit();

            frameIndex = (frameIndex + 1) % kMaxFramesInFlight;
        }

        // Drain the temporary memory pool for this frame
        framePool->release();
    }

    shutdownUI();

    // Explicit GPU Cleanups
    releaseAxisGizmo(axisGizmo);
    releaseLightMarker(lightMarker);
    shadowSamplerState->release();
    shadowSamplerDesc->release();
    samplerState->release();
    samplerDesc->release();
    colorTexture->release();
    normalTexture->release();
    for (size_t i = 0; i < kMaxLights; i++) {
        shadowCubeMaps[i]->release();
    }
    for (size_t i = 0; i < kMaxLights; i++) {
        shadow2DMaps[i]->release();
    }
    shadowScratchDepth->release();
    shadow2DScratchDepth->release();
    depthTexture->release();
    depthState->release();
    depthDesc->release();
    shadowPipelineState->release();
    shadowPipeDesc->release();
    cubeShadowVertFunc->release();
    cubeShadowFragFunc->release();
    pipelineState->release();
    pipeDesc->release();
    vertFunc->release();
    fragFunc->release();
    library->release();
    vertexBuffer->release();
    indexBuffer->release();
    for (auto& entry : meshCache) {
        if (entry.second.vertexBuffer) entry.second.vertexBuffer->release();
        if (entry.second.indexBuffer) entry.second.indexBuffer->release();
    }
    // Drain every in-flight frame before releasing the buffers it may still be reading.
    for (int i = 0; i < kMaxFramesInFlight; i++) {
        dispatch_semaphore_wait(frameBoundarySemaphore, DISPATCH_TIME_FOREVER);
    }
    for (int i = 0; i < kMaxFramesInFlight; i++) {
        uniformBuffers[i]->release();
    }
    cmdQueue->release();
    device->release();
    pool->release();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
