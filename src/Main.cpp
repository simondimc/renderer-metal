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
#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <dispatch/dispatch.h>
#include <unordered_map>
#include <vector>

// Slab-method ray/AABB intersection, used by the Scene Editor's click-to-select (see main()'s
// picking block). origin/dir are in the same (local) space as boxMin/boxMax - dir need not be
// normalized, since outT is only ever used to reconstruct a hit point (origin + dir*outT), not
// compared against other objects' t values directly. Returns false (outT untouched) on a miss.
static bool rayAABBIntersect(simd::float3 origin, simd::float3 dir, simd::float3 boxMin, simd::float3 boxMax, float& outT) {
    float tMin = 0.0f;
    float tMax = INFINITY;

    if (fabsf(dir.x) < 1e-8f) {
        if (origin.x < boxMin.x || origin.x > boxMax.x) return false;
    } else {
        float t1 = (boxMin.x - origin.x) / dir.x;
        float t2 = (boxMax.x - origin.x) / dir.x;
        if (t1 > t2) std::swap(t1, t2);
        tMin = fmaxf(tMin, t1);
        tMax = fminf(tMax, t2);
        if (tMin > tMax) return false;
    }

    if (fabsf(dir.y) < 1e-8f) {
        if (origin.y < boxMin.y || origin.y > boxMax.y) return false;
    } else {
        float t1 = (boxMin.y - origin.y) / dir.y;
        float t2 = (boxMax.y - origin.y) / dir.y;
        if (t1 > t2) std::swap(t1, t2);
        tMin = fmaxf(tMin, t1);
        tMax = fminf(tMax, t2);
        if (tMin > tMax) return false;
    }

    if (fabsf(dir.z) < 1e-8f) {
        if (origin.z < boxMin.z || origin.z > boxMax.z) return false;
    } else {
        float t1 = (boxMin.z - origin.z) / dir.z;
        float t2 = (boxMax.z - origin.z) / dir.z;
        if (t1 > t2) std::swap(t1, t2);
        tMin = fmaxf(tMin, t1);
        tMax = fminf(tMax, t2);
        if (tMin > tMax) return false;
    }

    outT = tMin;
    return true;
}

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

    // Allocate Depth Buffer based on initial window framebuffer sizing. ShaderRead (on top of
    // RenderTarget) so the post-process pass can sample it back for Depth of Field (see
    // postProcessFragmentMain in Shader.metal).
    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatDepth32Float, (NS::UInteger)width, (NS::UInteger)height, false
    );
    desc->setStorageMode(MTL::StorageModePrivate);
    desc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
    MTL::Texture* depthTexture = device->newTexture(desc);

    // Offscreen HDR color target: the scene pass (see fragmentMain in Shader.metal) writes
    // unbounded linear radiance here instead of straight to the drawable. RGBA16Float (not the
    // drawable's BGRA8Unorm) so values above 1.0 survive intact for the post-process pass
    // (postProcessFragmentMain) to exposure/tone-map/gamma-encode in one place, screen-space,
    // rather than per-object.
    MTL::TextureDescriptor* hdrDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRGBA16Float, (NS::UInteger)width, (NS::UInteger)height, false
    );
    hdrDesc->setStorageMode(MTL::StorageModePrivate);
    hdrDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
    MTL::Texture* hdrColorTexture = device->newTexture(hdrDesc);

    // Bloom ping-pong targets: bright-pass extract writes into A, then a horizontal blur reads A
    // and writes B, then a vertical blur reads B and writes back into A (see Main.cpp's bloom pass
    // chain and bloomExtractFragmentMain/blurFragmentMain in Shader.metal). Half the main HDR
    // resolution - the blur only needs to be a soft, low-frequency glow, not full-res detail, and
    // it's cheaper to extract/blur at quarter the pixel count.
    NS::UInteger bloomWidth = (NS::UInteger)width / 2;
    NS::UInteger bloomHeight = (NS::UInteger)height / 2;
    MTL::TextureDescriptor* bloomDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRGBA16Float, bloomWidth, bloomHeight, false
    );
    bloomDesc->setStorageMode(MTL::StorageModePrivate);
    bloomDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
    MTL::Texture* bloomTextureA = device->newTexture(bloomDesc);
    MTL::Texture* bloomTextureB = device->newTexture(bloomDesc);

    // Selection outline mask: flat-white silhouette of the Scene Editor's currently-selected
    // object (see the selection mask pass and selectionMaskFragmentMain in Shader.metal), edge-
    // detected in the post-process pass to draw an outline. Single-channel, full HDR resolution.
    MTL::TextureDescriptor* selectionMaskDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatR8Unorm, (NS::UInteger)width, (NS::UInteger)height, false
    );
    selectionMaskDesc->setStorageMode(MTL::StorageModePrivate);
    selectionMaskDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
    MTL::Texture* selectionMaskTexture = device->newTexture(selectionMaskDesc);

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

    // Build the Pipeline State Object (PSO). Targets the offscreen HDR buffer (RGBA16Float), not
    // the drawable directly - see fragmentMain's comment and the postProcess PSO below.
    MTL::RenderPipelineDescriptor* pipeDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pipeDesc->setVertexFunction(vertFunc);
    pipeDesc->setFragmentFunction(fragFunc);
    pipeDesc->setVertexDescriptor(vertexDesc);
    pipeDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA16Float);
    pipeDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);

    MTL::RenderPipelineState* pipelineState = device->newRenderPipelineState(pipeDesc, &error);

    // Selection mask PSO: reuses vertFunc/vertexDesc unchanged (same mvpMatrix as the main HDR
    // pass) but a trivial fragment function (selectionMaskFragmentMain) that just writes flat
    // white - see the mask pass in the render loop below and Shader.metal's comment.
    MTL::Function* selectionMaskFragFunc = library->newFunction(NS::String::string("selectionMaskFragmentMain", NS::UTF8StringEncoding));
    MTL::RenderPipelineDescriptor* selectionMaskPipeDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    selectionMaskPipeDesc->setVertexFunction(vertFunc);
    selectionMaskPipeDesc->setFragmentFunction(selectionMaskFragFunc);
    selectionMaskPipeDesc->setVertexDescriptor(vertexDesc);
    selectionMaskPipeDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatR8Unorm);
    selectionMaskPipeDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
    MTL::RenderPipelineState* selectionMaskPipelineState = device->newRenderPipelineState(selectionMaskPipeDesc, &error);

    // Post-process PSO: full-screen pass that resolves hdrColorTexture down to the drawable (see
    // postProcessVertexMain/postProcessFragmentMain in Shader.metal). No vertex descriptor (the
    // vertex function takes no [[stage_in]] input - see its comment) and no depth attachment (it
    // doesn't test/write depth).
    MTL::Function* postProcessVertFunc = library->newFunction(NS::String::string("postProcessVertexMain", NS::UTF8StringEncoding));
    MTL::Function* postProcessFragFunc = library->newFunction(NS::String::string("postProcessFragmentMain", NS::UTF8StringEncoding));
    MTL::RenderPipelineDescriptor* postProcessPipeDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    postProcessPipeDesc->setVertexFunction(postProcessVertFunc);
    postProcessPipeDesc->setFragmentFunction(postProcessFragFunc);
    postProcessPipeDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    MTL::RenderPipelineState* postProcessPipelineState = device->newRenderPipelineState(postProcessPipeDesc, &error);

    // Bloom PSOs: both reuse postProcessVertexMain's full-screen triangle, and both target
    // bloomTextureA/B (RGBA16Float, not the drawable) - see bloomExtractFragmentMain/
    // blurFragmentMain in Shader.metal and the 3-pass bloom chain in the render loop below.
    MTL::Function* bloomExtractFragFunc = library->newFunction(NS::String::string("bloomExtractFragmentMain", NS::UTF8StringEncoding));
    MTL::RenderPipelineDescriptor* bloomExtractPipeDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    bloomExtractPipeDesc->setVertexFunction(postProcessVertFunc);
    bloomExtractPipeDesc->setFragmentFunction(bloomExtractFragFunc);
    bloomExtractPipeDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA16Float);
    MTL::RenderPipelineState* bloomExtractPipelineState = device->newRenderPipelineState(bloomExtractPipeDesc, &error);

    MTL::Function* blurFragFunc = library->newFunction(NS::String::string("blurFragmentMain", NS::UTF8StringEncoding));
    MTL::RenderPipelineDescriptor* blurPipeDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    blurPipeDesc->setVertexFunction(postProcessVertFunc);
    blurPipeDesc->setFragmentFunction(blurFragFunc);
    blurPipeDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA16Float);
    MTL::RenderPipelineState* blurPipelineState = device->newRenderPipelineState(blurPipeDesc, &error);

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

    // Selection mask pass depth state: tests against the scene's depth (so the selected object's
    // mask is correctly occluded by anything already in front of it) but never writes - it must
    // leave depthTexture exactly as the HDR scene pass left it, since the overlay pass right after
    // still depends on those values for the gizmo/light markers' own depth test. LessEqual, not
    // Less: this pass redraws the very same geometry the HDR scene pass (Pass A) already wrote
    // depth for, so a visible fragment's depth here is exactly equal to what's already stored, not
    // less than it - a strict Less would reject every one of the selected object's own fragments
    // and the mask would always come out empty.
    MTL::DepthStencilDescriptor* maskDepthDesc = MTL::DepthStencilDescriptor::alloc()->init();
    maskDepthDesc->setDepthCompareFunction(MTL::CompareFunctionLessEqual);
    maskDepthDesc->setDepthWriteEnabled(false);
    MTL::DepthStencilState* maskDepthState = device->newDepthStencilState(maskDepthDesc);

    // run.sh/clean_run.sh launch the binary with the build/ directory as cwd
    MTL::Texture* colorTexture = loadTexture(device, "../texture/metal_plate_4k/textures/metal_plate_diff_4k.jpg", /*isSRGB=*/true);
    // Converted offline from the source EXR (DWAA compression, unsupported by stb_image) via ffmpeg.
    // Normal maps store linear tangent-space vectors, not color, so this one stays non-sRGB.
    MTL::Texture* normalTexture = loadTexture(device, "../texture/metal_plate_4k/textures/metal_plate_nor_gl_4k.png", /*isSRGB=*/false);
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

    // Post-process sampler: the HDR buffer and the drawable are always the same size (both resized
    // together above), so this is really a 1:1 resolve - clamp avoids sampling garbage at the
    // edges, and linear is a harmless no-op at that exact sample alignment.
    MTL::SamplerDescriptor* postProcessSamplerDesc = MTL::SamplerDescriptor::alloc()->init();
    postProcessSamplerDesc->setMinFilter(MTL::SamplerMinMagFilterLinear);
    postProcessSamplerDesc->setMagFilter(MTL::SamplerMinMagFilterLinear);
    postProcessSamplerDesc->setSAddressMode(MTL::SamplerAddressModeClampToEdge);
    postProcessSamplerDesc->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
    MTL::SamplerState* postProcessSamplerState = device->newSamplerState(postProcessSamplerDesc);

    // Reverse-gamma of the old flat background clear color (0.15) so that, once the post-process
    // pass tone-maps and gamma-encodes it again, the on-screen shade matches what a direct
    // (ungamma'd) clear straight to the drawable used to look like.
    float kBackgroundLinear = powf(0.15f, 2.2f);

    float lastFrameTime = (float)glfwGetTime();
    bool toggleKeyWasPressed = false;
    bool leftMouseWasPressed = false; // edge-detects a click for the Scene Editor's picking, below

    // Last frame's camera view-projection, for motion blur's reprojection (see postParams below).
    // Initialized to this frame's own VP on first use (a few lines into the loop) rather than
    // identity, so frame 1 - before anything has "last frame" data yet - reprojects to a no-op
    // instead of a huge bogus jump.
    simd::float4x4 previousViewProj = matrix_identity_float4x4;
    bool havePreviousViewProj = false;

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
            hdrColorTexture->release();
            bloomTextureA->release();
            bloomTextureB->release();
            selectionMaskTexture->release();
            width = liveWidth;
            height = liveHeight;
            cppMetalLayer->setDrawableSize(CGSizeMake(width, height));

            MTL::TextureDescriptor* newDesc = MTL::TextureDescriptor::texture2DDescriptor(
                MTL::PixelFormatDepth32Float, (NS::UInteger)width, (NS::UInteger)height, false
            );
            newDesc->setStorageMode(MTL::StorageModePrivate);
            newDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
            depthTexture = device->newTexture(newDesc);

            MTL::TextureDescriptor* newHdrDesc = MTL::TextureDescriptor::texture2DDescriptor(
                MTL::PixelFormatRGBA16Float, (NS::UInteger)width, (NS::UInteger)height, false
            );
            newHdrDesc->setStorageMode(MTL::StorageModePrivate);
            newHdrDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
            hdrColorTexture = device->newTexture(newHdrDesc);

            NS::UInteger newBloomWidth = (NS::UInteger)width / 2;
            NS::UInteger newBloomHeight = (NS::UInteger)height / 2;
            MTL::TextureDescriptor* newBloomDesc = MTL::TextureDescriptor::texture2DDescriptor(
                MTL::PixelFormatRGBA16Float, newBloomWidth, newBloomHeight, false
            );
            newBloomDesc->setStorageMode(MTL::StorageModePrivate);
            newBloomDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
            bloomTextureA = device->newTexture(newBloomDesc);
            bloomTextureB = device->newTexture(newBloomDesc);

            MTL::TextureDescriptor* newSelectionMaskDesc = MTL::TextureDescriptor::texture2DDescriptor(
                MTL::PixelFormatR8Unorm, (NS::UInteger)width, (NS::UInteger)height, false
            );
            newSelectionMaskDesc->setStorageMode(MTL::StorageModePrivate);
            newSelectionMaskDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
            selectionMaskTexture = device->newTexture(newSelectionMaskDesc);
        }

        // Fetch the canvas
        CA::MetalDrawable* drawable = cppMetalLayer->nextDrawable();

        if (drawable) {
            // Block here (not before nextDrawable() above) so waiting for a free uniform-buffer
            // slot doesn't also hold up drawable acquisition - the two are independent resources.
            // Released by this frame's command buffer completion handler, below.
            dispatch_semaphore_wait(frameBoundarySemaphore, DISPATCH_TIME_FOREVER);
            MTL::Buffer* uniformBuffer = uniformBuffers[frameIndex];

            // Motion blur's reprojection matrix (see postParams below): computed from this frame's
            // and last frame's bare camera view-projections (no per-object model matrix - see
            // computeViewProj in Uniforms.cpp), before previousViewProj is overwritten for next
            // frame. On the very first frame there's no real "last frame" yet, so it's seeded to
            // this frame's own VP - reprojecting to itself is a no-op rather than a bogus jump.
            simd::float4x4 currentViewProj = computeViewProj(camera, liveWidth, liveHeight);
            if (!havePreviousViewProj) {
                previousViewProj = currentViewProj;
                havePreviousViewProj = true;
            }
            simd::float4x4 reprojectionMatrix = previousViewProj * simd_inverse(currentViewProj);
            previousViewProj = currentViewProj;

            // Overlay pass descriptor (gizmo/light markers/rays + ImGui - see the pass split
            // below): drawn on top of the already-resolved drawable, after the HDR scene pass and
            // the post-process pass have both run, so both attachments here use LoadActionLoad
            // rather than Clear - this pass must preserve what those wrote, not erase it.
            // Built now (rather than down where it's actually used) because beginUIFrame needs it
            // immediately, before any ImGui:: widget calls - see its own comment.
            MTL::RenderPassDescriptor* overlayRPD = MTL::RenderPassDescriptor::renderPassDescriptor();
            auto overlayColorAttachment = overlayRPD->colorAttachments()->object(0);
            overlayColorAttachment->setLoadAction(MTL::LoadActionLoad);
            overlayColorAttachment->setStoreAction(MTL::StoreActionStore);
            overlayColorAttachment->setTexture(drawable->texture());

            // Reuses depthTexture as written by the HDR scene pass, so the gizmo/light markers
            // still depth-test correctly against the scene geometry already drawn there.
            overlayRPD->depthAttachment()->setTexture(depthTexture);
            overlayRPD->depthAttachment()->setLoadAction(MTL::LoadActionLoad);
            overlayRPD->depthAttachment()->setStoreAction(MTL::StoreActionDontCare);

            beginUIFrame(overlayRPD);
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

            // Click-to-select in the 3D viewport: a left-click while in edit mode (and not over an
            // ImGui widget) ray-casts against each renderable's local-space bounding box - Cube's
            // is the known unit box CubeMesh.hpp authors it at, Mesh's comes from MeshData::
            // localMin/Max (see MeshLoader.cpp, computed once at load time). The closest hit (by
            // actual world-space distance, not local-space t, since different objects' scales make
            // local t values incomparable) sets selectedObjectIndex - the same variable the object
            // list in drawSceneEditorPanel already drives, so either selection path highlights the
            // same object via the mask/outline pass further down. A click that hits nothing
            // deselects, matching common editor behavior.
            bool leftMouseIsPressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
            bool leftMouseClicked = leftMouseIsPressed && !leftMouseWasPressed;
            leftMouseWasPressed = leftMouseIsPressed;

            if (leftMouseClicked && camera.uiMode && !ImGui::GetIO().WantCaptureMouse) {
                int winWidth, winHeight;
                glfwGetWindowSize(window, &winWidth, &winHeight);
                double mouseX, mouseY;
                glfwGetCursorPos(window, &mouseX, &mouseY);
                float ndcX = (float)(mouseX / winWidth) * 2.0f - 1.0f;
                float ndcY = 1.0f - (float)(mouseY / winHeight) * 2.0f;

                simd::float4x4 invViewProj = simd_inverse(currentViewProj);
                simd::float4 nearPoint4 = invViewProj * simd_make_float4(ndcX, ndcY, 0.0f, 1.0f);
                simd::float4 farPoint4 = invViewProj * simd_make_float4(ndcX, ndcY, 1.0f, 1.0f);
                simd::float3 rayOrigin = simd_make_float3(nearPoint4.x, nearPoint4.y, nearPoint4.z) / nearPoint4.w;
                simd::float3 farPoint = simd_make_float3(farPoint4.x, farPoint4.y, farPoint4.z) / farPoint4.w;
                simd::float3 rayDir = simd_normalize(farPoint - rayOrigin);

                int hitIndex = -1;
                float closestWorldDist = INFINITY;
                for (const auto& r : renderables) {
                    simd::float3 localMin, localMax;
                    if (r.obj->type == SceneObjectType::Cube) {
                        localMin = simd_make_float3(-0.5f, -0.5f, -0.5f);
                        localMax = simd_make_float3(0.5f, 0.5f, 0.5f);
                    } else {
                        auto it = meshCache.find(r.obj->meshPath);
                        if (it == meshCache.end()) continue;
                        localMin = it->second.localMin;
                        localMax = it->second.localMax;
                    }

                    simd::float4x4 modelMatrix = objectModelMatrix(*r.obj);
                    simd::float4x4 invModel = simd_inverse(modelMatrix);
                    simd::float4 localOrigin4 = invModel * simd_make_float4(rayOrigin.x, rayOrigin.y, rayOrigin.z, 1.0f);
                    simd::float4 localDir4 = invModel * simd_make_float4(rayDir.x, rayDir.y, rayDir.z, 0.0f);
                    simd::float3 localOrigin = simd_make_float3(localOrigin4.x, localOrigin4.y, localOrigin4.z);
                    simd::float3 localDir = simd_make_float3(localDir4.x, localDir4.y, localDir4.z);

                    float localT;
                    if (!rayAABBIntersect(localOrigin, localDir, localMin, localMax, localT)) continue;

                    simd::float3 localHit = localOrigin + localDir * localT;
                    simd::float4 worldHit4 = modelMatrix * simd_make_float4(localHit.x, localHit.y, localHit.z, 1.0f);
                    simd::float3 worldHit = simd_make_float3(worldHit4.x, worldHit4.y, worldHit4.z);
                    float worldDist = simd_length(worldHit - rayOrigin);

                    if (worldDist < closestWorldDist) {
                        closestWorldDist = worldDist;
                        hitIndex = (int)(r.obj - scene.objects.data());
                    }
                }
                selectedObjectIndex = hitIndex;
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

            // --- Pass A: HDR scene pass - cube/mesh objects only, lit in linear space, written
            // unclamped into hdrColorTexture (see fragmentMain's comment). depthAttachment uses
            // StoreActionStore (not DontCare, unlike every other depth-only pass here) because the
            // overlay pass below re-reads it to depth-test the gizmo/light markers against this
            // geometry.
            MTL::RenderPassDescriptor* hdrRPD = MTL::RenderPassDescriptor::renderPassDescriptor();
            auto hdrColorAttachment = hdrRPD->colorAttachments()->object(0);
            hdrColorAttachment->setClearColor({(double)kBackgroundLinear, (double)kBackgroundLinear, (double)kBackgroundLinear, 1.0});
            hdrColorAttachment->setLoadAction(MTL::LoadActionClear);
            hdrColorAttachment->setStoreAction(MTL::StoreActionStore);
            hdrColorAttachment->setTexture(hdrColorTexture);
            hdrRPD->depthAttachment()->setTexture(depthTexture);
            hdrRPD->depthAttachment()->setLoadAction(MTL::LoadActionClear);
            hdrRPD->depthAttachment()->setClearDepth(1.0);
            hdrRPD->depthAttachment()->setStoreAction(MTL::StoreActionStore);

            MTL::RenderCommandEncoder* hdrEncoder = cmdBuffer->renderCommandEncoder(hdrRPD);
            hdrEncoder->setDepthStencilState(depthState);
            hdrEncoder->setRenderPipelineState(pipelineState);
            hdrEncoder->setFragmentTexture(colorTexture, 0);
            hdrEncoder->setFragmentTexture(normalTexture, 1);
            for (size_t i = 0; i < kMaxLights; i++) {
                hdrEncoder->setFragmentTexture(shadowCubeMaps[i], 2 + i);
            }
            for (size_t i = 0; i < kMaxLights; i++) {
                hdrEncoder->setFragmentTexture(shadow2DMaps[i], 2 + kMaxLights + i);
            }
            hdrEncoder->setFragmentSamplerState(samplerState, 0);
            hdrEncoder->setFragmentSamplerState(shadowSamplerState, 1);

            for (NS::UInteger i = 0; i < renderables.size(); i++) {
                const auto& r = renderables[i];
                NS::UInteger offset = i * kUniformStride;
                hdrEncoder->setVertexBuffer(r.vertexBuffer, 0, 0);
                hdrEncoder->setVertexBuffer(uniformBuffer, offset, 1);
                hdrEncoder->setFragmentBuffer(uniformBuffer, offset, 1);
                hdrEncoder->drawIndexedPrimitives(
                    MTL::PrimitiveTypeTriangle,
                    r.indexCount,
                    r.indexType,
                    r.indexBuffer,
                    0
                );
            }
            hdrEncoder->endEncoding();

            // --- Selection mask pass: draws just the selected renderable's silhouette (if any)
            // into selectionMaskTexture, depth-tested against the scene depth Pass A just wrote
            // (Load, not Clear) so it's correctly occluded by anything in front of it - see
            // selectionMaskFragmentMain's comment. Only active in edit mode (camera.uiMode),
            // matching every other Scene-Editor-only affordance (gizmo, light markers). Always
            // runs, even with nothing selected, so the mask is freshly cleared to 0 every frame -
            // skipping the pass entirely when unselected would leave stale white silhouette data
            // from whatever WAS selected last, and the outline has no other "off" gate to check.
            MTL::RenderPassDescriptor* selectionMaskRPD = MTL::RenderPassDescriptor::renderPassDescriptor();
            auto selectionMaskColorAttachment = selectionMaskRPD->colorAttachments()->object(0);
            selectionMaskColorAttachment->setClearColor({0.0, 0.0, 0.0, 1.0});
            selectionMaskColorAttachment->setLoadAction(MTL::LoadActionClear);
            selectionMaskColorAttachment->setStoreAction(MTL::StoreActionStore);
            selectionMaskColorAttachment->setTexture(selectionMaskTexture);
            selectionMaskRPD->depthAttachment()->setTexture(depthTexture);
            selectionMaskRPD->depthAttachment()->setLoadAction(MTL::LoadActionLoad);
            selectionMaskRPD->depthAttachment()->setStoreAction(MTL::StoreActionStore);

            MTL::RenderCommandEncoder* selectionMaskEncoder = cmdBuffer->renderCommandEncoder(selectionMaskRPD);
            if (camera.uiMode && selectedObjectIndex >= 0 && selectedObjectIndex < (int)scene.objects.size()) {
                const SceneObject* selectedObj = &scene.objects[selectedObjectIndex];
                for (NS::UInteger i = 0; i < renderables.size(); i++) {
                    if (renderables[i].obj != selectedObj) continue;
                    const auto& r = renderables[i];
                    selectionMaskEncoder->setDepthStencilState(maskDepthState);
                    selectionMaskEncoder->setRenderPipelineState(selectionMaskPipelineState);
                    NS::UInteger offset = i * kUniformStride;
                    selectionMaskEncoder->setVertexBuffer(r.vertexBuffer, 0, 0);
                    selectionMaskEncoder->setVertexBuffer(uniformBuffer, offset, 1);
                    selectionMaskEncoder->drawIndexedPrimitives(
                        MTL::PrimitiveTypeTriangle, r.indexCount, r.indexType, r.indexBuffer, 0
                    );
                    break;
                }
            }
            selectionMaskEncoder->endEncoding();

            // --- Bloom chain: bright-pass extract (hdrColorTexture -> bloomTextureA), then a
            // horizontal blur (A -> B) and a vertical blur (B -> back into A) - see
            // bloomExtractFragmentMain/blurFragmentMain in Shader.metal. Skipped whenever intensity
            // is 0 (the default): bloomTextureA is left holding whatever stale/garbage bloom data
            // it had, but postProcessFragmentMain multiplies it by bloomIntensity before adding it
            // in, so a 0 intensity makes that content irrelevant.
            if (scene.bloomIntensity > 0.0f) {
                MTL::RenderPassDescriptor* bloomExtractRPD = MTL::RenderPassDescriptor::renderPassDescriptor();
                auto bloomExtractColor = bloomExtractRPD->colorAttachments()->object(0);
                bloomExtractColor->setLoadAction(MTL::LoadActionDontCare);
                bloomExtractColor->setStoreAction(MTL::StoreActionStore);
                bloomExtractColor->setTexture(bloomTextureA);

                MTL::RenderCommandEncoder* bloomExtractEncoder = cmdBuffer->renderCommandEncoder(bloomExtractRPD);
                bloomExtractEncoder->setRenderPipelineState(bloomExtractPipelineState);
                bloomExtractEncoder->setFragmentTexture(hdrColorTexture, 0);
                bloomExtractEncoder->setFragmentSamplerState(postProcessSamplerState, 0);
                bloomExtractEncoder->setFragmentBytes(&scene.bloomThreshold, sizeof(float), 0);
                bloomExtractEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, (NS::UInteger)0, (NS::UInteger)3);
                bloomExtractEncoder->endEncoding();

                MTL::RenderPassDescriptor* blurHRPD = MTL::RenderPassDescriptor::renderPassDescriptor();
                auto blurHColor = blurHRPD->colorAttachments()->object(0);
                blurHColor->setLoadAction(MTL::LoadActionDontCare);
                blurHColor->setStoreAction(MTL::StoreActionStore);
                blurHColor->setTexture(bloomTextureB);

                MTL::RenderCommandEncoder* blurHEncoder = cmdBuffer->renderCommandEncoder(blurHRPD);
                blurHEncoder->setRenderPipelineState(blurPipelineState);
                blurHEncoder->setFragmentTexture(bloomTextureA, 0);
                blurHEncoder->setFragmentSamplerState(postProcessSamplerState, 0);
                simd::float2 horizontalDirection = simd_make_float2(1.0f, 0.0f);
                blurHEncoder->setFragmentBytes(&horizontalDirection, sizeof(simd::float2), 0);
                blurHEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, (NS::UInteger)0, (NS::UInteger)3);
                blurHEncoder->endEncoding();

                MTL::RenderPassDescriptor* blurVRPD = MTL::RenderPassDescriptor::renderPassDescriptor();
                auto blurVColor = blurVRPD->colorAttachments()->object(0);
                blurVColor->setLoadAction(MTL::LoadActionDontCare);
                blurVColor->setStoreAction(MTL::StoreActionStore);
                blurVColor->setTexture(bloomTextureA);

                MTL::RenderCommandEncoder* blurVEncoder = cmdBuffer->renderCommandEncoder(blurVRPD);
                blurVEncoder->setRenderPipelineState(blurPipelineState);
                blurVEncoder->setFragmentTexture(bloomTextureB, 0);
                blurVEncoder->setFragmentSamplerState(postProcessSamplerState, 0);
                simd::float2 verticalDirection = simd_make_float2(0.0f, 1.0f);
                blurVEncoder->setFragmentBytes(&verticalDirection, sizeof(simd::float2), 0);
                blurVEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, (NS::UInteger)0, (NS::UInteger)3);
                blurVEncoder->endEncoding();
            }

            // --- Pass B: post-process - resolves hdrColorTexture to the drawable via exposure +
            // tone mapping + gamma encoding, in one full-screen pass (see postProcessFragmentMain).
            // No depth attachment: this pass doesn't test/write depth, it just paints every pixel.
            MTL::RenderPassDescriptor* postRPD = MTL::RenderPassDescriptor::renderPassDescriptor();
            auto postColorAttachment = postRPD->colorAttachments()->object(0);
            postColorAttachment->setLoadAction(MTL::LoadActionDontCare);
            postColorAttachment->setStoreAction(MTL::StoreActionStore);
            postColorAttachment->setTexture(drawable->texture());

            MTL::RenderCommandEncoder* postEncoder = cmdBuffer->renderCommandEncoder(postRPD);
            postEncoder->setRenderPipelineState(postProcessPipelineState);
            postEncoder->setFragmentTexture(hdrColorTexture, 0);
            postEncoder->setFragmentTexture(bloomTextureA, 1);
            postEncoder->setFragmentTexture(depthTexture, 2);
            postEncoder->setFragmentTexture(selectionMaskTexture, 3);
            postEncoder->setFragmentSamplerState(postProcessSamplerState, 0);
            // Nearest + clamp, same as the shadow maps - see postProcessFragmentMain's comment on
            // why Depth of Field can't linearly filter raw depth.
            postEncoder->setFragmentSamplerState(shadowSamplerState, 1);
            // Field order/count must match PostProcessParams in Shader.metal exactly - this is a
            // raw byte copy, not a described/reflected layout. reprojectionMatrix goes first since
            // it's the one field wider than a float (16-byte aligned), so it can't drift out of
            // sync through mismatched padding the way a scalar in the middle could.
            struct {
                simd::float4x4 reprojectionMatrix;
                float exposure;
                float toneMapOperator;
                float vignetteStrength;
                float chromaticAberrationStrength;
                float filmGrainStrength;
                float sharpenStrength;
                float colorGradingSaturation;
                float colorGradingContrast;
                float bloomIntensity;
                float dofFocusDistance;
                float dofFocusRange;
                float dofStrength;
                float motionBlurStrength;
                float lensFlareStrength;
                float time;
            } postParams = {
                reprojectionMatrix,
                scene.exposure, (float)(int)scene.toneMapOperator,
                scene.vignetteStrength, scene.chromaticAberrationStrength,
                scene.filmGrainStrength, scene.sharpenStrength,
                scene.colorGradingSaturation, scene.colorGradingContrast,
                scene.bloomIntensity,
                scene.dofFocusDistance, scene.dofFocusRange, scene.dofStrength,
                scene.motionBlurStrength,
                scene.lensFlareStrength,
                currentTime
            };
            postEncoder->setFragmentBytes(&postParams, sizeof(postParams), 0);

            // Lens flare light data: each active light's screen-space position/depth (projected
            // through this frame's camera, same currentViewProj used for motion blur's
            // reprojectionMatrix above) plus its color - see LensFlareLight's comment in
            // Shader.metal. Field order/count must match that struct exactly, same raw-byte-copy
            // rule as PostProcessParams. Always kMaxLights slots (inactive ones left zeroed, i.e.
            // active=0) for the same fixed-size-array reason as the main pass's shadow map arrays.
            struct LensFlareLightGPU {
                float screenX, screenY, ndcDepth, active, colorR, colorG, colorB, pad;
            };
            LensFlareLightGPU lensFlareLights[kMaxLights] = {};
            if (scene.lensFlareStrength > 0.0f) {
                for (int i = 0; i < lightCount && i < (int)kMaxLights; i++) {
                    simd::float4 clip = currentViewProj * simd_make_float4(
                        lights[i].position.x, lights[i].position.y, lights[i].position.z, 1.0f
                    );
                    if (clip.w <= 0.0f) continue;
                    simd::float3 ndc = simd_make_float3(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
                    if (fabsf(ndc.x) > 1.0f || fabsf(ndc.y) > 1.0f) continue;

                    LensFlareLightGPU& entry = lensFlareLights[i];
                    entry.screenX = ndc.x * 0.5f + 0.5f;
                    entry.screenY = 0.5f - ndc.y * 0.5f; // matches the uv.y-flip convention used throughout the post pass
                    entry.ndcDepth = ndc.z;
                    entry.active = 1.0f;
                    entry.colorR = lights[i].color.x * lights[i].intensity;
                    entry.colorG = lights[i].color.y * lights[i].intensity;
                    entry.colorB = lights[i].color.z * lights[i].intensity;
                }
            }
            postEncoder->setFragmentBytes(lensFlareLights, sizeof(lensFlareLights), 1);

            postEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, (NS::UInteger)0, (NS::UInteger)3);
            postEncoder->endEncoding();

            // --- Pass C: overlay - gizmo/light markers/rays (depth-tested against the scene
            // geometry Pass A already wrote) plus ImGui, drawn on top of Pass B's tone-mapped
            // result. overlayRPD was already built above (beginUIFrame needed it before any
            // ImGui:: widget calls this frame).
            MTL::RenderCommandEncoder* overlayEncoder = cmdBuffer->renderCommandEncoder(overlayRPD);
            overlayEncoder->setDepthStencilState(depthState);

            if (camera.uiMode) {
                drawAxisGizmo(axisGizmo, overlayEncoder, uniformBuffer, kGizmoUniformOffset);
                for (int i = 0; i < lightCount; i++) {
                    drawLightMarker(lightMarker, overlayEncoder, uniformBuffer, kLightMarkerUniformOffset + i * kUniformStride);
                    if (lightObjects[i] && lights[i].type != LightType::Point) {
                        drawLightDirectionRay(lightMarker, overlayEncoder, uniformBuffer, kLightRayUniformOffset + i * kUniformStride);
                    }
                }
            }

            endUIFrame(cmdBuffer, overlayEncoder);

            // Free this frame's uniform-buffer slot for reuse once the GPU actually finishes
            // reading it, kMaxFramesInFlight frames from now.
            cmdBuffer->addCompletedHandler([frameBoundarySemaphore](MTL::CommandBuffer*) {
                dispatch_semaphore_signal(frameBoundarySemaphore);
            });

            // Present the texture onto the screen and submit to the GPU
            overlayEncoder->endEncoding();
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
    postProcessSamplerState->release();
    postProcessSamplerDesc->release();
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
    hdrColorTexture->release();
    bloomTextureA->release();
    bloomTextureB->release();
    selectionMaskTexture->release();
    depthTexture->release();
    depthState->release();
    depthDesc->release();
    maskDepthState->release();
    maskDepthDesc->release();
    shadowPipelineState->release();
    shadowPipeDesc->release();
    cubeShadowVertFunc->release();
    cubeShadowFragFunc->release();
    postProcessPipelineState->release();
    postProcessPipeDesc->release();
    postProcessVertFunc->release();
    postProcessFragFunc->release();
    bloomExtractPipelineState->release();
    bloomExtractPipeDesc->release();
    bloomExtractFragFunc->release();
    blurPipelineState->release();
    blurPipeDesc->release();
    blurFragFunc->release();
    selectionMaskPipelineState->release();
    selectionMaskPipeDesc->release();
    selectionMaskFragFunc->release();
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
