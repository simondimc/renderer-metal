#include "Environment.hpp"

#include "stb_image.h" // implementation lives in Texture.cpp

// tinyexr: single-header OpenEXR reader (implementation compiled here). Use the system zlib for the
// ZIP-compressed EXR variants instead of bundling miniz; PIZ (what Poly Haven ships) is built in.
#include <zlib.h>
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <set>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr NS::UInteger kSkySize = 1024;        // per cube face
constexpr NS::UInteger kIrradianceSize = 32;   // diffuse light is very low-frequency
constexpr NS::UInteger kPrefilterSize = 256;   // mip 0 is the sharpest reflection a mirror can show
constexpr NS::UInteger kBrdfLUTSize = 256;

// fp16 tops out at 65504; a real HDR's sun pixel can be far brighter, and an overflow to infinity
// would poison every blurred mip that averages it in.
constexpr float kMaxRadiance = 60000.0f;

MTL::Texture* makeCube(MTL::Device* device, NS::UInteger size, bool mipmapped) {
    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::textureCubeDescriptor(
        MTL::PixelFormatRGBA16Float, size, mipmapped
    );
    desc->setStorageMode(MTL::StorageModePrivate);
    desc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    return device->newTexture(desc);
}

bool hasExtension(const std::string& path, const char* ext) {
    std::string actual = fs::path(path).extension().string();
    std::transform(actual.begin(), actual.end(), actual.begin(), ::tolower);
    return actual == ext;
}

// Decodes a Radiance .hdr or OpenEXR file into tightly-packed float RGBA (row 0 = top of the
// panorama, what the equirect kernel expects), radiance clamped to kMaxRadiance. False on failure.
bool decodeEquirect(const std::string& path, std::vector<float>& rgba, int& w, int& h) {
    if (hasExtension(path, ".exr")) {
        float* pixels = nullptr;
        const char* error = nullptr;
        if (LoadEXR(&pixels, &w, &h, path.c_str(), &error) != TINYEXR_SUCCESS) {
            fprintf(stderr, "Failed to load environment %s: %s\n", path.c_str(), error ? error : "unknown error");
            if (error) FreeEXRErrorMessage(error);
            return false;
        }
        rgba.assign(pixels, pixels + (size_t)w * h * 4);
        free(pixels);
    } else {
        stbi_set_flip_vertically_on_load(false);
        int channels;
        float* rgb = stbi_loadf(path.c_str(), &w, &h, &channels, 3);
        if (!rgb) {
            fprintf(stderr, "Failed to load environment %s: %s\n", path.c_str(), stbi_failure_reason());
            return false;
        }
        rgba.resize((size_t)w * h * 4);
        for (size_t i = 0; i < (size_t)w * h; i++) {
            for (int c = 0; c < 3; c++) rgba[i * 4 + c] = rgb[i * 3 + c];
            rgba[i * 4 + 3] = 1.0f;
        }
        stbi_image_free(rgb);
    }

    for (size_t i = 0; i < (size_t)w * h; i++) {
        for (int c = 0; c < 3; c++) {
            float& v = rgba[i * 4 + c];
            v = std::isfinite(v) ? std::clamp(v, 0.0f, kMaxRadiance) : 0.0f; // NaN/inf pixels would poison the blurred mips
        }
        rgba[i * 4 + 3] = 1.0f;
    }
    return true;
}

// Loads a panorama as a float RGBA texture for the equirect->cube kernel to read. Null on failure.
MTL::Texture* loadEquirect(MTL::Device* device, const std::string& path) {
    std::vector<float> rgba;
    int w, h;
    if (!decodeEquirect(path, rgba, w, h)) return nullptr;

    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRGBA32Float, (NS::UInteger)w, (NS::UInteger)h, false
    );
    desc->setStorageMode(MTL::StorageModeShared);
    desc->setUsage(MTL::TextureUsageShaderRead);
    MTL::Texture* texture = device->newTexture(desc);
    texture->replaceRegion(MTL::Region(0, 0, (NS::UInteger)w, (NS::UInteger)h), 0, rgba.data(), (NS::UInteger)w * 16);
    return texture;
}

// One thread per texel (x, y) and per cube face (z) - 8x8 threadgroups tile every size used here.
void dispatchCube(MTL::ComputeCommandEncoder* encoder, NS::UInteger size) {
    encoder->dispatchThreads(MTL::Size::Make(size, size, 6), MTL::Size::Make(8, 8, 1));
}

} // namespace

MTL::ComputePipelineState* EnvironmentLibrary::makePipeline(MTL::Library* library, const char* functionName) {
    MTL::Function* function = library->newFunction(NS::String::string(functionName, NS::UTF8StringEncoding));
    NS::Error* error = nullptr;
    MTL::ComputePipelineState* pipeline = function ? device_->newComputePipelineState(function, &error) : nullptr;
    if (!pipeline) fprintf(stderr, "Failed to create compute pipeline %s\n", functionName);
    if (function) function->release();
    return pipeline;
}

bool EnvironmentLibrary::reloadPipelines(MTL::Library* library) {
    MTL::ComputePipelineState** slots[] = {&proceduralSkyPipeline_, &equirectPipeline_, &irradiancePipeline_,
                                           &prefilterPipeline_, &brdfPipeline_};
    const char* names[] = {"proceduralSkyToCubeKernel", "equirectToCubeKernel", "irradianceKernel",
                           "prefilterKernel", "brdfLUTKernel"};
    MTL::ComputePipelineState* fresh[5] = {};
    bool ok = true;
    for (int i = 0; i < 5; i++) {
        fresh[i] = makePipeline(library, names[i]);
        ok = ok && fresh[i];
    }
    for (int i = 0; i < 5; i++) {
        if (ok) {
            (*slots[i])->release();
            *slots[i] = fresh[i];
        } else if (fresh[i]) {
            fresh[i]->release();
        }
    }
    return ok;
}

EnvironmentLibrary::EnvironmentLibrary(MTL::Device* device, MTL::Library* library, std::string rootDir)
    : device_(device), queue_(device->newCommandQueue()), rootDir_(std::move(rootDir)) {
    proceduralSkyPipeline_ = makePipeline(library, "proceduralSkyToCubeKernel");
    equirectPipeline_ = makePipeline(library, "equirectToCubeKernel");
    irradiancePipeline_ = makePipeline(library, "irradianceKernel");
    prefilterPipeline_ = makePipeline(library, "prefilterKernel");
    brdfPipeline_ = makePipeline(library, "brdfLUTKernel");

    MTL::TextureDescriptor* lutDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRG16Float, kBrdfLUTSize, kBrdfLUTSize, false
    );
    lutDesc->setStorageMode(MTL::StorageModePrivate);
    lutDesc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    brdfLUT_ = device_->newTexture(lutDesc);

    MTL::CommandBuffer* cmd = queue_->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = cmd->computeCommandEncoder();
    encoder->setComputePipelineState(brdfPipeline_);
    encoder->setTexture(brdfLUT_, 0);
    encoder->dispatchThreads(MTL::Size::Make(kBrdfLUTSize, kBrdfLUTSize, 1), MTL::Size::Make(8, 8, 1));
    encoder->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    // The fallback every other environment can degrade to, so it's baked up front.
    loaded_[kDefaultEnvironment] = bake(nullptr);
    rescan();
}

EnvironmentLibrary::~EnvironmentLibrary() {
    // A failed load shares the procedural environment's textures, so release each set only once.
    std::set<MTL::Texture*> released;
    for (auto& entry : loaded_) {
        if (!released.insert(entry.second.sky).second) continue;
        entry.second.sky->release();
        entry.second.irradiance->release();
        entry.second.prefiltered->release();
    }
    brdfLUT_->release();
    proceduralSkyPipeline_->release();
    equirectPipeline_->release();
    irradiancePipeline_->release();
    prefilterPipeline_->release();
    brdfPipeline_->release();
    queue_->release();
}

void EnvironmentLibrary::rescan() {
    names_.clear();
    files_.clear();
    names_.push_back(kDefaultEnvironment);

    std::vector<std::string> found;
    std::error_code ec;
    for (const auto& file : fs::directory_iterator(rootDir_, ec)) {
        if (!file.is_regular_file()) continue;
        std::string ext = file.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".hdr" && ext != ".exr") continue;

        std::string name = file.path().stem().string();
        if (files_.emplace(name, file.path().string()).second) found.push_back(name); // same stem twice: first wins
    }
    std::sort(found.begin(), found.end());
    names_.insert(names_.end(), found.begin(), found.end());
}

const Environment& EnvironmentLibrary::get(const std::string& name) {
    auto cached = loaded_.find(name);
    if (cached != loaded_.end()) return cached->second;

    auto file = files_.find(name);
    if (file == files_.end()) return loaded_[kDefaultEnvironment]; // unknown; not cached, rescan() may add it

    MTL::Texture* equirect = loadEquirect(device_, file->second);
    if (!equirect) {
        loaded_[name] = loaded_[kDefaultEnvironment];
        return loaded_[name];
    }
    loaded_[name] = bake(equirect);
    equirect->release();
    return loaded_[name];
}

Environment EnvironmentLibrary::bake(MTL::Texture* equirect) {
    Environment env;
    env.sky = makeCube(device_, kSkySize, true);
    env.irradiance = makeCube(device_, kIrradianceSize, false);
    env.prefiltered = makeCube(device_, kPrefilterSize, true);

    MTL::CommandBuffer* cmd = queue_->commandBuffer();

    // 1. The environment as a cube map, plus its mip chain (the blurred mips are what the two
    // convolutions below sample from to keep a small bright sun from turning into noise).
    {
        MTL::ComputeCommandEncoder* encoder = cmd->computeCommandEncoder();
        if (equirect) {
            encoder->setComputePipelineState(equirectPipeline_);
            encoder->setTexture(equirect, 0);
            encoder->setTexture(env.sky, 1);
        } else {
            encoder->setComputePipelineState(proceduralSkyPipeline_);
            encoder->setTexture(env.sky, 0);
        }
        dispatchCube(encoder, kSkySize);
        encoder->endEncoding();

        MTL::BlitCommandEncoder* blit = cmd->blitCommandEncoder();
        blit->generateMipmaps(env.sky);
        blit->endEncoding();
    }

    // 2. Diffuse irradiance.
    {
        MTL::ComputeCommandEncoder* encoder = cmd->computeCommandEncoder();
        encoder->setComputePipelineState(irradiancePipeline_);
        encoder->setTexture(env.sky, 0);
        encoder->setTexture(env.irradiance, 1);
        dispatchCube(encoder, kIrradianceSize);
        encoder->endEncoding();
    }

    // 3. Specular prefilter: one dispatch per mip, each through a single-mip view of the output, at
    // the roughness that mip stands for (0 at the top mip up to 1 at the smallest).
    std::vector<MTL::Texture*> mipViews;
    NS::UInteger mipCount = env.prefiltered->mipmapLevelCount();
    for (NS::UInteger mip = 0; mip < mipCount; mip++) {
        MTL::Texture* view = env.prefiltered->newTextureView(
            MTL::PixelFormatRGBA16Float, MTL::TextureTypeCube, NS::Range::Make(mip, 1), NS::Range::Make(0, 6)
        );
        mipViews.push_back(view);
        float roughness = (float)mip / (float)(mipCount - 1);

        MTL::ComputeCommandEncoder* encoder = cmd->computeCommandEncoder();
        encoder->setComputePipelineState(prefilterPipeline_);
        encoder->setTexture(env.sky, 0);
        encoder->setTexture(view, 1);
        encoder->setBytes(&roughness, sizeof(float), 0);
        dispatchCube(encoder, std::max<NS::UInteger>(kPrefilterSize >> mip, 1));
        encoder->endEncoding();
    }

    cmd->commit();
    cmd->waitUntilCompleted();
    for (MTL::Texture* view : mipViews) view->release();
    return env;
}
