#pragma once
#include <Metal/Metal.hpp>
#include <map>
#include <string>
#include <vector>
#include "Scene.hpp" // kDefaultEnvironment: the built-in procedural sky, always present

// One image-based-lighting environment, precomputed from a single sky image (see fragmentMain in
// Shader.metal for how each part is used). All three are RGBA16Float cube maps.
struct Environment {
    MTL::Texture* sky = nullptr;         // full-res environment with a mip chain - the visible background
    MTL::Texture* irradiance = nullptr;  // small, heavily blurred: diffuse ambient light by surface normal
    MTL::Texture* prefiltered = nullptr; // one mip per roughness step (mip 0 = mirror, last = fully rough)
};

// Discovers and lazily bakes environments: the built-in procedural sky plus every equirectangular
// .hdr (Radiance) or .exr (OpenEXR) file directly inside `rootDir` (relative to build/, where run.sh launches the
// binary), named after the file. Baking runs compute kernels on the GPU and blocks until they finish,
// so it happens the first time get() asks for an environment, not at scan time.
class EnvironmentLibrary {
public:
    EnvironmentLibrary(MTL::Device* device, MTL::Library* library, std::string rootDir);
    ~EnvironmentLibrary();
    EnvironmentLibrary(const EnvironmentLibrary&) = delete;
    EnvironmentLibrary& operator=(const EnvironmentLibrary&) = delete;

    // Re-reads the folder so files added while running show up. Already baked ones stay loaded.
    void rescan();
    // kDefaultEnvironment first, then the .hdr files sorted by name.
    const std::vector<std::string>& names() const { return names_; }

    // Never null: an unknown name or a file that fails to load falls back to the procedural sky
    // (a failed load is remembered, so it isn't retried - and re-logged - every frame).
    const Environment& get(const std::string& name);

    // Split-sum BRDF lookup table (RG16Float): shared by every environment, baked once.
    MTL::Texture* brdfLUT() const { return brdfLUT_; }

private:
    Environment bake(MTL::Texture* equirect); // null equirect = procedural sky
    MTL::ComputePipelineState* makePipeline(MTL::Library* library, const char* functionName);

    MTL::Device* device_;
    MTL::CommandQueue* queue_;
    std::string rootDir_;
    std::vector<std::string> names_;
    std::map<std::string, std::string> files_;   // name -> path
    std::map<std::string, Environment> loaded_;  // failed loads map to the procedural environment

    MTL::ComputePipelineState* proceduralSkyPipeline_ = nullptr;
    MTL::ComputePipelineState* equirectPipeline_ = nullptr;
    MTL::ComputePipelineState* irradiancePipeline_ = nullptr;
    MTL::ComputePipelineState* prefilterPipeline_ = nullptr;
    MTL::ComputePipelineState* brdfPipeline_ = nullptr;
    MTL::Texture* brdfLUT_ = nullptr;
};
