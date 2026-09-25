#pragma once
#include <Metal/Metal.hpp>
#include <map>
#include <string>
#include <vector>
#include "Uniforms.hpp" // MaterialParams

// One PBR texture set from the texture/ folder, in the same three-texture form the shader takes
// glTF materials in (see MeshMaterial in MeshLoader.hpp). A null texture means the set has no such
// map; Main.cpp substitutes the neutral 1x1 stand-in, same as for glTF.
struct TextureSet {
    MTL::Texture* albedo = nullptr;  // sRGB
    MTL::Texture* normal = nullptr;  // tangent-space, OpenGL (+Y up)
    MTL::Texture* orm = nullptr;     // R = occlusion, G = roughness, B = metallic
    MaterialParams params;           // occlusion strength is 1 only when the set has an AO map
};

// Image paths (relative to build/) of one set's maps; empty = the set has no such map.
struct TextureSetFiles {
    std::string albedo, normal, occlusion, roughness, metallic;
};

// Discovers and lazily loads texture sets: every subfolder of `rootDir` (relative to build/, where
// run.sh launches the binary) holding a "textures/" folder - or the images directly - is one set,
// named after the folder. Maps are recognized by filename suffix (Poly Haven naming): _diff/_albedo,
// _nor_gl/_normal, _rough, _metal, _ao. Only formats stb_image reads work (png/jpg/tga/bmp);
// .exr files are ignored, so convert those to png first. A set needs at least an albedo map.
// Sets are decoded and uploaded the first time get() asks for them, not at scan time.
class TextureLibrary {
public:
    TextureLibrary(MTL::Device* device, std::string rootDir);
    ~TextureLibrary();
    TextureLibrary(const TextureLibrary&) = delete;
    TextureLibrary& operator=(const TextureLibrary&) = delete;

    // Re-reads the folder so sets added while running show up. Already loaded sets stay loaded.
    void rescan();
    const std::vector<std::string>& setNames() const { return names_; }

    // Loads the set on first use. Returns null if the name is unknown or its files fail to load
    // (a failed load is remembered, so it isn't retried - and re-logged - every frame).
    const TextureSet* get(const std::string& name);

private:
    MTL::Device* device_;
    std::string rootDir_;
    std::vector<std::string> names_;
    std::map<std::string, TextureSetFiles> files_;
    std::map<std::string, TextureSet*> loaded_; // null value = load failed
};
