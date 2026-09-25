#include "TextureLibrary.hpp"
#include "Texture.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

namespace {

bool contains(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

bool isStbImage(const std::string& ext) {
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".tga" || ext == ".bmp";
}

// Files a map under the slot its filename suggests. The first file matching a slot keeps it, so
// when a set ships the same map twice (e.g. png and jpg) the result is at least deterministic.
void classify(const fs::path& file, TextureSetFiles& out) {
    std::string ext = file.extension().string();
    std::string stem = file.stem().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    std::transform(stem.begin(), stem.end(), stem.begin(), ::tolower);
    if (!isStbImage(ext)) return;

    std::string* slot = nullptr;
    if (contains(stem, "_diff") || contains(stem, "_albedo") || contains(stem, "_basecolor")) slot = &out.albedo;
    else if (contains(stem, "_nor_dx") || contains(stem, "_normal_dx")) return; // DirectX +Y-down normals: wrong convention
    else if (contains(stem, "_nor")) slot = &out.normal;                       // _nor, _nor_gl, _normal, _normal_gl
    else if (contains(stem, "_rough")) slot = &out.roughness;
    else if (contains(stem, "_metal")) slot = &out.metallic;
    else if (contains(stem, "_ao")) slot = &out.occlusion;
    if (slot && slot->empty()) *slot = file.string();
}

const char* orNull(const std::string& s) { return s.empty() ? nullptr : s.c_str(); }

} // namespace

TextureLibrary::TextureLibrary(MTL::Device* device, std::string rootDir) : device_(device), rootDir_(std::move(rootDir)) {
    rescan();
}

TextureLibrary::~TextureLibrary() {
    for (auto& entry : loaded_) {
        TextureSet* set = entry.second;
        if (!set) continue;
        if (set->albedo) set->albedo->release();
        if (set->normal) set->normal->release();
        if (set->orm) set->orm->release();
        delete set;
    }
}

void TextureLibrary::rescan() {
    names_.clear();
    files_.clear();

    std::error_code ec;
    for (const auto& dir : fs::directory_iterator(rootDir_, ec)) {
        if (!dir.is_directory()) continue;
        // Poly Haven downloads keep the images in a textures/ subfolder; also accept them directly
        fs::path imageDir = dir.path() / "textures";
        if (!fs::is_directory(imageDir, ec)) imageDir = dir.path();

        TextureSetFiles files;
        for (const auto& file : fs::directory_iterator(imageDir, ec)) {
            if (file.is_regular_file()) classify(file.path(), files);
        }
        if (files.albedo.empty()) continue;

        std::string name = dir.path().filename().string();
        names_.push_back(name);
        files_[name] = files;
    }
    std::sort(names_.begin(), names_.end());
}

const TextureSet* TextureLibrary::get(const std::string& name) {
    auto cached = loaded_.find(name);
    if (cached != loaded_.end()) return cached->second;

    auto found = files_.find(name);
    if (found == files_.end()) return nullptr; // unknown; not cached, since rescan() may add it

    const TextureSetFiles& files = found->second;
    TextureSet* set = new TextureSet();
    set->albedo = loadTexture(device_, files.albedo.c_str(), /*isSRGB=*/true);
    // Normal maps hold linear vectors, not color, so they're never sRGB-decoded.
    if (!files.normal.empty()) set->normal = loadTexture(device_, files.normal.c_str(), /*isSRGB=*/false);
    bool hasOrmInput = !files.occlusion.empty() || !files.roughness.empty() || !files.metallic.empty();
    if (hasOrmInput) {
        set->orm = loadPackedORMTexture(device_, orNull(files.occlusion), orNull(files.roughness), orNull(files.metallic));
    }
    set->params.factors.z = files.occlusion.empty() ? 0.0f : 1.0f;

    // A map that exists but won't decode is a real failure - don't half-render the set.
    bool failed = !set->albedo || (!files.normal.empty() && !set->normal) || (hasOrmInput && !set->orm);
    if (failed) {
        fprintf(stderr, "Texture set '%s' failed to load\n", name.c_str());
        if (set->albedo) set->albedo->release();
        if (set->normal) set->normal->release();
        if (set->orm) set->orm->release();
        delete set;
        set = nullptr;
    }
    loaded_[name] = set;
    return set;
}
