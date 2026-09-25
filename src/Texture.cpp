#include "Texture.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <vector>

namespace {

// Uploads tightly-packed RGBA8 pixels, optionally followed by a full mip chain. Without mips a 4K
// texture minified onto a small object shimmers badly, and the sampler already asks for linear mip
// filtering.
MTL::Texture* uploadRGBA8(MTL::Device* device, const unsigned char* pixels, int w, int h, bool isSRGB, bool mips) {
    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::texture2DDescriptor(
        isSRGB ? MTL::PixelFormatRGBA8Unorm_sRGB : MTL::PixelFormatRGBA8Unorm, (NS::UInteger)w, (NS::UInteger)h, mips
    );
    desc->setStorageMode(MTL::StorageModeShared);
    desc->setUsage(MTL::TextureUsageShaderRead);
    MTL::Texture* texture = device->newTexture(desc);
    texture->replaceRegion(MTL::Region(0, 0, (NS::UInteger)w, (NS::UInteger)h), 0, pixels, (NS::UInteger)w * 4);

    if (mips && texture->mipmapLevelCount() > 1) {
        MTL::CommandQueue* queue = device->newCommandQueue();
        MTL::CommandBuffer* cmd = queue->commandBuffer();
        MTL::BlitCommandEncoder* blit = cmd->blitCommandEncoder();
        blit->generateMipmaps(texture);
        blit->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        queue->release();
    }
    return texture;
}

} // namespace

MTL::Texture* loadTexture(MTL::Device* device, const char* path, bool isSRGB) {
    stbi_set_flip_vertically_on_load(true);

    int w, h, channels;
    unsigned char* pixels = stbi_load(path, &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels) {
        fprintf(stderr, "Failed to load texture %s: %s\n", path, stbi_failure_reason());
        return nullptr;
    }
    MTL::Texture* texture = uploadRGBA8(device, pixels, w, h, isSRGB, true);
    stbi_image_free(pixels);
    return texture;
}

MTL::Texture* loadTextureFromMemory(MTL::Device* device, const unsigned char* data, size_t size, bool isSRGB) {
    stbi_set_flip_vertically_on_load(false);

    int w, h, channels;
    unsigned char* pixels = stbi_load_from_memory(data, (int)size, &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels) {
        fprintf(stderr, "Failed to decode embedded texture: %s\n", stbi_failure_reason());
        return nullptr;
    }
    MTL::Texture* texture = uploadRGBA8(device, pixels, w, h, isSRGB, true);
    stbi_image_free(pixels);
    return texture;
}

MTL::Texture* loadPackedORMTexture(MTL::Device* device, const char* roughnessPath, const char* metallicPath) {
    stbi_set_flip_vertically_on_load(true);

    int rw, rh, mw, mh, channels;
    unsigned char* rough = stbi_load(roughnessPath, &rw, &rh, &channels, STBI_grey);
    unsigned char* metal = stbi_load(metallicPath, &mw, &mh, &channels, STBI_grey);
    if (!rough || !metal || rw != mw || rh != mh) {
        fprintf(stderr, "Failed to load/match ORM inputs %s + %s\n", roughnessPath, metallicPath);
        if (rough) stbi_image_free(rough);
        if (metal) stbi_image_free(metal);
        return nullptr;
    }

    std::vector<unsigned char> packed((size_t)rw * rh * 4);
    for (size_t i = 0; i < (size_t)rw * rh; i++) {
        packed[i * 4 + 0] = 255;      // occlusion: none baked
        packed[i * 4 + 1] = rough[i]; // roughness
        packed[i * 4 + 2] = metal[i]; // metallic
        packed[i * 4 + 3] = 255;
    }
    stbi_image_free(rough);
    stbi_image_free(metal);
    return uploadRGBA8(device, packed.data(), rw, rh, /*isSRGB=*/false, true);
}

MTL::Texture* createSolidTexture(MTL::Device* device, unsigned char r, unsigned char g, unsigned char b,
                                 unsigned char a, bool isSRGB) {
    unsigned char pixel[4] = {r, g, b, a};
    return uploadRGBA8(device, pixel, 1, 1, isSRGB, false);
}
