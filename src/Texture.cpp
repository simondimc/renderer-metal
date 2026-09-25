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

MTL::Texture* loadPackedORMTexture(MTL::Device* device, const char* occlusionPath, const char* roughnessPath,
                                   const char* metallicPath) {
    stbi_set_flip_vertically_on_load(true);

    // R = occlusion, G = roughness, B = metallic; a missing map falls back to fillValue.
    const char* paths[3] = {occlusionPath, roughnessPath, metallicPath};
    const unsigned char fillValues[3] = {255, 255, 0};
    unsigned char* channelData[3] = {nullptr, nullptr, nullptr};
    int w = 0, h = 0;
    bool ok = true;
    for (int c = 0; c < 3 && ok; c++) {
        if (!paths[c]) continue;
        int cw, ch, channels;
        channelData[c] = stbi_load(paths[c], &cw, &ch, &channels, STBI_grey);
        if (!channelData[c]) {
            fprintf(stderr, "Failed to load ORM input %s: %s\n", paths[c], stbi_failure_reason());
            ok = false;
        } else if (w == 0) {
            w = cw;
            h = ch;
        } else if (cw != w || ch != h) {
            fprintf(stderr, "ORM input %s is %dx%d, expected %dx%d\n", paths[c], cw, ch, w, h);
            ok = false;
        }
    }

    MTL::Texture* texture = nullptr;
    if (ok && w > 0) {
        std::vector<unsigned char> packed((size_t)w * h * 4);
        for (size_t i = 0; i < (size_t)w * h; i++) {
            for (int c = 0; c < 3; c++) {
                packed[i * 4 + c] = channelData[c] ? channelData[c][i] : fillValues[c];
            }
            packed[i * 4 + 3] = 255;
        }
        texture = uploadRGBA8(device, packed.data(), w, h, /*isSRGB=*/false, true);
    }
    for (int c = 0; c < 3; c++) {
        if (channelData[c]) stbi_image_free(channelData[c]);
    }
    return texture;
}

MTL::Texture* createSolidTexture(MTL::Device* device, unsigned char r, unsigned char g, unsigned char b,
                                 unsigned char a, bool isSRGB) {
    unsigned char pixel[4] = {r, g, b, a};
    return uploadRGBA8(device, pixel, 1, 1, isSRGB, false);
}
