#include "Texture.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

MTL::Texture* loadTexture(MTL::Device* device, const char* path) {
    stbi_set_flip_vertically_on_load(true);

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
}
