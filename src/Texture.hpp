#pragma once
#include <Metal/Metal.hpp>

// Loads an image from disk (via stb_image) and uploads it into an RGBA8Unorm MTL::Texture.
// Returns nullptr and logs to stderr on failure.
MTL::Texture* loadTexture(MTL::Device* device, const char* path);
