#pragma once
#include <Metal/Metal.hpp>

// Loads an image from disk (via stb_image) and uploads it into an RGBA8Unorm MTL::Texture.
// isSRGB should be true for color/albedo textures (their bytes are gamma-encoded on disk, so the
// GPU needs to decode them to linear before lighting math touches them) and false for textures
// that store non-color data - e.g. normal maps, which are already linear vectors, not colors.
// Returns nullptr and logs to stderr on failure.
MTL::Texture* loadTexture(MTL::Device* device, const char* path, bool isSRGB);
