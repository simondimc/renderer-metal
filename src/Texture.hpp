#pragma once
#include <Metal/Metal.hpp>
#include <cstddef>

// Loads an image from disk (via stb_image) and uploads it into an RGBA8Unorm MTL::Texture with a
// full mip chain. Rows are flipped so the image's bottom-left is UV (0,0) - matching the OpenGL-
// style V-up UVs of CubeMesh.hpp and .obj files (glTF UVs are V-down, see loadTextureFromMemory).
// isSRGB should be true for color/albedo textures (their bytes are gamma-encoded on disk, so the
// GPU needs to decode them to linear before lighting math touches them) and false for textures
// that store non-color data - e.g. normal maps, which are already linear vectors, not colors.
// Returns nullptr and logs to stderr on failure.
MTL::Texture* loadTexture(MTL::Device* device, const char* path, bool isSRGB);

// Same as loadTexture, but decodes an encoded image (PNG/JPEG/...) already in memory - e.g. one
// embedded in a .glb - and never flips it: glTF UVs put (0,0) at the image's top-left.
MTL::Texture* loadTextureFromMemory(MTL::Device* device, const unsigned char* data, size_t size, bool isSRGB);

// Packs up to three single-channel images (same size) into one glTF-style "ORM" texture: R =
// occlusion, G = roughness, B = metallic. Any path may be null when the set lacks that map; it then
// gets a neutral constant (no occlusion, fully rough, non-metal). At least one path is required.
// Non-sRGB. Flipped like loadTexture.
MTL::Texture* loadPackedORMTexture(MTL::Device* device, const char* occlusionPath, const char* roughnessPath,
                                   const char* metallicPath);

// A 1x1 texture of one flat color - stands in for a material's missing albedo/normal/ORM map.
MTL::Texture* createSolidTexture(MTL::Device* device, unsigned char r, unsigned char g, unsigned char b,
                                 unsigned char a, bool isSRGB);
