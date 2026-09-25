#pragma once
#include <Metal/Metal.hpp>
#include <simd/simd.h>
#include <string>
#include <vector>
#include "Uniforms.hpp" // MaterialParams

// One glTF material as the renderer understands it: metallic-roughness, emissive (incl.
// KHR_materials_emissive_strength) and alphaMode (see AlphaMode). Other KHR_materials_* extensions
// such as transmission are ignored, so e.g. glass without alphaMode BLEND renders opaque.
// A null texture means "the material has none": Main.cpp substitutes a neutral 1x1 stand-in.
struct MeshMaterial {
    MaterialParams params;
    MTL::Texture* albedo = nullptr;    // sRGB; alpha channel drives Mask/Blend
    MTL::Texture* normal = nullptr;    // tangent-space, OpenGL (+Y up) convention as in glTF
    MTL::Texture* orm = nullptr;       // glTF packing: G = roughness, B = metallic (R unused, see occlusion)
    MTL::Texture* occlusion = nullptr; // R = occlusion. Often the same image as orm, but glTF allows a separate one
    MTL::Texture* emissive = nullptr;  // sRGB
    AlphaMode alphaMode() const { return (AlphaMode)(int)params.alphaParams.x; }
};

// A contiguous run of the mesh's index buffer drawn with one material.
struct MeshSubmesh {
    NS::UInteger indexOffset = 0; // in indices, not bytes
    NS::UInteger indexCount = 0;
    size_t materialIndex = 0;     // into MeshData::materials
    // Local-space center of the submesh's bounding box - what Blend submeshes are sorted by, so the
    // sort distance follows the object's model matrix. Only computed for glTF.
    simd::float3 center = {0.0f, 0.0f, 0.0f};
};

// GPU buffers for one loaded mesh. Vertex layout matches CubeMesh::vertexStrideFloats exactly:
// pos3, uv2, normal3, tangent4 = xyz + handedness (12 floats/vertex) - so it slots into the existing vertex
// descriptor/pipeline unchanged. indexCount == 0 (buffers left null) signals a failed load.
struct MeshData {
    MTL::Buffer* vertexBuffer = nullptr;
    MTL::Buffer* indexBuffer = nullptr;
    NS::UInteger indexCount = 0;
    MTL::IndexType indexType = MTL::IndexTypeUInt32;
    // Local-space (pre-model-matrix) axis-aligned bounding box, computed once at load time from
    // the raw vertex positions - Main.cpp's click-to-select ray casts against this rather than
    // per-triangle, since the CPU-side vertex data itself isn't kept around after upload.
    simd::float3 localMin = {0.0f, 0.0f, 0.0f};
    simd::float3 localMax = {0.0f, 0.0f, 0.0f};
    // glTF only: the file's own materials and how the index buffer splits across them. Empty for
    // .obj, which is drawn in one call with the renderer's default texture set.
    std::vector<MeshMaterial> materials;
    std::vector<MeshSubmesh> submeshes;
    // True if any submesh uses Mask or Blend - the shadow pass then has to go submesh by submesh
    // (cutting out Mask, skipping Blend) instead of drawing the whole index buffer at once.
    bool hasNonOpaqueSubmeshes = false;
};

// Loads a mesh from disk into GPU buffers, dispatching on file extension:
// .obj -> tinyobjloader, .gltf/.glb -> cgltf. glTF loads every node's mesh, merged (world-
// transformed) into one buffer split into per-material submeshes, plus its textures.
// Returns a MeshData with indexCount == 0 and logs to stderr on failure.
MeshData loadMesh(MTL::Device* device, const std::string& path);
