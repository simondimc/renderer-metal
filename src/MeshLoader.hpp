#pragma once
#include <Metal/Metal.hpp>
#include <simd/simd.h>
#include <string>

// GPU buffers for one loaded mesh. Vertex layout matches CubeMesh::vertexStrideFloats exactly:
// pos3, uv2, normal3, tangent3 (11 floats/vertex) - so it slots into the existing vertex
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
};

// Loads a mesh from disk into GPU buffers, dispatching on file extension:
// .obj -> tinyobjloader, .gltf/.glb -> cgltf. Only the first mesh/primitive is read.
// Returns a MeshData with indexCount == 0 and logs to stderr on failure.
MeshData loadMesh(MTL::Device* device, const std::string& path);
