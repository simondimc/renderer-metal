#pragma once
#include <Metal/Metal.hpp>
#include <string>

// GPU buffers for one loaded mesh. Vertex layout matches CubeMesh::vertexStrideFloats exactly:
// pos3, uv2, normal3, tangent3 (11 floats/vertex) - so it slots into the existing vertex
// descriptor/pipeline unchanged. indexCount == 0 (buffers left null) signals a failed load.
struct MeshData {
    MTL::Buffer* vertexBuffer = nullptr;
    MTL::Buffer* indexBuffer = nullptr;
    NS::UInteger indexCount = 0;
    MTL::IndexType indexType = MTL::IndexTypeUInt32;
};

// Loads a mesh from disk into GPU buffers, dispatching on file extension:
// .obj -> tinyobjloader, .gltf/.glb -> cgltf. Only the first mesh/primitive is read.
// Returns a MeshData with indexCount == 0 and logs to stderr on failure.
MeshData loadMesh(MTL::Device* device, const std::string& path);
