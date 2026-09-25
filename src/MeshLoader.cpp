#include "MeshLoader.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#include <simd/simd.h>
#include <vector>
#include <cstdio>
#include <cstring>

namespace {

// Vertex layout matches CubeMesh::vertexStrideFloats: pos3, uv2, normal3, tangent3.
constexpr size_t kVertexStrideFloats = 11;

// Same edge/UV formula CubeMesh.hpp's hand-authored tangents follow (see LearnOpenGL's normal
// mapping tutorial for the derivation) - degenerate UVs (denom ~ 0) fall back to the first edge
// direction so a bad UV never produces a NaN/Inf tangent that would corrupt the whole buffer.
simd::float3 computeFaceTangent(simd::float3 p0, simd::float3 p1, simd::float3 p2,
                                 simd::float2 uv0, simd::float2 uv1, simd::float2 uv2) {
    simd::float3 edge1 = p1 - p0;
    simd::float3 edge2 = p2 - p0;
    simd::float2 deltaUV1 = uv1 - uv0;
    simd::float2 deltaUV2 = uv2 - uv0;

    float denom = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
    if (fabsf(denom) < 1e-10f) {
        return simd_normalize(edge1);
    }
    float f = 1.0f / denom;
    simd::float3 tangent = f * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
    float len = simd_length(tangent);
    return (len > 1e-10f) ? (tangent / len) : simd_normalize(edge1);
}

void writeVertex(std::vector<float>& out, simd::float3 pos, simd::float2 uv, simd::float3 normal, simd::float3 tangent) {
    out.push_back(pos.x); out.push_back(pos.y); out.push_back(pos.z);
    out.push_back(uv.x); out.push_back(uv.y);
    out.push_back(normal.x); out.push_back(normal.y); out.push_back(normal.z);
    out.push_back(tangent.x); out.push_back(tangent.y); out.push_back(tangent.z);
}

MeshData uploadMesh(MTL::Device* device, const std::vector<float>& vertexData,
                     const std::vector<uint32_t>& indexData) {
    MeshData mesh;
    mesh.vertexBuffer = device->newBuffer(vertexData.data(), vertexData.size() * sizeof(float),
                                           MTL::ResourceStorageModeShared);
    mesh.indexBuffer = device->newBuffer(indexData.data(), indexData.size() * sizeof(uint32_t),
                                          MTL::ResourceStorageModeShared);
    mesh.indexCount = (NS::UInteger)indexData.size();
    mesh.indexType = MTL::IndexTypeUInt32;

    if (!vertexData.empty()) {
        simd::float3 pos0 = simd_make_float3(vertexData[0], vertexData[1], vertexData[2]);
        mesh.localMin = mesh.localMax = pos0;
        for (size_t i = kVertexStrideFloats; i + 2 < vertexData.size(); i += kVertexStrideFloats) {
            simd::float3 p = simd_make_float3(vertexData[i], vertexData[i + 1], vertexData[i + 2]);
            mesh.localMin = simd_min(mesh.localMin, p);
            mesh.localMax = simd_max(mesh.localMax, p);
        }
    }
    return mesh;
}

// Expands to flat, non-shared per-triangle vertices (no vertex welding) since tinyobjloader's
// triangulated face data isn't indexed to begin with. Normals are copied from the OBJ where
// present (falling back to the flat face normal); tangents are always a single flat per-triangle
// value, mirroring how CubeMesh.hpp hand-authors its own per-face tangents.
MeshData loadMeshObj(MTL::Device* device, const std::string& path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    // LoadObj otherwise only searches the process's cwd for the referenced .mtl file, not the
    // .obj's own directory - pass that explicitly so material warnings don't fire spuriously.
    size_t lastSlash = path.find_last_of("/\\");
    std::string baseDir = (lastSlash == std::string::npos) ? "" : path.substr(0, lastSlash + 1);
    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str(), baseDir.c_str());
    if (!warn.empty()) fprintf(stderr, "loadMesh warning (%s): %s\n", path.c_str(), warn.c_str());
    if (!ok) {
        fprintf(stderr, "Failed to load mesh %s: %s\n", path.c_str(), err.c_str());
        return MeshData{};
    }

    std::vector<float> vertexData;
    std::vector<uint32_t> indexData;

    for (const auto& shape : shapes) {
        size_t indexOffset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int faceVertexCount = shape.mesh.num_face_vertices[f];
            if (faceVertexCount != 3) {
                // LoadObj triangulates by default, so this shouldn't happen; skip defensively.
                indexOffset += faceVertexCount;
                continue;
            }

            simd::float3 pos[3];
            simd::float2 uv[3];
            simd::float3 normal[3];
            bool haveNormal = true;
            for (int v = 0; v < 3; v++) {
                tinyobj::index_t idx = shape.mesh.indices[indexOffset + v];
                pos[v] = simd_make_float3(attrib.vertices[3 * idx.vertex_index + 0],
                                           attrib.vertices[3 * idx.vertex_index + 1],
                                           attrib.vertices[3 * idx.vertex_index + 2]);
                uv[v] = (idx.texcoord_index >= 0)
                    ? simd_make_float2(attrib.texcoords[2 * idx.texcoord_index + 0],
                                        attrib.texcoords[2 * idx.texcoord_index + 1])
                    : simd_make_float2(0.0f, 0.0f);
                if (idx.normal_index >= 0) {
                    normal[v] = simd_make_float3(attrib.normals[3 * idx.normal_index + 0],
                                                  attrib.normals[3 * idx.normal_index + 1],
                                                  attrib.normals[3 * idx.normal_index + 2]);
                } else {
                    haveNormal = false;
                }
            }
            if (!haveNormal) {
                simd::float3 flat = simd_normalize(simd_cross(pos[1] - pos[0], pos[2] - pos[0]));
                normal[0] = normal[1] = normal[2] = flat;
            }

            simd::float3 tangent = computeFaceTangent(pos[0], pos[1], pos[2], uv[0], uv[1], uv[2]);

            uint32_t base = (uint32_t)(vertexData.size() / kVertexStrideFloats);
            for (int v = 0; v < 3; v++) {
                writeVertex(vertexData, pos[v], uv[v], normal[v], tangent);
                indexData.push_back(base + v);
            }
            indexOffset += 3;
        }
    }

    if (indexData.empty()) {
        fprintf(stderr, "Mesh %s has no triangles\n", path.c_str());
        return MeshData{};
    }
    return uploadMesh(device, vertexData, indexData);
}

// Transforms a point (w=1) by a column-major 4x4 matrix, as cgltf_node_transform_world returns.
simd::float3 transformPoint(const float* m, simd::float3 p) {
    return simd_make_float3(
        m[0] * p.x + m[4] * p.y + m[8]  * p.z + m[12],
        m[1] * p.x + m[5] * p.y + m[9]  * p.z + m[13],
        m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]);
}

// Transforms a direction (w=0, no translation) by the matrix's linear part. Not the mathematically
// correct inverse-transpose for non-uniform scale, but glTF scene-graph nodes are near-uniform in
// practice and this keeps the loader simple - matches the same pragmatic spirit as the OBJ path's
// flat per-triangle tangents.
simd::float3 transformDirection(const float* m, simd::float3 d) {
    simd::float3 r = simd_make_float3(
        m[0] * d.x + m[4] * d.y + m[8]  * d.z,
        m[1] * d.x + m[5] * d.y + m[9]  * d.z,
        m[2] * d.x + m[6] * d.y + m[10] * d.z);
    float len = simd_length(r);
    return (len > 1e-10f) ? (r / len) : d;
}

// Decodes one primitive's vertices/indices (world-transformed by the owning node) and appends them
// to the shared merged buffers - this is what lets a multi-part glTF file (e.g. separate meshes per
// chess piece/board section) come in as one combined static mesh, matching the renderer's
// one-buffer-per-SceneObject model.
void appendGltfPrimitive(const cgltf_primitive& prim, const float* worldMatrix,
                          std::vector<float>& vertexData, std::vector<uint32_t>& indexData) {
    if (prim.type != cgltf_primitive_type_triangles) return;
    const cgltf_accessor* posAcc = cgltf_find_accessor(&prim, cgltf_attribute_type_position, 0);
    if (!posAcc) return;
    const cgltf_accessor* normalAcc = cgltf_find_accessor(&prim, cgltf_attribute_type_normal, 0);
    const cgltf_accessor* uvAcc = cgltf_find_accessor(&prim, cgltf_attribute_type_texcoord, 0);
    const cgltf_accessor* tangentAcc = cgltf_find_accessor(&prim, cgltf_attribute_type_tangent, 0);

    size_t vertexCount = posAcc->count;
    uint32_t vertexBase = (uint32_t)(vertexData.size() / kVertexStrideFloats);
    vertexData.resize(vertexData.size() + vertexCount * kVertexStrideFloats);

    for (size_t i = 0; i < vertexCount; i++) {
        float pos[3] = {0, 0, 0};
        cgltf_accessor_read_float(posAcc, i, pos, 3);
        float uv[2] = {0, 0};
        if (uvAcc) cgltf_accessor_read_float(uvAcc, i, uv, 2);
        float normal[3] = {0, 1, 0};
        if (normalAcc) cgltf_accessor_read_float(normalAcc, i, normal, 3);
        float tangent[4] = {1, 0, 0, 1};
        if (tangentAcc) cgltf_accessor_read_float(tangentAcc, i, tangent, 4);

        simd::float3 worldPos = transformPoint(worldMatrix, simd_make_float3(pos[0], pos[1], pos[2]));
        simd::float3 worldNormal = transformDirection(worldMatrix, simd_make_float3(normal[0], normal[1], normal[2]));
        simd::float3 worldTangent = transformDirection(worldMatrix, simd_make_float3(tangent[0], tangent[1], tangent[2]));

        float* v = &vertexData[(vertexBase + i) * kVertexStrideFloats];
        v[0] = worldPos.x; v[1] = worldPos.y; v[2] = worldPos.z;
        v[3] = uv[0]; v[4] = uv[1];
        v[5] = worldNormal.x; v[6] = worldNormal.y; v[7] = worldNormal.z;
        v[8] = worldTangent.x; v[9] = worldTangent.y; v[10] = worldTangent.z;
    }

    size_t indexBase = indexData.size();
    if (prim.indices) {
        indexData.resize(indexBase + prim.indices->count);
        for (size_t i = 0; i < prim.indices->count; i++) {
            indexData[indexBase + i] = vertexBase + (uint32_t)cgltf_accessor_read_index(prim.indices, i);
        }
    } else {
        indexData.resize(indexBase + vertexCount);
        for (size_t i = 0; i < vertexCount; i++) indexData[indexBase + i] = vertexBase + (uint32_t)i;
    }

    if (!tangentAcc) {
        std::vector<simd::float3> accum(vertexCount, simd_make_float3(0, 0, 0));
        for (size_t t = indexBase; t + 2 < indexData.size(); t += 3) {
            uint32_t i0 = indexData[t] - vertexBase, i1 = indexData[t + 1] - vertexBase, i2 = indexData[t + 2] - vertexBase;
            float* v0 = &vertexData[(vertexBase + i0) * kVertexStrideFloats];
            float* v1 = &vertexData[(vertexBase + i1) * kVertexStrideFloats];
            float* v2 = &vertexData[(vertexBase + i2) * kVertexStrideFloats];
            simd::float3 face = computeFaceTangent(
                simd_make_float3(v0[0], v0[1], v0[2]), simd_make_float3(v1[0], v1[1], v1[2]), simd_make_float3(v2[0], v2[1], v2[2]),
                simd_make_float2(v0[3], v0[4]), simd_make_float2(v1[3], v1[4]), simd_make_float2(v2[3], v2[4]));
            accum[i0] += face; accum[i1] += face; accum[i2] += face;
        }
        for (size_t i = 0; i < vertexCount; i++) {
            float len = simd_length(accum[i]);
            simd::float3 t = (len > 1e-10f) ? (accum[i] / len) : simd_make_float3(1, 0, 0);
            float* v = &vertexData[(vertexBase + i) * kVertexStrideFloats];
            v[8] = t.x; v[9] = t.y; v[10] = t.z;
        }
    }
}

// Merges every node's mesh (world-transformed) into one combined buffer - a real-world glTF file
// commonly splits one visual model into many small meshes (e.g. one per chess piece/board section),
// but the renderer only has one draw call/model matrix per SceneObject. Tangents (when a primitive
// doesn't provide its own TANGENT attribute) are computed by accumulating each triangle's face
// tangent into its vertices and normalizing - smoother than the OBJ path's flat per-triangle
// tangent since glTF vertices are already shared/indexed within a primitive.
MeshData loadMeshGltf(MTL::Device* device, const std::string& path) {
    cgltf_options options = {};
    cgltf_data* data = nullptr;
    cgltf_result result = cgltf_parse_file(&options, path.c_str(), &data);
    if (result != cgltf_result_success) {
        fprintf(stderr, "Failed to parse glTF %s (error %d)\n", path.c_str(), (int)result);
        return MeshData{};
    }
    result = cgltf_load_buffers(&options, data, path.c_str());
    if (result != cgltf_result_success) {
        fprintf(stderr, "Failed to load glTF buffers %s (error %d)\n", path.c_str(), (int)result);
        cgltf_free(data);
        return MeshData{};
    }
    if (data->nodes_count == 0) {
        fprintf(stderr, "glTF %s has no nodes\n", path.c_str());
        cgltf_free(data);
        return MeshData{};
    }

    // Merge every node's mesh (world-transformed) into one combined buffer - a real-world glTF
    // file commonly splits one visual model into many small meshes (e.g. one per chess piece/
    // board section here), but the renderer only has one draw call/model matrix per SceneObject.
    std::vector<float> vertexData;
    std::vector<uint32_t> indexData;
    for (size_t n = 0; n < data->nodes_count; n++) {
        const cgltf_node* node = &data->nodes[n];
        if (!node->mesh) continue;
        float worldMatrix[16];
        cgltf_node_transform_world(node, worldMatrix);
        for (size_t p = 0; p < node->mesh->primitives_count; p++) {
            appendGltfPrimitive(node->mesh->primitives[p], worldMatrix, vertexData, indexData);
        }
    }

    cgltf_free(data);

    if (indexData.empty()) {
        fprintf(stderr, "glTF %s has no renderable triangles\n", path.c_str());
        return MeshData{};
    }
    return uploadMesh(device, vertexData, indexData);
}

bool hasSuffix(const std::string& s, const char* suffix) {
    size_t n = strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

} // namespace

MeshData loadMesh(MTL::Device* device, const std::string& path) {
    if (hasSuffix(path, ".gltf") || hasSuffix(path, ".glb")) {
        return loadMeshGltf(device, path);
    }
    if (hasSuffix(path, ".obj")) {
        return loadMeshObj(device, path);
    }
    fprintf(stderr, "loadMesh: unrecognized extension for %s (expected .obj/.gltf/.glb)\n", path.c_str());
    return MeshData{};
}
