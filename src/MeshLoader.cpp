#include "MeshLoader.hpp"
#include "Texture.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#include <simd/simd.h>
#include <vector>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <utility>

namespace {

// Vertex layout matches CubeMesh::vertexStrideFloats: pos3, uv2, normal3, tangent4 (xyz + handedness w).
constexpr size_t kVertexStrideFloats = 12;

// The tangent frame of one triangle, from the same edge/UV formula CubeMesh.hpp's hand-authored
// tangents follow (see LearnOpenGL's normal mapping tutorial for the derivation). `tangent` is the
// (normalized) direction of increasing U; `dPdv` is the (unnormalized) direction of increasing V,
// which is what the bitangent handedness is judged against - see tangentHandedness. Degenerate UVs
// (denom ~ 0) fall back to the first edge direction with a zero dPdv (handedness then defaults to
// +1), so a bad UV never produces a NaN/Inf tangent that would corrupt the whole buffer.
struct TriangleFrame {
    simd::float3 tangent;
    simd::float3 dPdv;
};

TriangleFrame computeTriangleFrame(simd::float3 p0, simd::float3 p1, simd::float3 p2,
                                   simd::float2 uv0, simd::float2 uv1, simd::float2 uv2) {
    simd::float3 edge1 = p1 - p0;
    simd::float3 edge2 = p2 - p0;
    simd::float2 deltaUV1 = uv1 - uv0;
    simd::float2 deltaUV2 = uv2 - uv0;

    float denom = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
    if (fabsf(denom) < 1e-10f) {
        return {simd_normalize(edge1), simd_make_float3(0, 0, 0)};
    }
    float f = 1.0f / denom;
    simd::float3 tangent = f * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
    simd::float3 dPdv = f * (deltaUV1.x * edge2 - deltaUV2.x * edge1);
    float len = simd_length(tangent);
    return {(len > 1e-10f) ? (tangent / len) : simd_normalize(edge1), dPdv};
}

// The shader builds the bitangent as cross(normal, tangent) * w (glTF's convention). `up` is the
// direction the normal map's +Y (green) channel points along in world space: dPdv for .obj
// (OpenGL-style, V up) and -dPdv for glTF (V down - image row 0 is the top). w = -1 when that
// differs from cross(normal, tangent), i.e. wherever the texture is mirrored.
float tangentHandedness(simd::float3 normal, simd::float3 tangent, simd::float3 up) {
    return simd_dot(simd_cross(normal, tangent), up) < 0.0f ? -1.0f : 1.0f;
}

void writeVertex(std::vector<float>& out, simd::float3 pos, simd::float2 uv, simd::float3 normal,
                 simd::float3 tangent, float handedness) {
    out.push_back(pos.x); out.push_back(pos.y); out.push_back(pos.z);
    out.push_back(uv.x); out.push_back(uv.y);
    out.push_back(normal.x); out.push_back(normal.y); out.push_back(normal.z);
    out.push_back(tangent.x); out.push_back(tangent.y); out.push_back(tangent.z);
    out.push_back(handedness);
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

            TriangleFrame frame = computeTriangleFrame(pos[0], pos[1], pos[2], uv[0], uv[1], uv[2]);

            uint32_t base = (uint32_t)(vertexData.size() / kVertexStrideFloats);
            for (int v = 0; v < 3; v++) {
                writeVertex(vertexData, pos[v], uv[v], normal[v], frame.tangent,
                            tangentHandedness(normal[v], frame.tangent, frame.dPdv));
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

    // A mirrored node transform (negative determinant) reverses the tangent frame's handedness.
    float detSign = simd_dot(simd_cross(simd_make_float3(worldMatrix[0], worldMatrix[1], worldMatrix[2]),
                                        simd_make_float3(worldMatrix[4], worldMatrix[5], worldMatrix[6])),
                             simd_make_float3(worldMatrix[8], worldMatrix[9], worldMatrix[10])) < 0.0f ? -1.0f : 1.0f;

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
        v[11] = (tangent[3] < 0.0f ? -1.0f : 1.0f) * detSign; // only meaningful with a TANGENT accessor
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
        // No TANGENT attribute: derive per-vertex tangents by accumulating each triangle's frame
        // into its vertices (smoother than one flat tangent per face), then normalizing.
        std::vector<simd::float3> accumT(vertexCount, simd_make_float3(0, 0, 0));
        std::vector<simd::float3> accumV(vertexCount, simd_make_float3(0, 0, 0));
        for (size_t t = indexBase; t + 2 < indexData.size(); t += 3) {
            uint32_t i0 = indexData[t] - vertexBase, i1 = indexData[t + 1] - vertexBase, i2 = indexData[t + 2] - vertexBase;
            float* v0 = &vertexData[(vertexBase + i0) * kVertexStrideFloats];
            float* v1 = &vertexData[(vertexBase + i1) * kVertexStrideFloats];
            float* v2 = &vertexData[(vertexBase + i2) * kVertexStrideFloats];
            TriangleFrame frame = computeTriangleFrame(
                simd_make_float3(v0[0], v0[1], v0[2]), simd_make_float3(v1[0], v1[1], v1[2]), simd_make_float3(v2[0], v2[1], v2[2]),
                simd_make_float2(v0[3], v0[4]), simd_make_float2(v1[3], v1[4]), simd_make_float2(v2[3], v2[4]));
            accumT[i0] += frame.tangent; accumT[i1] += frame.tangent; accumT[i2] += frame.tangent;
            accumV[i0] += frame.dPdv;    accumV[i1] += frame.dPdv;    accumV[i2] += frame.dPdv;
        }
        for (size_t i = 0; i < vertexCount; i++) {
            float len = simd_length(accumT[i]);
            simd::float3 t = (len > 1e-10f) ? (accumT[i] / len) : simd_make_float3(1, 0, 0);
            float* v = &vertexData[(vertexBase + i) * kVertexStrideFloats];
            v[8] = t.x; v[9] = t.y; v[10] = t.z;
            // glTF is V-down, so the normal map's +Y is -dPdv (see tangentHandedness).
            v[11] = tangentHandedness(simd_make_float3(v[5], v[6], v[7]), t, -accumV[i]);
        }
    }
}

// Decodes one glTF image into a GPU texture, memoized per (image, sRGB) so a texture shared by
// several materials is only decoded/uploaded once. Handles images embedded in a .glb/buffer view
// and external files next to the .gltf; base64 "data:" URIs are not supported (logged, skipped).
using TextureCache = std::map<std::pair<const cgltf_image*, bool>, MTL::Texture*>;

MTL::Texture* loadGltfImage(MTL::Device* device, const cgltf_data* data, const cgltf_image* image,
                             bool isSRGB, const std::string& gltfPath, TextureCache& cache) {
    if (!image) return nullptr;
    auto key = std::make_pair(image, isSRGB);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    MTL::Texture* texture = nullptr;
    if (image->buffer_view) {
        const cgltf_buffer_view* view = image->buffer_view;
        const unsigned char* bytes = (const unsigned char*)view->buffer->data + view->offset;
        texture = loadTextureFromMemory(device, bytes, view->size, isSRGB);
    } else if (image->uri && strncmp(image->uri, "data:", 5) != 0) {
        std::string uri = image->uri;
        uri.resize(cgltf_decode_uri(&uri[0])); // undo %20 etc. in place
        size_t slash = gltfPath.find_last_of('/');
        std::string dir = (slash == std::string::npos) ? "" : gltfPath.substr(0, slash + 1);
        std::ifstream in(dir + uri, std::ios::binary);
        std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (bytes.empty()) {
            fprintf(stderr, "glTF image %s%s not found\n", dir.c_str(), uri.c_str());
        } else {
            texture = loadTextureFromMemory(device, bytes.data(), bytes.size(), isSRGB);
        }
    } else {
        fprintf(stderr, "glTF image with an embedded data: URI is not supported\n");
    }
    cache[key] = texture;
    return texture;
}

// Slot 0 of MeshData::materials is the spec's default material (used by primitives that name none);
// glTF material i lives at slot i + 1.
std::vector<MeshMaterial> loadGltfMaterials(MTL::Device* device, const cgltf_data* data,
                                             const std::string& gltfPath) {
    std::vector<MeshMaterial> materials(data->materials_count + 1);
    TextureCache cache;
    for (size_t i = 0; i < data->materials_count; i++) {
        const cgltf_material& src = data->materials[i];
        MeshMaterial& dst = materials[i + 1];
        if (!src.has_pbr_metallic_roughness) continue; // spec defaults, already in dst

        const cgltf_pbr_metallic_roughness& pbr = src.pbr_metallic_roughness;
        dst.params.baseColorFactor = simd_make_float4(pbr.base_color_factor[0], pbr.base_color_factor[1],
                                                       pbr.base_color_factor[2], pbr.base_color_factor[3]);
        dst.params.factors.x = pbr.metallic_factor;
        dst.params.factors.y = pbr.roughness_factor;

        if (pbr.base_color_texture.texture) {
            dst.albedo = loadGltfImage(device, data, pbr.base_color_texture.texture->image, true, gltfPath, cache);
        }
        if (src.normal_texture.texture) {
            dst.normal = loadGltfImage(device, data, src.normal_texture.texture->image, false, gltfPath, cache);
        }
        if (pbr.metallic_roughness_texture.texture) {
            const cgltf_image* ormImage = pbr.metallic_roughness_texture.texture->image;
            dst.orm = loadGltfImage(device, data, ormImage, false, gltfPath, cache);
            // Occlusion is only honored when it's packed into that same image's R channel (the
            // usual glTF "ORM" layout) - a separate occlusion image is not loaded.
            if (src.occlusion_texture.texture && src.occlusion_texture.texture->image == ormImage) {
                dst.params.factors.z = src.occlusion_texture.scale;
            }
        }
    }
    return materials;
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
    std::vector<MeshSubmesh> submeshes;
    for (size_t n = 0; n < data->nodes_count; n++) {
        const cgltf_node* node = &data->nodes[n];
        if (!node->mesh) continue;
        float worldMatrix[16];
        cgltf_node_transform_world(node, worldMatrix);
        for (size_t p = 0; p < node->mesh->primitives_count; p++) {
            const cgltf_primitive& prim = node->mesh->primitives[p];
            size_t indexBefore = indexData.size();
            appendGltfPrimitive(prim, worldMatrix, vertexData, indexData);
            size_t added = indexData.size() - indexBefore;
            if (added == 0) continue;

            size_t materialIndex = prim.material ? (size_t)cgltf_material_index(data, prim.material) + 1 : 0;
            // Adjacent primitives with the same material share one submesh (one draw call).
            if (!submeshes.empty() && submeshes.back().materialIndex == materialIndex) {
                submeshes.back().indexCount += added;
            } else {
                submeshes.push_back({(NS::UInteger)indexBefore, (NS::UInteger)added, materialIndex});
            }
        }
    }

    std::vector<MeshMaterial> materials = loadGltfMaterials(device, data, path);
    cgltf_free(data);

    if (indexData.empty()) {
        fprintf(stderr, "glTF %s has no renderable triangles\n", path.c_str());
        return MeshData{};
    }
    MeshData mesh = uploadMesh(device, vertexData, indexData);
    mesh.materials = std::move(materials);
    mesh.submeshes = std::move(submeshes);
    return mesh;
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
