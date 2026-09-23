#pragma once
#include <simd/simd.h>
#include <string>
#include <vector>

enum class SceneObjectType { Cube, Light };

// A single instance in the scene - either a cube (mesh/textures are shared, only the transform
// differs) or a point light (position + color/intensity, no mesh).
// Plain float[3] (not simd::float3) so ImGui::DragFloat3 can take its address directly -
// simd::float3 is a compiler vector-extension type and its lanes aren't addressable.
struct SceneObject {
    SceneObjectType type = SceneObjectType::Cube;
    float position[3] = {0.0f, 0.0f, 0.0f};
    float rotationDegrees[3] = {0.0f, 0.0f, 0.0f}; // Cube only
    float scale[3] = {1.0f, 1.0f, 1.0f};           // Cube only
    float color[3] = {1.0f, 1.0f, 1.0f};           // Light only
    float intensity = 3.0f;                        // Light only
    std::string name = "Cube";
};

struct Scene {
    std::vector<SceneObject> objects;
};

// Caps the per-frame GPU uniform buffer sizing in Main.cpp (each object gets its own aligned slot)
constexpr size_t kMaxSceneObjects = 256;

// Builds a local-to-world model matrix (scale -> rotate -> translate) for one Cube object
simd::float4x4 objectModelMatrix(const SceneObject& obj);

// Simple line-based text format: one "object"/"light" block per line group, see Scene.cpp for the
// exact grammar. Returns false (and logs to stderr) on failure; the scene is left unmodified in
// that case.
bool saveScene(const Scene& scene, const std::string& path);
bool loadScene(Scene& scene, const std::string& path);
