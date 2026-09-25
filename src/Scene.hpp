#pragma once
#include <simd/simd.h>
#include <string>
#include <vector>

enum class SceneObjectType { Cube, Light, Mesh };

// Point: position + color/intensity only, radiates equally in all directions.
// Directional: no position (infinitely far away); direction only, no distance falloff - a sun.
// Spot: position + direction + cone angles - a point light restricted to a cone.
// Area: position + orientation + width/height - a rectangular emitter (soft-point approximation,
// see sampleAreaLight in Shader.metal; not physically-based, no soft shadows).
enum class LightType { Point, Directional, Spot, Area };

// Clamp: the old behavior - hard per-channel clip at 1.0, no curve. Reinhard: x/(1+x), a simple
// classic rolloff. ACES: Narkowicz's fit to the ACES filmic reference curve, punchier contrast.
// Uncharted2: Hable's filmic curve (as used in the game), a softer shoulder than ACES. Must match
// the order of TONE_MAP_* defines in Shader.metal.
enum class ToneMapOperator { Clamp, Reinhard, ACES, Uncharted2 };

// Metallic-roughness PBR material (see fragmentMain in Shader.metal). There is a single global
// albedo/normal/roughness/metallic texture set for now (loaded in Main.cpp), so materials don't pick
// textures per object yet - useTextures just chooses between that set and flat scalar values.
//   useTextures = true:  final albedo = texture * albedo, metallic = metalTexture * metallic,
//                        roughness = roughTexture * roughness, and the normal map is applied. The
//                        defaults (all 1) therefore leave the textures untouched.
//   useTextures = false: albedo/metallic/roughness are used as-is, and the normal map is skipped.
struct Material {
    float albedo[3] = {1.0f, 1.0f, 1.0f};
    float metallic = 1.0f;   // 0 = dielectric, 1 = metal
    float roughness = 1.0f;  // 0 = mirror-smooth, 1 = fully rough
    float ao = 1.0f;         // ambient occlusion multiplier on the (still constant) ambient term
    bool useTextures = true;
};

// A single instance in the scene - either a cube (mesh/textures are shared, only the transform
// differs) or a light (no mesh; see LightType for the supported kinds).
// Plain float[3] (not simd::float3) so ImGui::DragFloat3 can take its address directly -
// simd::float3 is a compiler vector-extension type and its lanes aren't addressable.
struct SceneObject {
    SceneObjectType type = SceneObjectType::Cube;
    float position[3] = {0.0f, 0.0f, 0.0f};
    // Cube: mesh rotation. Light: orientation for Directional/Spot/Area, unused for Point - see
    // objectForward/objectRight/objectUp (forward, i.e. 0,0,-1 rotated, is the emission direction).
    float rotationDegrees[3] = {0.0f, 0.0f, 0.0f};
    float scale[3] = {1.0f, 1.0f, 1.0f};           // Cube only
    float color[3] = {1.0f, 1.0f, 1.0f};           // Light only
    float intensity = 3.0f;                        // Light only
    LightType lightType = LightType::Point;        // Light only
    float spotInnerDegrees = 15.0f;                 // Spot only: half-angle of the full-bright cone
    float spotOuterDegrees = 25.0f;                 // Spot only: half-angle where light reaches zero
    float areaSize[2] = {1.0f, 1.0f};               // Area only: width/height of the rectangle
    Material material;                              // Cube/Mesh only
    std::string name = "Cube";
    std::string meshPath;                           // Mesh only: .obj/.gltf/.glb path, relative to build/
};

struct Scene {
    std::vector<SceneObject> objects;
    // Global HDR exposure multiplier applied before tone mapping (see fragmentMain in
    // Shader.metal) - scales linear scene radiance up/down before the curve compresses it into
    // displayable range, the same role a camera's exposure setting plays.
    float exposure = 1.0f;
    // Which curve fragmentMain uses to compress exposed HDR radiance into [0, 1] before gamma
    // encoding - see ToneMapOperator above.
    ToneMapOperator toneMapOperator = ToneMapOperator::ACES;

    // Post-processing (see postProcessFragmentMain in Shader.metal). All default to neutral/off so
    // an existing scene's look doesn't change until these are dialed in from the Scene Editor.
    float vignetteStrength = 0.0f;            // 0 = off, higher = darker/tighter screen edges
    float chromaticAberrationStrength = 0.0f; // 0 = off, per-channel UV offset growing toward the edges
    float filmGrainStrength = 0.0f;           // 0 = off, blended noise amount
    float sharpenStrength = 0.0f;             // 0 = off, unsharp-mask amount
    float colorGradingSaturation = 1.0f;      // 1 = neutral, 0 = grayscale
    float colorGradingContrast = 1.0f;        // 1 = neutral
    // Bloom: bright-pass threshold (linear HDR radiance above this value bleeds into the blur) and
    // blend intensity (0 = off) - see the bloomExtract/blur passes in Main.cpp and Shader.metal.
    float bloomThreshold = 1.0f;
    float bloomIntensity = 0.0f;
    // Depth of Field: world-space distance from the camera that stays sharp, the range either
    // side of it that also stays sharp, and the blend strength of the out-of-focus blur (0 = off)
    // - see postProcessFragmentMain in Shader.metal.
    float dofFocusDistance = 10.0f;
    float dofFocusRange = 5.0f;
    float dofStrength = 0.0f;
    // Motion Blur: camera-only (no per-object velocity data) - 0 = off, higher smears further
    // along the camera's own frame-to-frame motion. See Main.cpp's reprojectionMatrix.
    float motionBlurStrength = 0.0f;
    // Lens Flare: 0 = off - glow + ghost artifacts for lights on-screen and unoccluded. See
    // LensFlareLight in Shader.metal and Main.cpp's per-light screen projection.
    float lensFlareStrength = 0.0f;
};

// Caps the per-frame GPU uniform buffer sizing in Main.cpp (each object gets its own aligned slot)
constexpr size_t kMaxSceneObjects = 256;

// Relative to the build/ directory, where run.sh/clean_run.sh launch the binary from
constexpr const char* kSceneFilePath = "../scene/scene.txt";

// Builds a local-to-world model matrix (scale -> rotate -> translate) for one Cube object
simd::float4x4 objectModelMatrix(const SceneObject& obj);

// World-space local axes after applying obj.rotationDegrees only (no translation/scale) - used to
// orient Directional/Spot/Area lights. Forward is -Z, matching Camera.hpp's convention.
simd::float3 objectForward(const SceneObject& obj);
simd::float3 objectRight(const SceneObject& obj);
simd::float3 objectUp(const SceneObject& obj);

// Simple line-based text format: one "object"/"light" block per line group, see Scene.cpp for the
// exact grammar. Returns false (and logs to stderr) on failure; the scene is left unmodified in
// that case.
bool saveScene(const Scene& scene, const std::string& path);
bool loadScene(Scene& scene, const std::string& path);
