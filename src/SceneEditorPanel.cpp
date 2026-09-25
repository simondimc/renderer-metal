#include "SceneEditorPanel.hpp"
#include "imgui.h"
#include <algorithm>
#include <cstring>
#include <string>

namespace {

int countOfType(const Scene& scene, SceneObjectType type) {
    int count = 0;
    for (const auto& obj : scene.objects) {
        if (obj.type == type) count++;
    }
    return count;
}

// glTF meshes bring their own materials and textures, so a texture set only applies to everything else
bool usesTextureSet(const SceneObject& obj) {
    if (obj.type != SceneObjectType::Mesh) return true;
    auto endsWith = [&](const char* ext) {
        size_t n = strlen(ext);
        return obj.meshPath.size() >= n && obj.meshPath.compare(obj.meshPath.size() - n, n, ext) == 0;
    };
    return !endsWith(".gltf") && !endsWith(".glb");
}

// Order must match ToneMapOperator in Scene.hpp
constexpr const char* kToneMapOperatorNames[] = {"Clamp", "Reinhard", "ACES", "Uncharted2"};
// Order must match AmbientOcclusionMode in Scene.hpp
constexpr const char* kAmbientOcclusionModeNames[] = {"GTAO", "SSAO"};
} // namespace

void drawSceneEditorPanel(Scene& scene, int& selectedIndex, TextureLibrary& textureLibrary,
                          EnvironmentLibrary& environmentLibrary) {
    ImGui::Begin("Scene Editor");

    ImGui::DragFloat("Exposure", &scene.exposure, 0.01f, 0.01f, 10.0f);
    int toneMapIndex = (int)scene.toneMapOperator;
    if (ImGui::Combo("Tone Map", &toneMapIndex, kToneMapOperatorNames, IM_ARRAYSIZE(kToneMapOperatorNames))) {
        scene.toneMapOperator = (ToneMapOperator)toneMapIndex;
    }

    if (ImGui::TreeNode("Environment (IBL)")) {
        const auto& environments = environmentLibrary.names();
        bool known = std::find(environments.begin(), environments.end(), scene.environment) != environments.end();
        std::string preview = known ? scene.environment : scene.environment + " (missing)";
        if (ImGui::BeginCombo("Environment", preview.c_str())) {
            for (const std::string& name : environments) {
                if (ImGui::Selectable(name.c_str(), name == scene.environment)) {
                    scene.environment = name;
                }
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Rescan##environment")) {
            environmentLibrary.rescan();
        }
        ImGui::DragFloat("Intensity", &scene.environmentIntensity, 0.01f, 0.0f, 10.0f);
        ImGui::Checkbox("Show Sky", &scene.showSky);
        ImGui::TextDisabled("(.hdr/.exr panoramas in environment/)");
        ImGui::TreePop();
    }

    if (ImGui::TreeNode("Ambient Occlusion")) {
        int aoModeIndex = (int)scene.ambientOcclusionMode;
        if (ImGui::Combo("AO Mode", &aoModeIndex, kAmbientOcclusionModeNames, IM_ARRAYSIZE(kAmbientOcclusionModeNames))) {
            scene.ambientOcclusionMode = (AmbientOcclusionMode)aoModeIndex;
        }
        ImGui::DragFloat("AO Strength", &scene.ambientOcclusionStrength, 0.01f, 0.0f, 4.0f);
        ImGui::DragFloat("AO Radius", &scene.ambientOcclusionRadius, 0.01f, 0.05f, 5.0f);
        ImGui::TextDisabled("(0 strength = off)");
        ImGui::TreePop();
    }

    if (ImGui::TreeNode("Post-Processing")) {
        ImGui::DragFloat("Vignette", &scene.vignetteStrength, 0.01f, 0.0f, 2.0f);
        ImGui::DragFloat("Chromatic Aberration", &scene.chromaticAberrationStrength, 0.01f, 0.0f, 2.0f);
        ImGui::DragFloat("Film Grain", &scene.filmGrainStrength, 0.001f, 0.0f, 0.5f);
        ImGui::DragFloat("Sharpen", &scene.sharpenStrength, 0.01f, 0.0f, 2.0f);
        ImGui::DragFloat("Saturation", &scene.colorGradingSaturation, 0.01f, 0.0f, 2.0f);
        ImGui::DragFloat("Contrast", &scene.colorGradingContrast, 0.01f, 0.0f, 2.0f);
        ImGui::DragFloat("Bloom Threshold", &scene.bloomThreshold, 0.01f, 0.0f, 10.0f);
        ImGui::DragFloat("Bloom Intensity", &scene.bloomIntensity, 0.01f, 0.0f, 5.0f);
        ImGui::DragFloat("DoF Focus Distance", &scene.dofFocusDistance, 0.1f, 0.0f, 100.0f);
        ImGui::DragFloat("DoF Focus Range", &scene.dofFocusRange, 0.1f, 0.01f, 50.0f);
        ImGui::DragFloat("DoF Strength", &scene.dofStrength, 0.01f, 0.0f, 1.0f);
        ImGui::DragFloat("Motion Blur", &scene.motionBlurStrength, 0.01f, 0.0f, 4.0f);
        ImGui::DragFloat("Lens Flare", &scene.lensFlareStrength, 0.01f, 0.0f, 5.0f);
        ImGui::TreePop();
    }
    ImGui::Separator();

    ImGui::BeginDisabled(scene.objects.size() >= kMaxSceneObjects);
    if (ImGui::Button("Add Cube")) {
        SceneObject obj;
        obj.type = SceneObjectType::Cube;
        obj.name = "Cube " + std::to_string(countOfType(scene, SceneObjectType::Cube) + 1);
        scene.objects.push_back(obj);
        selectedIndex = (int)scene.objects.size() - 1;
    }
    ImGui::SameLine();
    if (ImGui::Button("Add Light")) {
        SceneObject obj;
        obj.type = SceneObjectType::Light;
        obj.name = "Light " + std::to_string(countOfType(scene, SceneObjectType::Light) + 1);
        obj.position[0] = 2.0f;
        obj.position[1] = 4.0f;
        obj.position[2] = 2.0f;
        scene.objects.push_back(obj);
        selectedIndex = (int)scene.objects.size() - 1;
    }
    ImGui::SameLine();
    if (ImGui::Button("Add Mesh")) {
        SceneObject obj;
        obj.type = SceneObjectType::Mesh;
        obj.name = "Mesh " + std::to_string(countOfType(scene, SceneObjectType::Mesh) + 1);
        scene.objects.push_back(obj);
        selectedIndex = (int)scene.objects.size() - 1;
    }
    ImGui::EndDisabled();

    bool hasSelection = selectedIndex >= 0 && selectedIndex < (int)scene.objects.size();
    ImGui::SameLine();
    ImGui::BeginDisabled(!hasSelection);
    if (ImGui::Button("Remove Selected")) {
        scene.objects.erase(scene.objects.begin() + selectedIndex);
        selectedIndex = -1;
        hasSelection = false;
    }
    ImGui::EndDisabled();

    ImGui::Separator();

    if (ImGui::Button("Save")) {
        saveScene(scene, kSceneFilePath);
    }
    ImGui::SameLine();
    if (ImGui::Button("Load")) {
        loadScene(scene, kSceneFilePath);
        selectedIndex = -1;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(%s)", kSceneFilePath);

    ImGui::Separator();
    ImGui::Text("Objects (%d)", (int)scene.objects.size());
    ImGui::BeginChild("ObjectList", ImVec2(0, 120), true);
    for (int i = 0; i < (int)scene.objects.size(); i++) {
        ImGui::PushID(i);
        const char* tag = (scene.objects[i].type == SceneObjectType::Light) ? "[L] "
                        : (scene.objects[i].type == SceneObjectType::Mesh) ? "[M] " : "[C] ";
        std::string label = tag + scene.objects[i].name;
        if (ImGui::Selectable(label.c_str(), i == selectedIndex)) {
            selectedIndex = i;
        }
        ImGui::PopID();
    }
    ImGui::EndChild();

    ImGui::Separator();
    if (hasSelection) {
        SceneObject& obj = scene.objects[selectedIndex];
        ImGui::Text("%s - %s", obj.type == SceneObjectType::Light ? "Light" : "Transform", obj.name.c_str());
        ImGui::DragFloat3("Position", obj.position, 0.05f);
        if (obj.type == SceneObjectType::Light) {
            static const char* kLightTypeNames[] = {"Point", "Directional", "Spot", "Area"};
            int lightTypeIndex = (int)obj.lightType;
            if (ImGui::Combo("Light Type", &lightTypeIndex, kLightTypeNames, IM_ARRAYSIZE(kLightTypeNames))) {
                obj.lightType = (LightType)lightTypeIndex;
            }
            ImGui::ColorEdit3("Color", obj.color);
            ImGui::DragFloat("Intensity", &obj.intensity, 0.1f, 0.0f, 50.0f);
            if (obj.lightType != LightType::Point) {
                ImGui::DragFloat3("Direction (rotation)", obj.rotationDegrees, 1.0f);
            }
            if (obj.lightType == LightType::Spot) {
                ImGui::DragFloat("Inner Angle", &obj.spotInnerDegrees, 0.5f, 0.0f, obj.spotOuterDegrees);
                ImGui::DragFloat("Outer Angle", &obj.spotOuterDegrees, 0.5f, obj.spotInnerDegrees, 89.0f);
            }
            if (obj.lightType == LightType::Area) {
                ImGui::DragFloat2("Size", obj.areaSize, 0.05f, 0.01f, 20.0f);
            }
        } else {
            ImGui::DragFloat3("Rotation", obj.rotationDegrees, 1.0f);
            ImGui::DragFloat3("Scale", obj.scale, 0.05f, 0.01f, 10.0f);
            if (ImGui::TreeNodeEx("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
                Material& m = obj.material;
                ImGui::Checkbox("Use Textures", &m.useTextures);
                if (usesTextureSet(obj)) {
                    const auto& sets = textureLibrary.setNames();
                    bool known = std::find(sets.begin(), sets.end(), m.textureSet) != sets.end();
                    std::string preview = known ? m.textureSet : m.textureSet + " (missing)";
                    ImGui::BeginDisabled(!m.useTextures);
                    if (ImGui::BeginCombo("Texture Set", preview.c_str())) {
                        for (const std::string& name : sets) {
                            if (ImGui::Selectable(name.c_str(), name == m.textureSet)) {
                                m.textureSet = name;
                            }
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::EndDisabled();
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Rescan")) {
                        textureLibrary.rescan();
                    }
                } else {
                    ImGui::TextDisabled("Texture set: from the glTF file");
                }
                ImGui::ColorEdit3("Albedo", m.albedo);
                ImGui::SliderFloat("Metallic", &m.metallic, 0.0f, 1.0f);
                ImGui::SliderFloat("Roughness", &m.roughness, 0.0f, 1.0f);
                ImGui::SliderFloat("AO", &m.ao, 0.0f, 1.0f);
                ImGui::TreePop();
            }
            if (obj.type == SceneObjectType::Mesh) {
                char pathBuf[256];
                strncpy(pathBuf, obj.meshPath.c_str(), sizeof(pathBuf) - 1);
                pathBuf[sizeof(pathBuf) - 1] = '\0';
                if (ImGui::InputText("Mesh Path", pathBuf, sizeof(pathBuf))) {
                    obj.meshPath = pathBuf;
                }
                ImGui::TextDisabled("(.obj/.gltf/.glb, relative to build/)");
            }
        }
    } else {
        ImGui::TextDisabled("No object selected");
    }

    ImGui::End();
}
