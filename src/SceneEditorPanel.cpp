#include "SceneEditorPanel.hpp"
#include "imgui.h"
#include <string>

namespace {

int countOfType(const Scene& scene, SceneObjectType type) {
    int count = 0;
    for (const auto& obj : scene.objects) {
        if (obj.type == type) count++;
    }
    return count;
}
} // namespace

void drawSceneEditorPanel(Scene& scene, int& selectedIndex) {
    ImGui::Begin("Scene Editor");

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
        const char* tag = (scene.objects[i].type == SceneObjectType::Light) ? "[L] " : "[C] ";
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
            ImGui::ColorEdit3("Color", obj.color);
            ImGui::DragFloat("Intensity", &obj.intensity, 0.1f, 0.0f, 50.0f);
        } else {
            ImGui::DragFloat3("Rotation", obj.rotationDegrees, 1.0f);
            ImGui::DragFloat3("Scale", obj.scale, 0.05f, 0.01f, 10.0f);
        }
    } else {
        ImGui::TextDisabled("No object selected");
    }

    ImGui::End();
}
