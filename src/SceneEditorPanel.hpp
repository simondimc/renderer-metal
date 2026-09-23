#pragma once
#include "Scene.hpp"

// Draws the "Scene Editor" ImGui window: object list, add/remove, save/load, transform sliders
// for the currently selected object. selectedIndex is owned by the caller and persists across
// frames (index into scene.objects, or -1 when nothing is selected).
void drawSceneEditorPanel(Scene& scene, int& selectedIndex);
