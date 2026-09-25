#pragma once
#include "Environment.hpp"
#include "Scene.hpp"
#include "TextureLibrary.hpp"

// Draws the "Scene Editor" ImGui window: object list, add/remove, save/load, transform sliders
// for the currently selected object. selectedIndex is owned by the caller and persists across
// frames (index into scene.objects, or -1 when nothing is selected). textureLibrary supplies the
// texture sets offered in a material's "Texture Set" list, environmentLibrary the environments offered
// in the "Environment" list.
void drawSceneEditorPanel(Scene& scene, int& selectedIndex, TextureLibrary& textureLibrary,
                          EnvironmentLibrary& environmentLibrary);
