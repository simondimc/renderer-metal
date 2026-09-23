#include "UI.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_metal.h"

void initUI(GLFWwindow* window, MTL::Device* device) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    // "ForOther" because we're not using ImGui's OpenGL/Vulkan GLFW glue - Metal draw calls are
    // issued manually in endUIFrame() through the same render pass as the 3D scene.
    ImGui_ImplGlfw_InitForOther(window, true);
    ImGui_ImplMetal_Init(device);
}

void shutdownUI() {
    ImGui_ImplMetal_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void beginUIFrame(MTL::RenderPassDescriptor* renderPassDescriptor) {
    ImGui_ImplMetal_NewFrame(renderPassDescriptor);
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void endUIFrame(MTL::CommandBuffer* cmdBuffer, MTL::RenderCommandEncoder* encoder) {
    ImGui::Render();
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), cmdBuffer, encoder);
}
