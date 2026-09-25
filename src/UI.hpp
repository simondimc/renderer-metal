#pragma once
#include <Metal/Metal.hpp>
#include <GLFW/glfw3.h>

// Thin wrapper around the ImGui GLFW+Metal backend lifecycle, so Main.cpp doesn't need to
// know about ImGui's init/frame/shutdown calls directly.
void initUI(GLFWwindow* window, MTL::Device* device);
void shutdownUI();

// Call once per frame, before issuing any ImGui:: widget calls
void beginUIFrame(MTL::RenderPassDescriptor* renderPassDescriptor);

// Small always-on overlay in the top-right corner: frames per second and frame time. Call once per
// frame between beginUIFrame and endUIFrame, with that frame's duration in seconds (the full loop
// iteration, so vsync waits count - it shows what the display actually gets, not just GPU cost).
void drawFpsCounter(float deltaTime);

// Call after all ImGui:: widget calls for the frame, inside the active render encoder
// (ImGui draws into the same color/depth attachments as the 3D scene)
void endUIFrame(MTL::CommandBuffer* cmdBuffer, MTL::RenderCommandEncoder* encoder);
