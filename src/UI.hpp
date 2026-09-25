#pragma once
#include <Metal/Metal.hpp>
#include <GLFW/glfw3.h>
#include "GpuTimer.hpp"

// Thin wrapper around the ImGui GLFW+Metal backend lifecycle, so Main.cpp doesn't need to
// know about ImGui's init/frame/shutdown calls directly.
void initUI(GLFWwindow* window, MTL::Device* device);
void shutdownUI();

// Call once per frame, before issuing any ImGui:: widget calls
void beginUIFrame(MTL::RenderPassDescriptor* renderPassDescriptor);

// Small always-on overlay in the top-right corner: frames per second and frame time, the frame's GPU
// time, and (when showPassBreakdown) how that GPU time splits across the passes, slowest first. Call
// once per frame between beginUIFrame and endUIFrame, with that frame's duration in seconds (the full
// loop iteration, so vsync waits count - it shows what the display actually gets, while the GPU time
// is what the GPU really spent, whatever vsync does).
void drawPerformanceOverlay(float deltaTime, const GpuTimer& gpuTimer, bool showPassBreakdown);

// Call after all ImGui:: widget calls for the frame, inside the active render encoder
// (ImGui draws into the same color/depth attachments as the 3D scene)
void endUIFrame(MTL::CommandBuffer* cmdBuffer, MTL::RenderCommandEncoder* encoder);
