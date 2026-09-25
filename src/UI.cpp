#include "UI.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_metal.h"
#include <algorithm>
#include <vector>

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

namespace {
bool shaderReloadShown = false;
bool shaderReloadSucceeded = false;
double shaderReloadTime = 0.0;
std::string shaderReloadLine;
constexpr double kShaderReloadSuccessSeconds = 2.5;
} // namespace

void showShaderReloadResult(bool succeeded, const std::string& message) {
    shaderReloadShown = true;
    shaderReloadSucceeded = succeeded;
    shaderReloadTime = ImGui::GetTime();
    // The compiler's output starts with a line like "program_source:12:5: error: ..." - that one says enough.
    shaderReloadLine = message.substr(0, message.find('\n'));
    if (shaderReloadLine.size() > 90) shaderReloadLine.resize(90);
}

void drawPerformanceOverlay(float deltaTime, const GpuTimer& gpuTimer, bool showPassBreakdown) {
    // Averaged over a window (rather than shown per frame) so the numbers stay readable: the text
    // only changes every kRefreshSeconds, from the frames and time accumulated since the last change.
    constexpr float kRefreshSeconds = 0.5f;
    static float accumulatedSeconds = 0.0f;
    static int accumulatedFrames = 0;
    static float shownFps = 0.0f;
    static float shownFrameMs = 0.0f;

    accumulatedSeconds += deltaTime;
    accumulatedFrames++;
    if (accumulatedSeconds >= kRefreshSeconds) {
        shownFps = (float)accumulatedFrames / accumulatedSeconds;
        shownFrameMs = 1000.0f * accumulatedSeconds / (float)accumulatedFrames;
        accumulatedSeconds = 0.0f;
        accumulatedFrames = 0;
    }

    constexpr float kMargin = 10.0f;
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - kMargin, kMargin), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.35f);
    ImGui::Begin("##fps", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                 ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoInputs);
    ImGui::Text("%.0f FPS", shownFps);
    ImGui::Text("%.2f ms", shownFrameMs);

    std::vector<GpuTimer::Entry> passes;
    float gpuMilliseconds = 0.0f;
    gpuTimer.snapshot(passes, gpuMilliseconds);
    ImGui::Text("GPU %.2f ms", gpuMilliseconds);

    if (shaderReloadShown) {
        if (!shaderReloadSucceeded) {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.3f, 1.0f), "Shader error (terminal has the full text):");
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.3f, 1.0f), "%s", shaderReloadLine.c_str());
        } else if (ImGui::GetTime() - shaderReloadTime < kShaderReloadSuccessSeconds) {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.5f, 1.0f), "Shaders reloaded");
        }
    }

    if (showPassBreakdown) {
        if (!gpuTimer.passTimingSupported()) {
            ImGui::TextDisabled("per-pass timing unsupported");
        } else {
            std::sort(passes.begin(), passes.end(),
                      [](const GpuTimer::Entry& a, const GpuTimer::Entry& b) { return a.milliseconds > b.milliseconds; });
            ImGui::Separator();
            if (ImGui::BeginTable("##passes", 2, ImGuiTableFlags_SizingFixedFit)) {
                for (const GpuTimer::Entry& pass : passes) {
                    if (pass.milliseconds < 0.005f) continue; // a disabled effect's fading-out entry
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(pass.label.c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("%.2f", pass.milliseconds);
                }
                ImGui::EndTable();
            }
        }
        ImGui::TextDisabled("F3: hide");
    }
    ImGui::End();
}

void endUIFrame(MTL::CommandBuffer* cmdBuffer, MTL::RenderCommandEncoder* encoder) {
    ImGui::Render();
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), cmdBuffer, encoder);
}
