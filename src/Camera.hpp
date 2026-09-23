#pragma once
#include <GLFW/glfw3.h>
#include <simd/simd.h>

struct Camera {
    simd::float3 position = simd_make_float3(0.0f, 0.0f, 2.5f);
    float yaw = 0.0f;   // radians, 0 = looking down -Z
    float pitch = 0.0f; // radians, clamped to avoid gimbal flip
    float moveSpeed = 2.5f;
    float mouseSensitivity = 0.0025f;
    bool firstMouse = true;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    bool uiMode = false; // true while the cursor is free for UI interaction - suppresses mouse-look
};

// Forward direction derived from yaw/pitch; 0,0 points down -Z to match the rest of the renderer
simd::float3 cameraFront(const Camera& cam);

// Standard lookAt view matrix built from the camera's own right/up/front basis
simd::float4x4 viewMatrix(const Camera& cam);

// WASD + Space/Ctrl movement, applied each frame
void processCameraInput(GLFWwindow* window, Camera& cam, float deltaTime);

// GLFW cursor position callback - expects the Camera to be set via glfwSetWindowUserPointer
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
