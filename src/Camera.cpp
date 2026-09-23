#include "Camera.hpp"
#include <cmath>

simd::float3 cameraFront(const Camera& cam) {
    return simd_normalize(simd_make_float3(
        sinf(cam.yaw) * cosf(cam.pitch),
        sinf(cam.pitch),
        -cosf(cam.yaw) * cosf(cam.pitch)
    ));
}

simd::float4x4 viewMatrix(const Camera& cam) {
    simd::float3 front = cameraFront(cam);
    simd::float3 worldUp = simd_make_float3(0.0f, 1.0f, 0.0f);
    simd::float3 right = simd_normalize(simd_cross(front, worldUp));
    simd::float3 up = simd_cross(right, front);

    return simd_matrix(
        simd_make_float4(right.x, up.x, -front.x, 0.0f),
        simd_make_float4(right.y, up.y, -front.y, 0.0f),
        simd_make_float4(right.z, up.z, -front.z, 0.0f),
        simd_make_float4(-simd_dot(right, cam.position), -simd_dot(up, cam.position), simd_dot(front, cam.position), 1.0f)
    );
}

void processCameraInput(GLFWwindow* window, Camera& cam, float deltaTime) {
    simd::float3 front = cameraFront(cam);
    simd::float3 worldUp = simd_make_float3(0.0f, 1.0f, 0.0f);
    simd::float3 right = simd_normalize(simd_cross(front, worldUp));

    float velocity = cam.moveSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) cam.position += front * velocity;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cam.position -= front * velocity;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) cam.position -= right * velocity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) cam.position += right * velocity;
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) cam.position += worldUp * velocity;
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) cam.position -= worldUp * velocity;
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    Camera* cam = static_cast<Camera*>(glfwGetWindowUserPointer(window));
    if (cam->uiMode) return; // cursor is free for the UI - don't steer the camera

    if (cam->firstMouse) {
        cam->lastMouseX = xpos;
        cam->lastMouseY = ypos;
        cam->firstMouse = false;
    }

    double dx = xpos - cam->lastMouseX;
    double dy = cam->lastMouseY - ypos; // reversed: screen Y grows downward
    cam->lastMouseX = xpos;
    cam->lastMouseY = ypos;

    cam->yaw += (float)dx * cam->mouseSensitivity;
    cam->pitch += (float)dy * cam->mouseSensitivity;

    const float maxPitch = 1.5533f; // ~89 degrees
    cam->pitch = fmaxf(-maxPitch, fminf(maxPitch, cam->pitch));
}
