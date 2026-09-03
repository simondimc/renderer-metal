#pragma once
#include <GLFW/glfw3.h>

// A clean C++ function that takes the GLFW window and the raw Metal device pointer
void setupMetalLayerForWindow(GLFWwindow* window, void* metalDevicePtr);
