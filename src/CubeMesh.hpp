#pragma once
#include <cstddef>
#include <cstdint>

// Standard Right-Handed Cube: +Z is Front (toward viewer), -Z is Back (away)
// Each face gets its own 4 vertices (24 total) so every face can have its own 0..1 UV range.
// Layout per vertex: position(3) + uv(2) + normal(3) + tangent(4) = 12 floats.
// Tangent xyz = world-space direction of increasing U (needed to build the TBN basis for normal mapping);
// w = handedness (+1/-1): the bitangent is cross(normal, tangent.xyz) * w, as in glTF. All +1 here since
// the cube's UVs are never mirrored.
namespace CubeMesh {

constexpr size_t vertexStrideFloats = 12;

constexpr float vertices[] = {
    // Front (Z = 0.5)               normal              tangent + w
    -0.5f,  0.5f,  0.5f,   0.0f, 1.0f,   0.0f, 0.0f, 1.0f,   1.0f, 0.0f, 0.0f, 1.0f,
    0.5f,  0.5f,  0.5f,   1.0f, 1.0f,   0.0f, 0.0f, 1.0f,   1.0f, 0.0f, 0.0f, 1.0f,
    -0.5f, -0.5f,  0.5f,   0.0f, 0.0f,   0.0f, 0.0f, 1.0f,   1.0f, 0.0f, 0.0f, 1.0f,
    0.5f, -0.5f,  0.5f,   1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   1.0f, 0.0f, 0.0f, 1.0f,
    // Back (Z = -0.5)
    0.5f,  0.5f, -0.5f,   0.0f, 1.0f,   0.0f, 0.0f, -1.0f,  -1.0f, 0.0f, 0.0f, 1.0f,
    -0.5f,  0.5f, -0.5f,   1.0f, 1.0f,   0.0f, 0.0f, -1.0f,  -1.0f, 0.0f, 0.0f, 1.0f,
    0.5f, -0.5f, -0.5f,   0.0f, 0.0f,   0.0f, 0.0f, -1.0f,  -1.0f, 0.0f, 0.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,   1.0f, 0.0f,   0.0f, 0.0f, -1.0f,  -1.0f, 0.0f, 0.0f, 1.0f,
    // Top (Y = 0.5)
    -0.5f,  0.5f, -0.5f,   0.0f, 1.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, 0.0f, 1.0f,
    0.5f,  0.5f, -0.5f,   1.0f, 1.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, 0.0f, 1.0f,
    -0.5f,  0.5f,  0.5f,   0.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, 0.0f, 1.0f,
    0.5f,  0.5f,  0.5f,   1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, 0.0f, 1.0f,
    // Bottom (Y = -0.5)
    -0.5f, -0.5f,  0.5f,   0.0f, 1.0f,   0.0f, -1.0f, 0.0f,  1.0f, 0.0f, 0.0f, 1.0f,
    0.5f, -0.5f,  0.5f,   1.0f, 1.0f,   0.0f, -1.0f, 0.0f,  1.0f, 0.0f, 0.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,   0.0f, 0.0f,   0.0f, -1.0f, 0.0f,  1.0f, 0.0f, 0.0f, 1.0f,
    0.5f, -0.5f, -0.5f,   1.0f, 0.0f,   0.0f, -1.0f, 0.0f,  1.0f, 0.0f, 0.0f, 1.0f,
    // Left (X = -0.5)
    -0.5f,  0.5f, -0.5f,   0.0f, 1.0f,   -1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f, 1.0f,
    -0.5f,  0.5f,  0.5f,   1.0f, 1.0f,   -1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,   0.0f, 0.0f,   -1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f, 1.0f,
    -0.5f, -0.5f,  0.5f,   1.0f, 0.0f,   -1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f, 1.0f,
    // Right (X = 0.5)
    0.5f,  0.5f,  0.5f,   0.0f, 1.0f,   1.0f, 0.0f, 0.0f,   0.0f, 0.0f, -1.0f, 1.0f,
    0.5f,  0.5f, -0.5f,   1.0f, 1.0f,   1.0f, 0.0f, 0.0f,   0.0f, 0.0f, -1.0f, 1.0f,
    0.5f, -0.5f,  0.5f,   0.0f, 0.0f,   1.0f, 0.0f, 0.0f,   0.0f, 0.0f, -1.0f, 1.0f,
    0.5f, -0.5f, -0.5f,   1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   0.0f, 0.0f, -1.0f, 1.0f,
};

// 12 triangles mapped Counter-Clockwise (CCW). Each face block is 4 verts: a,b,c,d -> (a,c,d) + (a,d,b)
constexpr uint16_t indices[] = {
    0, 2, 3,   0, 3, 1,     // Front
    4, 6, 7,   4, 7, 5,     // Back
    8, 10, 11, 8, 11, 9,    // Top
    12, 14, 15, 12, 15, 13, // Bottom
    16, 18, 19, 16, 19, 17, // Left
    20, 22, 23, 20, 23, 21  // Right
};

} // namespace CubeMesh
