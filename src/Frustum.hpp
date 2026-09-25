#pragma once
#include <simd/simd.h>

// View-frustum culling: skipping the draw of anything that cannot end up inside a pass's view volume - the
// camera's for the scene passes, a light's for its shadow map (a shadow caster outside the camera's view
// can still cast into it, so shadow passes are culled against the light's frustum, never the camera's).

// The six planes of a view volume, each (a, b, c, d) with a unit normal (a, b, c) pointing inward: a point
// is inside the plane when a*x + b*y + c*z + d >= 0.
struct Frustum {
    simd::float4 planes[6]; // left, right, bottom, top, near, far
};

// Planes of the volume a view-projection matrix maps to clip space, for Metal's conventions
// (-w <= x,y <= w, 0 <= z <= w). Works for perspective and orthographic projections alike.
Frustum frustumFromViewProj(const simd::float4x4& viewProj);

// An axis-aligned box in world space, as centre and half-size along each axis.
struct WorldBounds {
    simd::float3 center = {0.0f, 0.0f, 0.0f};
    simd::float3 extents = {0.0f, 0.0f, 0.0f};
};

// The world-space box that encloses a local-space box (localMin..localMax) after the model matrix is applied.
// Conservative for rotated objects (the box of the rotated box), which is what culling needs.
WorldBounds transformBounds(simd::float3 localMin, simd::float3 localMax, const simd::float4x4& model);

// False only when the box is entirely outside one of the planes: a box merely near or straddling the volume
// counts as inside, so culling never removes anything that could be visible.
bool frustumIntersects(const Frustum& frustum, const WorldBounds& bounds);
