#include "Frustum.hpp"
#include <cmath>

Frustum frustumFromViewProj(const simd::float4x4& viewProj) {
    // clip = viewProj * p, so row i of the matrix gives clip component i as a plane equation; a point is
    // inside when -w <= x <= w, -w <= y <= w and 0 <= z <= w, i.e. when each of the sums/differences
    // below is >= 0. (simd matrices are column-major, so row i is gathered across the columns.)
    auto row = [&](int i) {
        return simd_make_float4(viewProj.columns[0][i], viewProj.columns[1][i], viewProj.columns[2][i], viewProj.columns[3][i]);
    };
    simd::float4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);

    Frustum frustum;
    frustum.planes[0] = r3 + r0; // left
    frustum.planes[1] = r3 - r0; // right
    frustum.planes[2] = r3 + r1; // bottom
    frustum.planes[3] = r3 - r1; // top
    frustum.planes[4] = r2;      // near (z >= 0 in Metal clip space)
    frustum.planes[5] = r3 - r2; // far
    for (simd::float4& plane : frustum.planes) {
        float length = sqrtf(plane.x * plane.x + plane.y * plane.y + plane.z * plane.z);
        if (length > 0.0f) plane /= length;
    }
    return frustum;
}

WorldBounds transformBounds(simd::float3 localMin, simd::float3 localMax, const simd::float4x4& model) {
    simd::float3 localCenter = (localMin + localMax) * 0.5f;
    simd::float3 localExtents = (localMax - localMin) * 0.5f;

    simd::float4 center = model * simd_make_float4(localCenter.x, localCenter.y, localCenter.z, 1.0f);
    // The extents along each world axis: the absolute value of the model matrix's 3x3 part applied to the
    // local extents (each axis' reach is the sum of every local axis' contribution to it).
    WorldBounds bounds;
    bounds.center = simd_make_float3(center.x, center.y, center.z);
    for (int axis = 0; axis < 3; axis++) {
        bounds.extents[axis] = fabsf(model.columns[0][axis]) * localExtents.x
                             + fabsf(model.columns[1][axis]) * localExtents.y
                             + fabsf(model.columns[2][axis]) * localExtents.z;
    }
    return bounds;
}

bool frustumIntersects(const Frustum& frustum, const WorldBounds& bounds) {
    for (const simd::float4& plane : frustum.planes) {
        // The box's reach along the plane normal: if even its nearest corner is behind the plane, all of it is.
        float reach = bounds.extents.x * fabsf(plane.x) + bounds.extents.y * fabsf(plane.y) + bounds.extents.z * fabsf(plane.z);
        float distance = plane.x * bounds.center.x + plane.y * bounds.center.y + plane.z * bounds.center.z + plane.w;
        if (distance < -reach) return false;
    }
    return true;
}
