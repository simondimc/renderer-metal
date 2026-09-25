#include <metal_stdlib>
using namespace metal;

#define MAX_LIGHTS 4 // must match kMaxLights in Uniforms.hpp

// Must match the LightType enum in Scene.hpp
#define LIGHT_TYPE_POINT 0
#define LIGHT_TYPE_DIRECTIONAL 1
#define LIGHT_TYPE_SPOT 2
#define LIGHT_TYPE_AREA 3

// Must match the ToneMapOperator enum in Scene.hpp
#define TONE_MAP_CLAMP 0
#define TONE_MAP_REINHARD 1
#define TONE_MAP_ACES 2
#define TONE_MAP_UNCHARTED2 3

struct VertexInput {
    float3 position [[attribute(0)]];
    float2 uv       [[attribute(1)]];
    float3 normal   [[attribute(2)]];
    float3 tangent  [[attribute(3)]];
};

struct RasterData {
    float4 position [[position]];
    float2 uv;
    float3 worldPosition;
    float3 worldNormal;
    float3 worldTangent;
};

struct Uniforms {
    float4x4 mvpMatrix;
    float4x4 modelMatrix;
    float4x4 lightViewProj[MAX_LIGHTS]; // Directional/Spot shadow-map view-projection
    float4 lightPositions[MAX_LIGHTS];  // xyz = position (Point/Spot/Area)
    float4 lightDirections[MAX_LIGHTS]; // xyz = normalized emission direction (Directional/Spot/Area)
    float4 lightRight[MAX_LIGHTS];      // xyz = area light local right axis
    float4 lightUp[MAX_LIGHTS];         // xyz = area light local up axis
    float4 lightColors[MAX_LIGHTS];     // rgb = color, a = intensity
    float4 lightParams[MAX_LIGHTS];     // x=spotCosInner, y=spotCosOuter, z=areaHalfWidth, w=areaHalfHeight
    int4 lightTypes[MAX_LIGHTS];        // x = LightType of that light
    int4 lightMeta;                     // x = active light count
    float4 cameraPosition;
    float4 renderParams;                // x = exposure (see fragmentMain's tone mapping)
};

// Vertex Shader
vertex RasterData vertexMain(VertexInput in [[stage_in]],
                             constant Uniforms& uniforms [[buffer(1)]]) {
    RasterData out;
    out.position = uniforms.mvpMatrix * float4(in.position, 1.0);
    out.uv = in.uv;
    out.worldPosition = (uniforms.modelMatrix * float4(in.position, 1.0)).xyz;
    out.worldNormal = (uniforms.modelMatrix * float4(in.normal, 0.0)).xyz;
    out.worldTangent = (uniforms.modelMatrix * float4(in.tangent, 0.0)).xyz;
    return out;
}

// Cube shadow pass: rendered 6 times per light (once per cube face, see Shadow.hpp's
// computeCubeShadowMatrices and Main.cpp's per-light/per-face shadow pass loop). Rather than
// hardware depth-compare (which would need to know, at sampling time, which of the 6 faces and
// which face-local UV a world position falls into), this stores plain world-space distance from
// the light as a color value - so shading can later sample the cube by direction alone and get a
// distance to compare against, with Metal's hardware cube-map fetch (and its automatic
// cross-face filtering) handling face selection for free.
struct CubeShadowRasterData {
    float4 position [[position]];
    float3 worldPosition;
};

vertex CubeShadowRasterData cubeShadowVertexMain(VertexInput in [[stage_in]],
                                                 constant Uniforms& uniforms [[buffer(1)]],
                                                 constant float4x4& lightViewProj [[buffer(2)]]) {
    CubeShadowRasterData out;
    float4 worldPos = uniforms.modelMatrix * float4(in.position, 1.0);
    out.position = lightViewProj * worldPos;
    out.worldPosition = worldPos.xyz;
    return out;
}

fragment float cubeShadowFragmentMain(CubeShadowRasterData in [[stage_in]],
                                      constant float3& lightPosition [[buffer(3)]]) {
    return length(in.worldPosition - lightPosition);
}

// 1.0 = fully lit, 0.0 = fully in shadow. Sampling by direction (rather than a light-space
// projected UV) means every direction around the light is valid - no "outside the frustum" case
// to special-case, unlike a single-frustum shadow map.
static float sampleCubeShadow(float3 worldPosition, float3 lightPosition,
                              texturecube<float> shadowCube, sampler shadowSampler) {
    float3 fragToLight = worldPosition - lightPosition;
    float currentDistance = length(fragToLight);
    float closestDistance = shadowCube.sample(shadowSampler, fragToLight).r;

    constexpr float bias = 0.05; // world-space units; mitigates shadow acne from limited face resolution
    return (currentDistance - bias > closestDistance) ? 0.0 : 1.0;
}

// Single-frustum shadow pass for Directional/Spot lights: reuses cubeShadowVertexMain/
// cubeShadowFragmentMain above unchanged (they just need *some* lightViewProj and reference
// point, cube or not) - rendered into a plain 2D texture instead of one cube face. Kept as
// world-space distance, like the cube shadow, rather than hardware depth: hardware depth from a
// perspective (Spot) projection is extremely non-linear, so a fixed depth-space bias would need to
// be tiny near the light and huge far from it - no single value works across the frustum. A flat
// world-space distance bias (the same value sampleCubeShadow already uses) doesn't have that
// problem.
//
// Must match kDirectionalShadowDistance in Shadow.hpp - see computeDirectionalShadowMatrix. A
// directional light has no true position, so its shadow pass renders from a virtual eye this far
// behind the light object's `position` (reused as the shadow volume's anchor); sampling needs that
// same eye as its distance reference point, so it's rederived here rather than passed through.
#define DIRECTIONAL_SHADOW_DISTANCE 25.0

// 1.0 = fully lit, 0.0 = fully in shadow, via classic projective shadow mapping: reproject the
// shading point into the light's clip space to find where it landed in the shadow map, then
// compare distances-from-eye (not depth) for the reason above. Points outside the frustum (or
// behind the camera, for a spot) read as lit rather than shadowed, since nothing was rendered
// there to compare against.
static float sampleProjectedShadow(float3 worldPosition, float3 shadowEye, float4x4 lightViewProj,
                                   texture2d<float> shadowMap, sampler shadowSampler) {
    float4 lightSpace = lightViewProj * float4(worldPosition, 1.0);
    if (lightSpace.w <= 0.0) return 1.0;
    float3 ndc = lightSpace.xyz / lightSpace.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) return 1.0;

    float2 uv = float2(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5); // NDC +Y up -> texture V down
    float closestDistance = shadowMap.sample(shadowSampler, uv).r;
    float currentDistance = length(worldPosition - shadowEye);

    constexpr float bias = 0.05; // world-space units, same as sampleCubeShadow's
    return (currentDistance - bias > closestDistance) ? 0.0 : 1.0;
}

// Four interchangeable ways to compress unbounded linear HDR radiance down to [0, 1] before gamma
// encoding (see fragmentMain and toneMap below, and ToneMapOperator in Scene.hpp). All operate
// per-channel, so a strongly single-hued light can still push that channel toward 1.0 faster than
// the others - none of these are luminance-preserving desaturating operators.

// The pre-HDR behavior: hard per-channel clip at 1.0. No highlight rolloff, so anything overbright
// blows out to flat white/primary colors.
static float3 clampToneMap(float3 color) {
    return saturate(color);
}

// Classic x / (1 + x): a simple, cheap rolloff that approaches but never reaches 1.0. Softer and
// less contrasty than the filmic curves below - midtones get compressed more, since the curve
// starts bending immediately rather than staying linear at low values.
static float3 reinhardToneMap(float3 color) {
    return color / (1.0 + color);
}

// Narkowicz's fit to the ACES filmic reference curve: stays closer to linear through the midtones
// than Reinhard, with a punchier, more contrasty shoulder into the highlights.
static float3 acesFilmicToneMap(float3 color) {
    constexpr float a = 2.51;
    constexpr float b = 0.03;
    constexpr float c = 2.43;
    constexpr float d = 0.59;
    constexpr float e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

// Hable's filmic curve (as used in Uncharted 2): a longer, gentler shoulder than ACES, so bright
// highlights roll off more gradually instead of compressing hard near 1.0. Needs its own white-
// point normalization (whiteScale) since the raw curve doesn't map input 1.0 to output 1.0.
static float3 uncharted2Partial(float3 x) {
    constexpr float A = 0.15;
    constexpr float B = 0.50;
    constexpr float C = 0.10;
    constexpr float D = 0.20;
    constexpr float E = 0.02;
    constexpr float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

static float3 uncharted2ToneMap(float3 color) {
    constexpr float W = 11.2; // linear white point the curve is normalized against
    float3 curved = uncharted2Partial(color);
    float3 whiteScale = 1.0 / uncharted2Partial(float3(W));
    return saturate(curved * whiteScale);
}

static float3 toneMap(float3 color, int op) {
    if (op == TONE_MAP_CLAMP) return clampToneMap(color);
    if (op == TONE_MAP_REINHARD) return reinhardToneMap(color);
    if (op == TONE_MAP_UNCHARTED2) return uncharted2ToneMap(color);
    return acesFilmicToneMap(color); // TONE_MAP_ACES, and the default for any unrecognized value
}

// Fragment Shader
fragment float4 fragmentMain(RasterData in [[stage_in]],
                             constant Uniforms& uniforms [[buffer(1)]],
                             texture2d<float> tex [[texture(0)]],
                             texture2d<float> normalMap [[texture(1)]],
                             array<texturecube<float>, MAX_LIGHTS> shadowCubes [[texture(2)]],
                             array<texture2d<float>, MAX_LIGHTS> shadow2DMaps [[texture(2 + MAX_LIGHTS)]],
                             sampler smp [[sampler(0)]],
                             sampler shadowSampler [[sampler(1)]]) {
    constexpr float ambientStrength = 0.15;
    constexpr float specularStrength = 0.5;
    constexpr float shininess = 32.0;

    // Build the TBN basis and use it to rotate the tangent-space normal map sample into world space
    float3 N = normalize(in.worldNormal);
    float3 T = normalize(in.worldTangent);
    float3 B = cross(N, T);
    float3x3 TBN = float3x3(T, B, N);

    float3 tangentNormal = normalMap.sample(smp, in.uv).rgb * 2.0 - 1.0;
    float3 normal = normalize(TBN * tangentNormal);

    float3 viewDir = normalize(uniforms.cameraPosition.xyz - in.worldPosition);
    float4 texColor = tex.sample(smp, in.uv);
    float3 litColor = texColor.rgb * ambientStrength;

    int lightCount = uniforms.lightMeta.x;
    for (int i = 0; i < lightCount; i++) {
        int lightType = uniforms.lightTypes[i].x;
        float3 lightPos = uniforms.lightPositions[i].xyz;
        float3 emitDir = uniforms.lightDirections[i].xyz; // direction the light travels outward

        float3 lightDir;  // surface -> light, normalized
        float lightDist = 0.0;
        float attenuation = 1.0;

        if (lightType == LIGHT_TYPE_DIRECTIONAL) {
            // No position/distance - the source is treated as infinitely far away.
            lightDir = -emitDir;
        } else if (lightType == LIGHT_TYPE_AREA) {
            // Closest point on the rectangle to the shading point ("most representative point"),
            // treated like a point light placed there. An approximation, not physically-based -
            // it doesn't reproduce the soft penumbra a real area light would cast.
            float3 right = uniforms.lightRight[i].xyz;
            float3 up = uniforms.lightUp[i].xyz;
            float2 halfSize = uniforms.lightParams[i].zw;
            float3 toPoint = in.worldPosition - lightPos;
            float2 local = clamp(float2(dot(toPoint, right), dot(toPoint, up)), -halfSize, halfSize);
            float3 closest = lightPos + right * local.x + up * local.y;
            float3 toLight = closest - in.worldPosition;
            lightDist = length(toLight);
            lightDir = toLight / max(lightDist, 1e-4);
            // Only the front face of the rectangle emits light.
            attenuation = max(dot(-lightDir, emitDir), 0.0);
        } else {
            // Point and Spot both radiate from a world-space position.
            float3 toLight = lightPos - in.worldPosition;
            lightDist = length(toLight);
            lightDir = toLight / max(lightDist, 1e-4);
            if (lightType == LIGHT_TYPE_SPOT) {
                float cosAngle = dot(-lightDir, emitDir);
                float cosInner = uniforms.lightParams[i].x;
                float cosOuter = uniforms.lightParams[i].y;
                attenuation = smoothstep(cosOuter, cosInner, cosAngle);
            }
        }

        // Standard constant/linear/quadratic falloff - not applicable to a directional light,
        // which has no distance to the (infinitely far away) source.
        if (lightType != LIGHT_TYPE_DIRECTIONAL) {
            attenuation *= 1.0 / (1.0 + 0.09 * lightDist + 0.032 * lightDist * lightDist);
        }

        float3 halfVector = normalize(lightDir + viewDir);
        float3 radiance = uniforms.lightColors[i].rgb * uniforms.lightColors[i].a * attenuation;

        float diffuse = max(dot(normal, lightDir), 0.0);
        float specular = powr(max(dot(normal, halfVector), 0.0), shininess) * specularStrength;

        // Point lights use the cube shadow maps (no single frustum covers all directions);
        // Directional/Spot use a single projected shadow map. Area lights don't cast shadows yet
        // (see Main.cpp's shadow pass) and read as always-lit.
        float shadow = 1.0;
        if (lightType == LIGHT_TYPE_POINT) {
            shadow = sampleCubeShadow(in.worldPosition, lightPos, shadowCubes[i], shadowSampler);
        } else if (lightType == LIGHT_TYPE_SPOT) {
            shadow = sampleProjectedShadow(in.worldPosition, lightPos, uniforms.lightViewProj[i], shadow2DMaps[i], shadowSampler);
        } else if (lightType == LIGHT_TYPE_DIRECTIONAL) {
            float3 shadowEye = lightPos - emitDir * DIRECTIONAL_SHADOW_DISTANCE;
            shadow = sampleProjectedShadow(in.worldPosition, shadowEye, uniforms.lightViewProj[i], shadow2DMaps[i], shadowSampler);
        }

        litColor += shadow * (texColor.rgb * diffuse + specular) * radiance;
    }

    // litColor is unbounded linear HDR radiance (bright/overlapping lights can push it well past
    // 1.0) - exposure scales it, then the selected curve compresses it into [0, 1], softening the
    // highlight rolloff instead of the harsh clipping a plain saturate() would give. The display
    // expects gamma-encoded (sRGB) values, so that's applied last, after tone mapping.
    float3 exposed = litColor * uniforms.renderParams.x;
    float3 toneMapped = toneMap(exposed, int(uniforms.renderParams.y));
    float3 gammaEncoded = pow(toneMapped, 1.0 / 2.2);
    return float4(gammaEncoded, texColor.a);
}

// --- Axis gizmo: flat-colored lines, no lighting/texturing ---

struct AxisVertexInput {
    float3 position [[attribute(0)]];
    float3 color    [[attribute(1)]];
};

struct AxisRasterData {
    float4 position [[position]];
    float3 color;
};

vertex AxisRasterData axisVertexMain(AxisVertexInput in [[stage_in]],
                                     constant Uniforms& uniforms [[buffer(1)]]) {
    AxisRasterData out;
    out.position = uniforms.mvpMatrix * float4(in.position, 1.0);
    out.color = in.color;
    return out;
}

fragment float4 axisFragmentMain(AxisRasterData in [[stage_in]]) {
    return float4(in.color, 1.0);
}
