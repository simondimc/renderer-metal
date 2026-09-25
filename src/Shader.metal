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

// Must match nearPlane/farPlane in computeUniforms (Uniforms.cpp) - used by postProcessFragmentMain
// to linearize the raw depth buffer for Depth of Field (see its comment).
#define CAMERA_NEAR_PLANE 0.1
#define CAMERA_FAR_PLANE 100.0

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
    // 1.0) - written straight into the offscreen HDR color target, untouched. Exposure, tone
    // mapping, and gamma encoding all happen once, screen-space, in postProcessFragmentMain below,
    // rather than per-object here - see Main.cpp's HDR scene / post-process / overlay pass split.
    return float4(litColor, texColor.a);
}

// --- Selection mask: renders just the Scene Editor's currently-selected object as flat white
// (see Main.cpp's mask pass) into a small single-channel buffer, depth-tested against the scene
// so it's correctly occluded by anything in front of it. Reuses vertexMain unchanged - the
// outline only needs mvpMatrix, none of the lighting data.

fragment float selectionMaskFragmentMain(RasterData in [[stage_in]]) {
    return 1.0;
}

// --- Post-process: full-screen pass that resolves the offscreen HDR buffer to the display ---

struct PostProcessVertexOut {
    float4 position [[position]];
    float2 uv;
};

// No vertex buffer: a single oversized triangle covering the whole screen, built purely from
// vertex_id (0, 1, 2). Standard trick to avoid a full-screen quad's extra vertices/index buffer -
// the part outside the viewport is simply clipped by the rasterizer.
vertex PostProcessVertexOut postProcessVertexMain(uint vertexID [[vertex_id]]) {
    PostProcessVertexOut out;
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    out.uv = uv;
    // Flip Y in the position (not the uv) so uv=(0,0) lands at the top-left of both the screen and
    // the source texture (Metal's texture-sample and viewport-row conventions already agree on
    // that), matching how the HDR scene pass itself was rasterized.
    out.position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return out;
}

// --- Bloom: bright-pass extract + separable Gaussian blur, both run at half the main HDR
// resolution (see Main.cpp's bloomTextureA/B) before postProcessFragmentMain samples the result
// back in. All three passes reuse postProcessVertexMain's full-screen triangle.

// Hard-threshold extract: only the linear radiance above the threshold survives (and only that
// excess amount - the sub-threshold base stays out of the bloom buffer entirely), so bloom reads
// as light spilling from genuinely bright regions rather than brightening the whole image.
fragment float4 bloomExtractFragmentMain(PostProcessVertexOut in [[stage_in]],
                                         texture2d<float> hdrTexture [[texture(0)]],
                                         constant float& threshold [[buffer(0)]],
                                         sampler smp [[sampler(0)]]) {
    float3 color = hdrTexture.sample(smp, in.uv).rgb;
    float3 bright = max(color - threshold, 0.0);
    return float4(bright, 1.0);
}

// Separable 9-tap Gaussian blur (5 unique weights, symmetric) - run once with a horizontal
// direction and once with a vertical direction (see Main.cpp) to approximate a full 2D blur at a
// fraction of the cost of a single-pass 2D kernel.
fragment float4 blurFragmentMain(PostProcessVertexOut in [[stage_in]],
                                 texture2d<float> tex [[texture(0)]],
                                 constant float2& direction [[buffer(0)]],
                                 sampler smp [[sampler(0)]]) {
    float2 texelSize = 1.0 / float2(tex.get_width(), tex.get_height());
    float2 step = direction * texelSize;

    constexpr float weights[5] = {0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216};
    float3 result = tex.sample(smp, in.uv).rgb * weights[0];
    for (int i = 1; i < 5; i++) {
        result += tex.sample(smp, in.uv + step * float(i)).rgb * weights[i];
        result += tex.sample(smp, in.uv - step * float(i)).rgb * weights[i];
    }
    return float4(result, 1.0);
}

struct PostProcessParams {
    // previousFrameViewProj * inverse(currentFrameViewProj) - reprojects a current-frame clip-space
    // position straight into where it was on screen last frame, for motion blur below. Kept first
    // in the struct (float4x4 wants 16-byte alignment) so its layout can't drift out of sync with
    // Main.cpp's mirrored C++ struct through mismatched padding around it - see its comment.
    float4x4 reprojectionMatrix;
    float exposure;
    float toneMapOperator;            // cast to int - see ToneMapOperator in Scene.hpp
    float vignetteStrength;           // 0 = off
    float chromaticAberrationStrength; // 0 = off
    float filmGrainStrength;          // 0 = off
    float sharpenStrength;            // 0 = off
    float colorGradingSaturation;     // 1 = neutral
    float colorGradingContrast;       // 1 = neutral
    float bloomIntensity;             // 0 = off - blend amount of the blurred bright-pass buffer
    float dofFocusDistance;           // world-space distance from the camera that stays sharp
    float dofFocusRange;              // distance either side of dofFocusDistance that stays sharp
    float dofStrength;                // 0 = off - max blend-in amount of the out-of-focus blur
    float motionBlurStrength;         // 0 = off - how far (in UV space) the smear reaches
    float lensFlareStrength;          // 0 = off - see LensFlareLight/lensFlareLights below
    float time;                       // seconds - animates the film grain so it doesn't look static
};

// One active light's screen-space data for the lens flare pass (see Main.cpp, which projects
// SceneLight::position through the current camera view-projection to fill this each frame). Plain
// floats throughout, like PostProcessParams, so its layout can't drift from the mirrored C++
// struct through a vector type's alignment padding.
struct LensFlareLight {
    float screenX;
    float screenY;
    float ndcDepth; // this light's own projected depth (Metal's [0,1]) - for the occlusion test
    float active;   // >0.5 = on-screen and in front of the camera; 0 = ignore this slot
    float colorR;
    float colorG;
    float colorB;
    float _pad;
};

// Base HDR scene color plus bloom, at one UV - used everywhere below that averages several taps of
// the scene (Depth of Field's disk blur, Motion Blur's directional smear), so their multi-tap
// average doesn't dilute bloom down to 1/sampleCount of its set intensity by mixing in taps that
// skip it (which is what made bloom look like it vanished whenever motion blur was active).
static float3 sampleSceneColor(float2 uv, texture2d<float> hdrTexture, texture2d<float> bloomTexture,
                               sampler smp, constant PostProcessParams& params) {
    float3 color = hdrTexture.sample(smp, uv).rgb;
    if (params.bloomIntensity > 0.0) {
        color += bloomTexture.sample(smp, uv).rgb * params.bloomIntensity;
    }
    return color;
}

// Exposure -> tone map -> saturation/contrast grading -> gamma encode, applied to one HDR sample.
// Pulled out of postProcessFragmentMain so the sharpen pass below can run it on each of its
// neighborhood taps too, comparing everything in the same display-referred space the final image
// is actually shown in (rather than sharpening raw, unbounded HDR values).
static float3 resolveColor(float3 hdrColor, constant PostProcessParams& params) {
    float3 exposed = hdrColor * params.exposure;
    float3 toneMapped = toneMap(exposed, int(params.toneMapOperator));

    // Saturation: blend toward the sample's own luminance (Rec.709 weights). Contrast: push away
    // from/toward mid-gray. Both are simple, classic display-space grading controls, not a full
    // LUT-based grade.
    float luminance = dot(toneMapped, float3(0.2126, 0.7152, 0.0722));
    float3 saturated = mix(float3(luminance), toneMapped, params.colorGradingSaturation);
    float3 graded = saturate((saturated - 0.5) * params.colorGradingContrast + 0.5);

    return pow(graded, 1.0 / 2.2);
}

fragment float4 postProcessFragmentMain(PostProcessVertexOut in [[stage_in]],
                                        texture2d<float> hdrTexture [[texture(0)]],
                                        texture2d<float> bloomTexture [[texture(1)]],
                                        depth2d<float> depthTexture [[texture(2)]],
                                        texture2d<float> selectionMaskTexture [[texture(3)]],
                                        constant PostProcessParams& params [[buffer(0)]],
                                        constant LensFlareLight* lensFlareLights [[buffer(1)]],
                                        sampler smp [[sampler(0)]],
                                        sampler depthSampler [[sampler(1)]]) {
    float2 uv = in.uv;
    float2 texelSize = 1.0 / float2(hdrTexture.get_width(), hdrTexture.get_height());
    float2 centerOffset = uv - 0.5; // screen-center-relative, for vignette/aberration falloff

    // Chromatic aberration: sample R/G/B at UVs offset outward from center, growing with distance
    // from center - like a real lens, the fringe is worst at the edges and ~0 in the middle.
    float2 aberrationOffset = centerOffset * params.chromaticAberrationStrength * 0.02;
    float r = hdrTexture.sample(smp, uv - aberrationOffset).r;
    float g = hdrTexture.sample(smp, uv).g;
    float b = hdrTexture.sample(smp, uv + aberrationOffset).b;
    float3 hdrColor = float3(r, g, b);

    // Bloom is added to the linear HDR color before tone mapping (not blended in afterward), so
    // the same curve that compresses the rest of the image also softens the bloom's own highlights
    // - bloomTexture is already the blurred bright-pass buffer (see Main.cpp's 3-pass bloom chain),
    // sampled here with bilinear filtering to upscale it back from half resolution. Guarded (not
    // just multiplied by 0) because Main.cpp skips the whole bloom chain when intensity is 0, so
    // bloomTexture can hold stale/uninitialized data then - sampling it unconditionally could pull
    // in a NaN that survives being multiplied by 0.
    if (params.bloomIntensity > 0.0) {
        hdrColor += bloomTexture.sample(smp, uv).rgb * params.bloomIntensity;
    }

    // Depth of Field: linearize the raw hardware depth (see the CAMERA_NEAR_PLANE/FAR_PLANE
    // comment - this reverses computeUniforms' projection matrix) into a view-space distance, then
    // blend toward a small blurred average as that distance moves away from dofFocusDistance.
    // depthSampler is nearest + clamp (see Main.cpp) - linearly filtering raw depth would blend
    // foreground/background distances at silhouette edges into a meaningless value, same reasoning
    // as the shadow maps' sampler.
    if (params.dofStrength > 0.0) {
        float rawDepth = depthTexture.sample(depthSampler, uv);
        float linearDepth = (CAMERA_FAR_PLANE * CAMERA_NEAR_PLANE)
                           / (CAMERA_FAR_PLANE - rawDepth * (CAMERA_FAR_PLANE - CAMERA_NEAR_PLANE));
        float coc = saturate(abs(linearDepth - params.dofFocusDistance) / max(params.dofFocusRange, 1e-4))
                  * params.dofStrength;

        if (coc > 0.0) {
            constexpr float2 diskOffsets[8] = {
                float2(1.0, 0.0),       float2(0.7071, 0.7071),
                float2(0.0, 1.0),       float2(-0.7071, 0.7071),
                float2(-1.0, 0.0),      float2(-0.7071, -0.7071),
                float2(0.0, -1.0),      float2(0.7071, -0.7071),
            };
            float2 blurRadius = texelSize * coc * 8.0; // *8 so a moderate dofStrength reads as a visible blur, not a faint softening
            float3 blurSum = float3(0.0);
            for (int i = 0; i < 8; i++) {
                blurSum += sampleSceneColor(uv + diskOffsets[i] * blurRadius, hdrTexture, bloomTexture, smp, params);
            }
            hdrColor = mix(hdrColor, blurSum / 8.0, coc);
        }
    }

    // Motion Blur: reconstructs this pixel's current-frame clip position from screen UV + raw
    // depth, reprojects it with reprojectionMatrix to find where it was on screen last frame, and
    // smears the HDR color backward along that path. Camera motion only - there's no per-object
    // velocity data, so a moving object against a static camera won't blur, only camera pans/
    // rotations/moves will (see Main.cpp's reprojectionMatrix comment).
    if (params.motionBlurStrength > 0.0) {
        float rawDepth = depthTexture.sample(depthSampler, uv);
        // uv -> NDC: x flips [0,1]->[-1,1] directly; y also flips sign because uv=0 is the screen
        // top but NDC+1 is also "up" - matching postProcessVertexMain's own uv/position mapping.
        float3 ndc = float3(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, rawDepth);
        float4 previousClip = params.reprojectionMatrix * float4(ndc, 1.0);
        if (previousClip.w > 0.0) {
            float2 previousNDC = previousClip.xy / previousClip.w;
            float2 previousUV = previousNDC * float2(0.5, -0.5) + float2(0.5, 0.5);
            // Strength is already baked into reprojectionMatrix (as shutter time, see Main.cpp), so
            // this is the actual smear to apply - not scaled again here.
            float2 motionVector = uv - previousUV;

            // Under half a texel of smear is invisible - skipping the 7 extra (2-sample) taps there
            // makes motion blur free while the camera is static, instead of always paying full cost.
            float smearTexels = length(motionVector / texelSize);
            if (smearTexels > 0.5) {
                // A fixed 8 taps stretched over a long smear (high strength / fast camera) lands
                // them tens of pixels apart, which reads as discrete "ghost" copies of the frame -
                // stepped and laggy-looking, not a smooth blur. So: cap the smear length (a real
                // shutter never blurs across many frames of travel), scale the tap count with the
                // remaining length (~1 tap per 6px, 4..12), and jitter each pixel's tap positions
                // with interleaved-gradient noise so any leftover banding turns into fine grain.
                constexpr float maxSmearTexels = 48.0;
                if (smearTexels > maxSmearTexels) {
                    motionVector *= maxSmearTexels / smearTexels;
                    smearTexels = maxSmearTexels;
                }
                int sampleCount = clamp(int(smearTexels / 6.0) + 2, 4, 12);
                float jitter = fract(52.9829189 * fract(dot(in.position.xy, float2(0.06711056, 0.00583715))));

                float3 blurSum = hdrColor;
                for (int i = 1; i < sampleCount; i++) {
                    float t = (float(i) + jitter - 0.5) / float(sampleCount - 1);
                    float2 sampleUV = uv - motionVector * t;
                    blurSum += sampleSceneColor(sampleUV, hdrTexture, bloomTexture, smp, params);
                }
                hdrColor = blurSum / float(sampleCount);
            }
        }
    }

    // Lens Flare: added last (after Depth of Field/Motion Blur, not before) so their multi-tap
    // averaging can't dilute it the same way it used to dilute bloom - a flare is an artifact of
    // the lens/camera itself, not scene light, so it shouldn't be softened by effects that operate
    // on the scene's own geometry. For each active light (screen-projected on the CPU each frame -
    // see LensFlareLight's comment), draws a soft glow at its screen position plus a few "ghost"
    // artifacts strung along the line through the screen center - the secondary reflections a real
    // lens catches when pointed near a bright source.
    if (params.lensFlareStrength > 0.0) {
        for (int i = 0; i < MAX_LIGHTS; i++) {
            LensFlareLight light = lensFlareLights[i];
            if (light.active < 0.5) continue;

            float2 lightUV = float2(light.screenX, light.screenY);
            // Smaller depth = closer to camera (Metal's [0,1], 0 = near). Occluded (skip this
            // light entirely) if something in the depth buffer sits closer than the light itself.
            float occluderDepth = depthTexture.sample(depthSampler, lightUV);
            if (occluderDepth < light.ndcDepth - 0.001) continue;

            float3 flareColor = float3(light.colorR, light.colorG, light.colorB);

            float glowDist = length(uv - lightUV);
            float glow = exp(-glowDist * glowDist * 400.0);
            hdrColor += flareColor * glow * params.lensFlareStrength;

            float2 towardCenter = float2(0.5, 0.5) - lightUV;
            constexpr float ghostT[4] = {0.3, 0.6, 1.0, 1.4}; // position along the light->center line
            for (int g = 0; g < 4; g++) {
                float2 ghostUV = lightUV + towardCenter * ghostT[g];
                float ghostDist = length(uv - ghostUV);
                float ghost = exp(-ghostDist * ghostDist * 2000.0);
                hdrColor += flareColor * ghost * params.lensFlareStrength * 0.3;
            }
        }
    }

    float3 color = resolveColor(hdrColor, params);

    // Sharpen: unsharp mask - push the pixel away from a cheap 4-tap neighborhood average. Skipped
    // entirely (rather than just multiplied by 0) when off, to avoid the 4 extra resolveColor
    // evaluations on the common case - this is a uniform (non-per-pixel) branch, so it's a real
    // cost saving, not just dead math.
    if (params.sharpenStrength > 0.0) {
        float3 up    = resolveColor(hdrTexture.sample(smp, uv + float2(0.0, -texelSize.y)).rgb, params);
        float3 down  = resolveColor(hdrTexture.sample(smp, uv + float2(0.0, texelSize.y)).rgb, params);
        float3 left  = resolveColor(hdrTexture.sample(smp, uv + float2(-texelSize.x, 0.0)).rgb, params);
        float3 right = resolveColor(hdrTexture.sample(smp, uv + float2(texelSize.x, 0.0)).rgb, params);
        float3 blurred = (up + down + left + right) * 0.25;
        color += (color - blurred) * params.sharpenStrength;
    }

    // Vignette: darken toward the screen edges. Applied last (alongside grain) in display space,
    // as a straightforward multiplicative falloff rather than something the tone curve should see.
    float vignette = saturate(1.0 - params.vignetteStrength * dot(centerOffset, centerOffset) * 2.0);
    color *= vignette;

    // Film grain: cheap hash noise from screen position + time, so it flickers frame-to-frame
    // instead of reading as a fixed print/overlay pattern.
    float2 texSize = 1.0 / texelSize;
    float noise = fract(sin(dot(uv * texSize + params.time, float2(12.9898, 78.233))) * 43758.5453);
    color += (noise - 0.5) * params.filmGrainStrength;

    // Scene Editor selection outline: edge-detects selectionMaskTexture (a binary silhouette of
    // whatever object is currently selected - see Main.cpp's mask pass, which clears it to 0 every
    // frame and only draws into it while in edit mode) and draws a bright ring wherever the mask
    // transitions between "inside" and "outside". Nearest sampling (depthSampler, reused) since the
    // mask is a hard 0/1 edge - bilinear filtering would blur it into a soft false gradient instead
    // of a clean boundary. Naturally a no-op when nothing is selected: an all-zero mask has no
    // transitions, so no explicit on/off flag is needed here.
    float2 outlineTexel = texelSize * 2.0; // ~2px thick line
    float maskCenter = selectionMaskTexture.sample(depthSampler, uv).r;
    float maskUp    = selectionMaskTexture.sample(depthSampler, uv + float2(0.0, -outlineTexel.y)).r;
    float maskDown  = selectionMaskTexture.sample(depthSampler, uv + float2(0.0, outlineTexel.y)).r;
    float maskLeft  = selectionMaskTexture.sample(depthSampler, uv + float2(-outlineTexel.x, 0.0)).r;
    float maskRight = selectionMaskTexture.sample(depthSampler, uv + float2(outlineTexel.x, 0.0)).r;
    float maskEdge = max(max(abs(maskCenter - maskUp), abs(maskCenter - maskDown)),
                          max(abs(maskCenter - maskLeft), abs(maskCenter - maskRight)));
    color = mix(color, float3(1.0, 0.6, 0.1), saturate(maskEdge));

    return float4(saturate(color), 1.0);
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
