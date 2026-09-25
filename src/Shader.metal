#include <metal_stdlib>
using namespace metal;

#define MAX_LIGHTS 4 // must match kMaxLights in Uniforms.hpp

// Must match the LightType enum in Scene.hpp
#define LIGHT_TYPE_POINT 0
#define LIGHT_TYPE_DIRECTIONAL 1
#define LIGHT_TYPE_SPOT 2
#define LIGHT_TYPE_AREA 3

// Must match the AlphaMode enum in Uniforms.hpp
#define ALPHA_MODE_OPAQUE 0
#define ALPHA_MODE_MASK 1
#define ALPHA_MODE_BLEND 2

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
    float4 tangent  [[attribute(3)]]; // xyz = tangent, w = bitangent handedness (+1/-1)
};

struct RasterData {
    float4 position [[position]];
    float2 uv;
    float3 worldPosition;
    float3 worldNormal;
    float4 worldTangent; // xyz = world-space tangent, w = handedness
};

struct Uniforms {
    float4x4 mvpMatrix;
    float4x4 modelMatrix;
    float4x4 lightViewProj[MAX_LIGHTS]; // Directional/Spot shadow-map view-projection (other light data: GPULight buffer)
    float4 cameraPosition;
    float4 materialAlbedo;              // rgb = albedo tint
    float4 materialParams;              // x = metallic, y = roughness, z = ao, w = useTextures (0/1)
    float4x4 viewProjMatrix;            // camera only, no model - projects refracted points for transmission
};

// Per-draw glTF material factors, bound at fragment buffer 2 - must match MaterialParams in
// Uniforms.hpp. Neutral (all 1, occlusion strength 0) for anything that isn't a glTF submesh.
struct MaterialParams {
    float4 baseColorFactor;
    float4 factors;        // x = metallic, y = roughness, z = occlusion strength
    float4 emissiveFactor; // rgb = emissive color * strength (HDR)
    float4 alphaParams;    // x = alpha mode (ALPHA_MODE_*), y = Mask cutoff
    float4 transmissionParams; // x = transmission, y = thickness (mesh units), z = attenuation distance, w = IOR
    float4 attenuationColor;   // rgb = volume absorption color
};

// --- Clustered forward lighting (see LightCulling.hpp). The lights live in one buffer shared by every draw
// (directional lights first, then the rest), and lightCullKernel sorts the non-directional ones into the
// camera's cluster grid; fragmentMain then shades only its own cluster's list.

// Must match GPULight in LightCulling.hpp.
struct GPULight {
    float4 positionRange;  // xyz = world position (Point/Spot/Area), w = cull radius (its reach)
    float4 directionType;  // xyz = normalized emission direction, w = LightType
    float4 colorIntensity; // rgb = color, a = intensity
    float4 params;         // x=spotCosInner, y=spotCosOuter, z=areaHalfWidth, w=areaHalfHeight
    float4 right;          // xyz = area light local right axis, w = shadow-map slot (-1 = none)
    float4 up;             // xyz = area light local up axis, w = attenuation at the cull radius
};

// Must match ClusterParams in LightCulling.hpp.
struct ClusterParams {
    float4x4 view;
    float4 projection; // x = proj[0][0], y = proj[1][1] (un-jittered), z = near, w = far
    uint4 grid;        // x, y = tile counts, z = depth slices, w = tile size in pixels
    uint4 lights;      // x = directional lights, y = clustered lights (after the directional ones)
    float4 screen;     // x, y = width, height in pixels, z = slice scale, w = slice bias
};

#define MAX_LIGHTS_PER_CLUSTER 64 // must match kMaxLightsPerCluster in LightCulling.hpp

// Depth slices are exponential: slice = floor(log(d) * scale + bias) for a view-space distance d in front of
// the camera, so slice k spans near * (far/near)^(k/N) to near * (far/near)^((k+1)/N).
static uint clusterSliceOf(float viewDepth, constant ClusterParams& cluster) {
    float slice = log(max(viewDepth, 1e-4)) * cluster.screen.z + cluster.screen.w;
    return uint(clamp(slice, 0.0, float(cluster.grid.z - 1)));
}

// One thread per cluster: builds its view-space bounding box from the tile's four corners at the slice's near
// and far depth, and keeps every clustered light whose reach (a sphere) touches that box.
kernel void lightCullKernel(constant ClusterParams& cluster [[buffer(0)]],
                            constant GPULight* lights [[buffer(1)]],
                            device uint* clusterCounts [[buffer(2)]],
                            device ushort* clusterIndices [[buffer(3)]],
                            uint3 gid [[thread_position_in_grid]]) {
    if (any(gid >= cluster.grid.xyz)) return;

    float tilePixels = float(cluster.grid.w);
    float2 screenSize = cluster.screen.xy;
    float2 tileMin = float2(gid.xy) * tilePixels;
    float2 tileMax = min(tileMin + tilePixels, screenSize);
    // Pixel -> NDC (pixel y grows downward, NDC y upward).
    float2 ndcMin = float2(tileMin.x / screenSize.x * 2.0 - 1.0, 1.0 - tileMax.y / screenSize.y * 2.0);
    float2 ndcMax = float2(tileMax.x / screenSize.x * 2.0 - 1.0, 1.0 - tileMin.y / screenSize.y * 2.0);

    float nearPlane = cluster.projection.z;
    float farPlane = cluster.projection.w;
    float sliceCount = float(cluster.grid.z);
    float depthNear = nearPlane * pow(farPlane / nearPlane, float(gid.z) / sliceCount);
    float depthFar = nearPlane * pow(farPlane / nearPlane, float(gid.z + 1) / sliceCount);

    // A point at NDC (x, y) and distance d ahead of the camera is (x*d/P00, y*d/P11, -d) in view space.
    float3 boxMin = float3(INFINITY), boxMax = float3(-INFINITY);
    for (uint corner = 0; corner < 8; corner++) {
        float2 ndc = float2((corner & 1) ? ndcMax.x : ndcMin.x, (corner & 2) ? ndcMax.y : ndcMin.y);
        float depth = (corner & 4) ? depthFar : depthNear;
        float3 point = float3(ndc.x * depth / cluster.projection.x, ndc.y * depth / cluster.projection.y, -depth);
        boxMin = min(boxMin, point);
        boxMax = max(boxMax, point);
    }

    uint clusterIndex = (gid.z * cluster.grid.y + gid.y) * cluster.grid.x + gid.x;
    uint count = 0;
    uint first = cluster.lights.x;
    for (uint i = first; i < first + cluster.lights.y && count < MAX_LIGHTS_PER_CLUSTER; i++) {
        float3 center = (cluster.view * float4(lights[i].positionRange.xyz, 1.0)).xyz;
        float radius = lights[i].positionRange.w;
        float3 nearest = clamp(center, boxMin, boxMax); // the box's point closest to the light
        if (distance_squared(center, nearest) <= radius * radius) {
            clusterIndices[clusterIndex * MAX_LIGHTS_PER_CLUSTER + count] = ushort(i);
            count++;
        }
    }
    clusterCounts[clusterIndex] = count;
}

// Vertex Shader
vertex RasterData vertexMain(VertexInput in [[stage_in]],
                             constant Uniforms& uniforms [[buffer(1)]]) {
    RasterData out;
    out.position = uniforms.mvpMatrix * float4(in.position, 1.0);
    out.uv = in.uv;
    out.worldPosition = (uniforms.modelMatrix * float4(in.position, 1.0)).xyz;
    out.worldNormal = (uniforms.modelMatrix * float4(in.normal, 0.0)).xyz;
    out.worldTangent = float4((uniforms.modelMatrix * float4(in.tangent.xyz, 0.0)).xyz, in.tangent.w);
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
    float2 uv; // only for the Mask cutout test in cubeShadowFragmentMain
};

vertex CubeShadowRasterData cubeShadowVertexMain(VertexInput in [[stage_in]],
                                                 constant Uniforms& uniforms [[buffer(1)]],
                                                 constant float4x4& lightViewProj [[buffer(2)]]) {
    CubeShadowRasterData out;
    float4 worldPos = uniforms.modelMatrix * float4(in.position, 1.0);
    out.position = lightViewProj * worldPos;
    out.worldPosition = worldPos.xyz;
    out.uv = in.uv;
    return out;
}

// Mask materials cast a cut-out shadow (a leaf texture's transparent corners let light through), so
// the shadow pass needs the material's alpha too. Opaque draws pay nothing but the mode check, and
// Blend draws are skipped by the CPU side altogether (see Main.cpp) - glass doesn't cast a shadow.
fragment float cubeShadowFragmentMain(CubeShadowRasterData in [[stage_in]],
                                      constant float3& lightPosition [[buffer(3)]],
                                      constant MaterialParams& material [[buffer(4)]],
                                      texture2d<float> albedoMap [[texture(0)]]) {
    if (int(material.alphaParams.x + 0.5) == ALPHA_MODE_MASK) {
        constexpr sampler alphaSampler(filter::linear, mip_filter::linear, address::repeat);
        float alpha = albedoMap.sample(alphaSampler, in.uv).a * material.baseColorFactor.a;
        if (alpha < material.alphaParams.y) discard_fragment();
    }
    return length(in.worldPosition - lightPosition);
}

// Percentage-closer filtering: a single nearest-sampled lookup gives a hard, stair-stepped
// shadow edge (one shadow-map texel = one visible jag), and the distance-encoded maps can't use
// the hardware's bilinear compare. So each shadow lookup takes PCF_SAMPLES taps spread over a
// disk, compares every tap on its own, and averages the 0/1 results - which blurs the edge into a
// smooth penumbra. The disk is rotated per pixel (interleaved gradient noise) so the fixed tap
// pattern turns into fine, unobtrusive noise instead of visible banding.
#define PCF_SAMPLES 16
#define PCF_RADIUS_TEXELS 2.0
// Only the first PCF_PROBE_SAMPLES taps are read up front: if they all agree (all lit or all shadowed) the
// pixel is well inside a lit or a shadowed region and the other taps could only repeat that answer, so
// they are skipped. Only pixels near a shadow edge, where the probe disagrees, pay for the full 16.
// The first four Poisson-disk taps are spread around the disk, so the probe sees an edge from any side.
#define PCF_PROBE_SAMPLES 4

constant float2 kPoissonDisk[PCF_SAMPLES] = {
    float2(-0.94201624, -0.39906216), float2( 0.94558609, -0.76890725),
    float2(-0.09418410, -0.92938870), float2( 0.34495938,  0.29387760),
    float2(-0.91588581,  0.45771432), float2(-0.81544232, -0.87912464),
    float2(-0.38277543,  0.27676845), float2( 0.97484398,  0.75648379),
    float2( 0.44323325, -0.97511554), float2( 0.53742981, -0.47373420),
    float2(-0.26496911, -0.41893023), float2( 0.79197514,  0.19090188),
    float2(-0.24188840,  0.99706507), float2(-0.81409955,  0.91437590),
    float2( 0.19984126,  0.78641367), float2( 0.14383161, -0.14100790)
};

// The per-pixel rotation of the tap pattern (interleaved gradient noise). One per pixel - shared by every
// light - so fragmentMain computes it once and hands it to the sampling functions.
static float2x2 pcfRotation(float2 pixel) {
    float noise = fract(52.9829189 * fract(dot(pixel, float2(0.06711056, 0.00583715))));
    float angle = noise * 6.2831853;
    float c = cos(angle), s = sin(angle);
    return float2x2(float2(c, s), float2(-s, c));
}

// 1.0 = fully lit, 0.0 = fully in shadow. Sampling by direction (rather than a light-space
// projected UV) means every direction around the light is valid - no "outside the frustum" case
// to special-case, unlike a single-frustum shadow map.
static float sampleCubeShadow(float3 worldPosition, float3 lightPosition,
                              texturecube<float> shadowCube, sampler shadowSampler,
                              float bias, float2x2 rot) {
    float3 fragToLight = worldPosition - lightPosition;
    float currentDistance = length(fragToLight);
    float3 dir = fragToLight / max(currentDistance, 1e-4);

    // Tangent basis for offsetting the lookup direction. One texel spans 2/size of the dominant
    // axis component (90-degree face), and the dominant component of a unit direction is >= 0.577.
    float3 helper = abs(dir.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(helper, dir));
    float3 bitangent = cross(dir, tangent);
    float radius = PCF_RADIUS_TEXELS * (2.0 / float(shadowCube.get_width()));

    float lit = 0.0;
    for (int k = 0; k < PCF_PROBE_SAMPLES; k++) {
        float2 o = rot * kPoissonDisk[k] * radius;
        float closestDistance = shadowCube.sample(shadowSampler, dir + tangent * o.x + bitangent * o.y).r;
        lit += (currentDistance - bias > closestDistance) ? 0.0 : 1.0;
    }
    if (lit == 0.0 || lit == float(PCF_PROBE_SAMPLES)) return lit / float(PCF_PROBE_SAMPLES); // agreed: not near an edge
    for (int k = PCF_PROBE_SAMPLES; k < PCF_SAMPLES; k++) {
        float2 o = rot * kPoissonDisk[k] * radius;
        float closestDistance = shadowCube.sample(shadowSampler, dir + tangent * o.x + bitangent * o.y).r;
        lit += (currentDistance - bias > closestDistance) ? 0.0 : 1.0;
    }
    return lit / float(PCF_SAMPLES);
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
                                   texture2d<float> shadowMap, sampler shadowSampler,
                                   float bias, float2x2 rot) {
    float4 lightSpace = lightViewProj * float4(worldPosition, 1.0);
    if (lightSpace.w <= 0.0) return 1.0;
    float3 ndc = lightSpace.xyz / lightSpace.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) return 1.0;

    float2 uv = float2(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5); // NDC +Y up -> texture V down
    float currentDistance = length(worldPosition - shadowEye);
    float2 texel = 1.0 / float2(shadowMap.get_width(), shadowMap.get_height());

    float lit = 0.0;
    for (int k = 0; k < PCF_PROBE_SAMPLES; k++) {
        float2 o = rot * kPoissonDisk[k] * (PCF_RADIUS_TEXELS * texel);
        float closestDistance = shadowMap.sample(shadowSampler, uv + o).r;
        lit += (currentDistance - bias > closestDistance) ? 0.0 : 1.0;
    }
    if (lit == 0.0 || lit == float(PCF_PROBE_SAMPLES)) return lit / float(PCF_PROBE_SAMPLES); // agreed: not near an edge
    for (int k = PCF_PROBE_SAMPLES; k < PCF_SAMPLES; k++) {
        float2 o = rot * kPoissonDisk[k] * (PCF_RADIUS_TEXELS * texel);
        float closestDistance = shadowMap.sample(shadowSampler, uv + o).r;
        lit += (currentDistance - bias > closestDistance) ? 0.0 : 1.0;
    }
    return lit / float(PCF_SAMPLES);
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

// --- Cook-Torrance microfacet BRDF (GGX / Smith / Schlick), the standard metallic-roughness model.
// alpha = roughness^2 is the perceptually-linear remapping used by all three terms below.

constant float PI = 3.14159265358979;

// GGX / Trowbridge-Reitz normal distribution: how many microfacets point along the half vector.
static float distributionGGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Schlick-GGX geometry term for one direction (view or light), with the direct-lighting remap of k.
static float geometrySchlickGGX(float NdotX, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotX / (NdotX * (1.0 - k) + k);
}

// Smith's method: shadowing/masking is the product of the light-side and view-side terms.
static float geometrySmith(float NdotV, float NdotL, float roughness) {
    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

static float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * powr(saturate(1.0 - cosTheta), 5.0);
}

// Schlick's Fresnel for ambient (image-based) light: there's no single half vector to measure the
// angle against, so the view angle stands in for it, and rough surfaces are kept from reflecting
// more at grazing angles than their F0 lets a rough microsurface actually do.
static float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness) {
    return F0 + (max(float3(1.0 - roughness), F0) - F0) * powr(saturate(1.0 - cosTheta), 5.0);
}

// The scene pass writes three targets: the lit HDR color, plus a small G-buffer the screen-space
// reflection pass (see ssrTraceFragmentMain) reads back - the shading normal with roughness, and how much
// of the image-based specular light this pixel's reflection is worth (see the write at the end of
// fragmentMain). Sky pixels and anything never drawn keep the cleared zeros, which SSR skips.
struct SceneFragmentOut {
    float4 color           [[color(0)]];
    float4 normalRoughness [[color(1)]]; // xyz = world-space shading normal, w = roughness
    float4 specularWeight  [[color(2)]]; // rgb = multiplier on the environment's specular radiance
};

// Fragment Shader
fragment SceneFragmentOut fragmentMain(RasterData in [[stage_in]],
                             constant Uniforms& uniforms [[buffer(1)]],
                             texture2d<float> tex [[texture(0)]],
                             texture2d<float> normalMap [[texture(1)]],
                             array<texturecube<float>, MAX_LIGHTS> shadowCubes [[texture(2)]],
                             array<texture2d<float>, MAX_LIGHTS> shadow2DMaps [[texture(2 + MAX_LIGHTS)]],
                             texture2d<float> ormMap [[texture(2 + 2 * MAX_LIGHTS)]],
                             texturecube<float> irradianceMap [[texture(3 + 2 * MAX_LIGHTS)]],
                             texturecube<float> prefilterMap [[texture(4 + 2 * MAX_LIGHTS)]],
                             texture2d<float> brdfLUT [[texture(5 + 2 * MAX_LIGHTS)]],
                             texture2d<float> occlusionMap [[texture(6 + 2 * MAX_LIGHTS)]],
                             texture2d<float> emissiveMap [[texture(7 + 2 * MAX_LIGHTS)]],
                             texture2d<float> transmissionMap [[texture(8 + 2 * MAX_LIGHTS)]],
                             texture2d<float> thicknessMap [[texture(9 + 2 * MAX_LIGHTS)]],
                             texture2d<float> transmissionSource [[texture(10 + 2 * MAX_LIGHTS)]],
                             texture2d<float> screenAOMap [[texture(11 + 2 * MAX_LIGHTS)]],
                             constant MaterialParams& material [[buffer(2)]],
                             constant float& environmentIntensity [[buffer(3)]],
                             constant ClusterParams& cluster [[buffer(4)]],
                             constant GPULight* lights [[buffer(5)]],
                             constant uint* clusterCounts [[buffer(6)]],
                             constant ushort* clusterIndices [[buffer(7)]],
                             sampler smp [[sampler(0)]],
                             sampler shadowSampler [[sampler(1)]],
                             sampler envSampler [[sampler(2)]],
                             sampler screenSampler [[sampler(3)]]) {
    bool useTextures = uniforms.materialParams.w > 0.5;

    float3 N = normalize(in.worldNormal);
    float3 normal = N;
    if (useTextures) {
        // Build the TBN basis and use it to rotate the tangent-space normal map sample into world space
        float3 T = normalize(in.worldTangent.xyz);
        // w = -1 on mirrored UVs flips the bitangent (glTF: B = cross(N, T) * w); sign() so
        // interpolating between vertices can't produce a fractional handedness.
        float3 B = cross(N, T) * (in.worldTangent.w < 0.0 ? -1.0 : 1.0);
        float3x3 TBN = float3x3(T, B, N);
        float3 tangentNormal = normalMap.sample(smp, in.uv).rgb * 2.0 - 1.0;
        normal = normalize(TBN * tangentNormal);
    }

    float3 viewDir = normalize(uniforms.cameraPosition.xyz - in.worldPosition);
    // Scene Editor material (uniforms.material*) x the draw's glTF factors and textures (see
    // Material in Scene.hpp). ORM = glTF packing: G = roughness, B = metallic; occlusion (R of its
    // own map - the same image as ORM in the usual packed layout, but glTF allows a separate one).
    float3 albedo = uniforms.materialAlbedo.rgb;
    float metallic = uniforms.materialParams.x;
    float roughness = uniforms.materialParams.y;
    float ao = uniforms.materialParams.z;
    float alpha = material.baseColorFactor.a;
    if (useTextures) {
        float4 baseColor = tex.sample(smp, in.uv);
        albedo *= baseColor.rgb * material.baseColorFactor.rgb;
        alpha *= baseColor.a;
        float3 orm = ormMap.sample(smp, in.uv).rgb;
        metallic *= material.factors.x * orm.b;
        roughness *= material.factors.y * orm.g;
        ao *= mix(1.0, occlusionMap.sample(smp, in.uv).r, material.factors.z);
    }
    int alphaMode = int(material.alphaParams.x + 0.5);
    if (alphaMode == ALPHA_MODE_MASK && alpha < material.alphaParams.y) discard_fragment();
    metallic = saturate(metallic);
    // A perfectly smooth surface makes the GGX lobe infinitely narrow (a point light would vanish
    // or blow out to a single pixel), so keep a small floor.
    roughness = clamp(roughness, 0.04, 1.0);

    // Reflectance at normal incidence: from the index of refraction for dielectrics (1.5 -> ~4%),
    // the albedo itself (tinted) for metals.
    float ior = material.transmissionParams.w;
    float dielectricF0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
    float3 F0 = mix(float3(dielectricF0), albedo, metallic);
    float NdotV = max(dot(normal, viewDir), 1e-4);

    // Image-based lighting (split-sum approximation): the environment is precomputed once (see
    // Environment.cpp and the *Kernel functions at the bottom of this file) into a diffuse irradiance
    // cube, a specular cube prefiltered per roughness (one mip per roughness step), and a 2D BRDF
    // lookup table - so the whole ambient term costs three texture reads instead of an integral.
    float3 F_ibl = fresnelSchlickRoughness(NdotV, F0, roughness);
    float3 kD_ibl = (1.0 - F_ibl) * (1.0 - metallic); // metals have no diffuse term
    float3 irradiance = irradianceMap.sample(envSampler, normal).rgb;
    float3 diffuseIBL = irradiance * albedo;

    float3 R = reflect(-viewDir, normal);
    float lod = roughness * float(prefilterMap.get_num_mip_levels() - 1);
    float3 prefiltered = prefilterMap.sample(envSampler, R, level(lod)).rgb;
    float2 envBRDF = brdfLUT.sample(envSampler, float2(NdotV, roughness)).rg;
    float3 specularIBL = prefiltered * (F_ibl * envBRDF.x + envBRDF.y);

    // Transmission (KHR_materials_transmission/volume): a transmissive dielectric passes light
    // through instead of scattering it diffusely, so that share of the diffuse term is replaced by
    // what's behind the surface. "Behind" is transmissionSource, a copy of the opaque scene made
    // before this pass (so glass never sees other glass, only opaque objects and the sky). The view
    // ray is bent by the IOR and carried through the object's thickness, and that exit point's screen
    // position is where the copy is read - so thick glass distorts, thickness 0 (thin-walled) doesn't.
    // Roughness picks a blurrier mip (frosted glass), and the volume's absorption tints by how far
    // the light travelled inside (Beer-Lambert).
    float transmission = 0.0;
    float3 transmitted = float3(0.0);
    if (material.transmissionParams.x > 0.0) {
        transmission = material.transmissionParams.x;
        float thickness = material.transmissionParams.y;
        if (useTextures) {
            transmission *= transmissionMap.sample(smp, in.uv).r;
            thickness *= thicknessMap.sample(smp, in.uv).g;
        }
        // Thickness is in mesh units; the model matrix's scale (assumed uniform) converts to world.
        float thicknessWorld = thickness * length(uniforms.modelMatrix[0].xyz);
        float3 refracted = refract(-viewDir, normal, 1.0 / ior);
        float4 exitClip = uniforms.viewProjMatrix * float4(in.worldPosition + refracted * thicknessWorld, 1.0);
        float2 exitUV = (exitClip.xy / exitClip.w) * float2(0.5, -0.5) + 0.5;
        float lod = log2(float(transmissionSource.get_width())) * saturate(roughness * saturate(ior * 2.0 - 2.0));
        float3 background = transmissionSource.sample(screenSampler, exitUV, level(lod)).rgb;

        float attenuationDistance = material.transmissionParams.z;
        float3 attenuation = float3(1.0);
        if (attenuationDistance < 1e29) {
            float3 coefficient = -log(max(material.attenuationColor.rgb, float3(1e-4))) / attenuationDistance;
            attenuation = exp(-coefficient * thicknessWorld);
        }
        transmitted = background * albedo * attenuation;
    }

    // Screen-space ambient occlusion (see aoFragmentMain): the visibility of this pixel's hemisphere,
    // computed from nearby depth. It only dims ambient light - direct lights have their own shadow
    // maps. Where AO is off (or for glass/blend draws, whose pixels the AO buffer doesn't describe)
    // Main.cpp binds a white texture, so this reads 1 and changes nothing.
    // The diffuse term uses Jimenez et al.'s multi-bounce fit, which brightens the occlusion of light
    // surfaces (light bouncing between nearby walls); the specular term is Lagarde's specular
    // occlusion, which depends on roughness and view angle since a mirror-like lobe is only blocked
    // along its reflection direction.
    float screenAO = screenAOMap.sample(screenSampler, in.position.xy / float2(screenAOMap.get_width(), screenAOMap.get_height())).r;
    float3 aoA = 2.0404 * albedo - 0.3324;
    float3 aoB = -4.7951 * albedo + 0.6417;
    float3 aoC = 2.7552 * albedo + 0.6903;
    float3 diffuseAO = saturate(max(float3(screenAO), ((screenAO * aoA + aoB) * screenAO + aoC) * screenAO));
    float specularAO = saturate(pow(NdotV + screenAO, exp2(-16.0 * roughness - 1.0)) - 1.0 + screenAO);

    float3 litColor = (kD_ibl * diffuseIBL * (1.0 - transmission) * diffuseAO + specularIBL * specularAO) * ao * environmentIntensity;
    litColor += transmission * (1.0 - metallic) * (1.0 - F_ibl) * transmitted;

    // This pixel's cluster: the screen tile it is in, and the depth slice of its distance in front of the camera
    // (see lightCullKernel). Its list holds every non-directional light that can reach it; directional lights,
    // which reach everywhere, come first in the buffer and are not part of any list.
    float viewDepth = -(cluster.view * float4(in.worldPosition, 1.0)).z;
    uint2 tile = min(uint2(in.position.xy) / cluster.grid.w, cluster.grid.xy - 1);
    uint clusterIndex = (clusterSliceOf(viewDepth, cluster) * cluster.grid.y + tile.y) * cluster.grid.x + tile.x;
    uint directionalCount = cluster.lights.x;
    uint lightCount = directionalCount + clusterCounts[clusterIndex];

    float2x2 pcfRot = pcfRotation(in.position.xy);
    for (uint k = 0; k < lightCount; k++) {
        uint lightIndex = k < directionalCount ? k : uint(clusterIndices[clusterIndex * MAX_LIGHTS_PER_CLUSTER + (k - directionalCount)]);
        GPULight light = lights[lightIndex];
        int lightType = int(light.directionType.w);
        int shadowSlot = int(light.right.w); // -1: this light has no shadow map
        float3 lightPos = light.positionRange.xyz;
        float3 emitDir = light.directionType.xyz; // direction the light travels outward

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
            float3 right = light.right.xyz;
            float3 up = light.up.xyz;
            float2 halfSize = light.params.zw;
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
                float cosInner = light.params.x;
                float cosOuter = light.params.y;
                attenuation = smoothstep(cosOuter, cosInner, cosAngle);
            }
        }

        // Standard constant/linear/quadratic falloff - not applicable to a directional light,
        // which has no distance to the (infinitely far away) source. The falloff never reaches zero, so the
        // value it has at the light's cull radius (light.up.w) is subtracted and the rest renormalized: the
        // light then ends exactly where the culling stops considering it, and the change is below the cutoff.
        if (lightType != LIGHT_TYPE_DIRECTIONAL) {
            float falloff = 1.0 / (1.0 + 0.09 * lightDist + 0.032 * lightDist * lightDist);
            attenuation *= max(falloff - light.up.w, 0.0) / (1.0 - light.up.w);
        }

        // A surface turned away from the light, or a light that has faded to nothing, gets nothing from it: every
        // term below is scaled by NdotL and the attenuation, so skipping now (before the shadow lookups, the most
        // expensive part of the loop) changes no result.
        float NdotL = max(dot(normal, lightDir), 0.0);
        if (NdotL <= 0.0 || attenuation <= 0.0) continue;

        float3 halfVector = normalize(lightDir + viewDir);
        // Multiplied by PI to cancel the Lambert BRDF's albedo/PI below, so light intensities keep
        // the pre-PBR meaning: intensity 1 = a white diffuse surface facing the light returns 1.0.
        float3 radiance = light.colorIntensity.rgb * light.colorIntensity.a * attenuation * PI;

        float NdotH = max(dot(normal, halfVector), 0.0);
        float VdotH = max(dot(viewDir, halfVector), 0.0);

        float D = distributionGGX(NdotH, roughness);
        float G = geometrySmith(NdotV, NdotL, roughness);
        float3 F = fresnelSchlick(VdotH, F0);
        float3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4);
        // Energy conservation: light reflected specularly (F) can't also be diffusely scattered, and
        // metals absorb whatever they don't reflect.
        float3 kD = (1.0 - F) * (1.0 - metallic);
        float3 brdf = kD * albedo / PI * (1.0 - transmission) + specular;

        // Point lights use the cube shadow maps (no single frustum covers all directions);
        // Directional/Spot use a single projected shadow map. Area lights don't cast shadows yet
        // (see Main.cpp's shadow pass) and read as always-lit - as do all lights past the first
        // kMaxLights, which have no shadow map slot.
        // The PCF disk reads neighbors at slightly different depths, so surfaces at grazing angles
        // to the light need a larger bias than face-on ones to avoid self-shadowing acne.
        float shadowBias = 0.05 + 0.1 * (1.0 - NdotL);
        float shadow = 1.0;
        if (shadowSlot >= 0) {
            if (lightType == LIGHT_TYPE_POINT) {
                shadow = sampleCubeShadow(in.worldPosition, lightPos, shadowCubes[shadowSlot], shadowSampler, shadowBias, pcfRot);
            } else if (lightType == LIGHT_TYPE_SPOT) {
                shadow = sampleProjectedShadow(in.worldPosition, lightPos, uniforms.lightViewProj[shadowSlot], shadow2DMaps[shadowSlot], shadowSampler, shadowBias, pcfRot);
            } else if (lightType == LIGHT_TYPE_DIRECTIONAL) {
                float3 shadowEye = lightPos - emitDir * DIRECTIONAL_SHADOW_DISTANCE;
                shadow = sampleProjectedShadow(in.worldPosition, shadowEye, uniforms.lightViewProj[shadowSlot], shadow2DMaps[shadowSlot], shadowSampler, shadowBias, pcfRot);
            }
        }

        litColor += shadow * brdf * radiance * NdotL;
    }

    // Emissive: light the surface itself gives off, independent of any light or the environment
    // (so not scaled by ao/environmentIntensity). Unbounded HDR, so a strong factor feeds Bloom.
    float3 emissive = material.emissiveFactor.rgb;
    if (useTextures) emissive *= emissiveMap.sample(smp, in.uv).rgb;
    litColor += emissive;

    // litColor is unbounded linear HDR radiance (bright/overlapping lights can push it well past
    // 1.0) - written straight into the offscreen HDR color target, untouched. Exposure, tone
    // mapping, and gamma encoding all happen once, screen-space, in postProcessFragmentMain below,
    // rather than per-object here - see Main.cpp's HDR scene / post-process / overlay pass split.
    // Alpha only matters to the Blend pipeline (blending on); the opaque one ignores it.
    SceneFragmentOut out;
    out.color = float4(litColor, alphaMode == ALPHA_MODE_BLEND ? alpha : 1.0);

    // G-buffer for screen-space reflections: everything the image-based specular term above was
    // scaled by except the environment's own radiance and intensity (the SSR pass swaps that radiance
    // for a ray-traced one and multiplies the same weight back in, so a traced reflection is shaded
    // exactly like the environment reflection it replaces - Fresnel, roughness, occlusion). Glass
    // and blended surfaces get zero: what is behind them isn't what the depth buffer describes.
    bool reflective = transmission == 0.0 && alphaMode != ALPHA_MODE_BLEND;
    out.normalRoughness = float4(normal, roughness);
    out.specularWeight = float4(reflective ? (F_ibl * envBRDF.x + envBRDF.y) * specularAO * ao : float3(0.0), 0.0);
    return out;
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

// --- Ambient occlusion (GTAO by default: Jimenez et al., "Practical Real-Time Strategies for Accurate Indirect
// Occlusion", 2016, in the form XeGTAO popularized). Three passes run before the main scene pass
// (see Main.cpp): a depth + normal prepass, the occlusion estimate itself, and a depth-aware blur.
// The main pass then multiplies the result into its ambient/image-based light (see fragmentMain).

// Depth prepass: the opaque geometry's depth (hardware depth attachment) and world-space normal
// (color). Mask materials discard exactly like the shadow pass does, so a cut-out leaf casts no
// occlusion where it is transparent. Draws through the same lambda as the shadow casters, so it
// binds the same slots: uniforms at vertex buffer 1, albedo at texture 0, material at buffer 4.
struct AOPrepassRasterData {
    float4 position [[position]];
    float2 uv;
    float3 worldNormal;
};

vertex AOPrepassRasterData aoPrepassVertexMain(VertexInput in [[stage_in]],
                                               constant Uniforms& uniforms [[buffer(1)]]) {
    AOPrepassRasterData out;
    out.position = uniforms.mvpMatrix * float4(in.position, 1.0);
    out.uv = in.uv;
    out.worldNormal = (uniforms.modelMatrix * float4(in.normal, 0.0)).xyz;
    return out;
}

fragment float4 aoPrepassFragmentMain(AOPrepassRasterData in [[stage_in]],
                                      constant MaterialParams& material [[buffer(4)]],
                                      texture2d<float> albedoMap [[texture(0)]]) {
    if (int(material.alphaParams.x + 0.5) == ALPHA_MODE_MASK) {
        constexpr sampler alphaSampler(filter::linear, mip_filter::linear, address::repeat);
        float alpha = albedoMap.sample(alphaSampler, in.uv).a * material.baseColorFactor.a;
        if (alpha < material.alphaParams.y) discard_fragment();
    }
    return float4(normalize(in.worldNormal), 0.0);
}

struct AOParams {
    float4x4 viewMatrix; // world -> view, to bring the prepass's world normals into view space
    float projScaleX;    // the projection matrix's [0][0] and [1][1] - reconstruct view position from depth
    float projScaleY;
    float radius;        // world-space reach of the occlusion search
    float strength;      // power the visibility is raised to
    float mode;          // AO_MODE_* (cast to int)
    float noiseSeed;     // 0..1, different every frame under TAA so it averages the noise away (0 = a fixed pattern)
    float pixelScale;    // full-res pixels per AO pixel (2 = the AO buffer is half resolution); each AO pixel is
                         // computed at the depth/normal of the full-res pixel it starts at, see aoUpsampleFragmentMain
};

// Must match the AmbientOcclusionMode enum in Scene.hpp
#define AO_MODE_GTAO 0
#define AO_MODE_SSAO 1

#define AO_SSAO_SAMPLES 16     // hemisphere taps per pixel in the classic SSAO mode
#define AO_SLICES 3            // directions searched per pixel (each looks both ways along its line)
#define AO_STEPS 6             // depth taps per direction per side
#define AO_MAX_PIXEL_RADIUS 96.0 // cap on the screen-space search reach, so nearby surfaces stay cache-friendly

// Hardware depth ([0,1]) -> positive distance along the view axis (reverses the projection matrix,
// same formula the post-process pass uses for Depth of Field).
static float aoLinearDepth(float rawDepth) {
    return (CAMERA_FAR_PLANE * CAMERA_NEAR_PLANE)
         / (CAMERA_FAR_PLANE - rawDepth * (CAMERA_FAR_PLANE - CAMERA_NEAR_PLANE));
}

// View-space position (x right, y up, camera looks down -Z) of a pixel centre at the given depth.
static float3 aoViewPosition(float2 pixelCenter, float rawDepth, float2 size, constant AOParams& params) {
    float2 ndc = float2(pixelCenter.x / size.x * 2.0 - 1.0, 1.0 - pixelCenter.y / size.y * 2.0);
    float z = aoLinearDepth(rawDepth);
    return float3(ndc.x * z / params.projScaleX, ndc.y * z / params.projScaleY, -z);
}

// Classic SSAO (Crytek-style, normal-oriented hemisphere): scatter AO_SSAO_SAMPLES points through the
// hemisphere above the surface, project each back to the screen, and count it as occluded when the
// depth buffer there is in front of it. Cosine-weighted directions with lengths that bunch up near the
// surface, the whole pattern spun about the normal by per-pixel noise (the blur pass averages that
// away). A range check fades an occluder out when it sits far in front of the sample rather than
// close to it, so a distant foreground object doesn't darken what's behind it.
static float aoClassicSSAO(uint2 pixel, float3 P, float3 N, float2 size, float noise,
                           depth2d<float> depthTexture, constant AOParams& params) {
    float3 helper = abs(N.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(helper, N));
    float3 bitangent = cross(N, tangent);

    float bias = 0.05 * params.radius;
    float occlusion = 0.0;
    for (int i = 0; i < AO_SSAO_SAMPLES; i++) {
        float u = (float(i) + 0.5) / float(AO_SSAO_SAMPLES);
        float phi = float(i) * 2.39996323 + noise * 2.0 * PI; // golden angle spiral, rotated per pixel
        float r = sqrt(u);
        float3 local = float3(r * cos(phi), r * sin(phi), sqrt(1.0 - u)); // z = along the normal
        float scale = mix(0.1, 1.0, u * u);
        float3 samplePos = P + (tangent * local.x + bitangent * local.y + N * local.z) * (params.radius * scale);

        float sampleDist = -samplePos.z; // positive distance along the view axis
        if (sampleDist <= 0.0) continue;
        float2 ndc = float2(samplePos.x * params.projScaleX, samplePos.y * params.projScaleY) / sampleDist;
        float2 samplePixel = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * size;
        if (any(samplePixel < 0.0) || any(samplePixel >= size)) continue;

        float d = depthTexture.read(uint2(samplePixel));
        if (d >= 1.0) continue; // sky: nothing there
        float sceneDist = aoLinearDepth(d);
        float rangeCheck = smoothstep(0.0, 1.0, params.radius / max(abs(-P.z - sceneDist), 1e-4));
        occlusion += (sceneDist <= sampleDist - bias ? 1.0 : 0.0) * rangeCheck;
    }
    return 1.0 - occlusion / float(AO_SSAO_SAMPLES);
}

fragment float aoFragmentMain(PostProcessVertexOut in [[stage_in]],
                              depth2d<float> depthTexture [[texture(0)]],
                              texture2d<float> normalTexture [[texture(1)]],
                              constant AOParams& params [[buffer(0)]]) {
    // Everything below works in full-resolution pixels, whatever size the AO target is.
    uint2 pixel = min(uint2(floor(in.position.xy)) * uint(params.pixelScale + 0.5),
                      uint2(depthTexture.get_width() - 1, depthTexture.get_height() - 1));
    float rawDepth = depthTexture.read(pixel);
    if (rawDepth >= 1.0) return 1.0; // sky / nothing drawn: nothing to occlude

    float2 size = float2(depthTexture.get_width(), depthTexture.get_height());
    float3 P = aoViewPosition(float2(pixel) + 0.5, rawDepth, size, params);
    // Nudge toward the camera a hair so a flat surface's own neighbours (whose reconstructed depth
    // wobbles by a few ulp) don't read as occluders sitting slightly in front of it.
    P *= 0.9992;
    float3 V = normalize(-P);

    float3 N = normalize((params.viewMatrix * float4(normalTexture.read(pixel).xyz, 0.0)).xyz);
    if (dot(N, V) < 0.0) N = -N; // back face seen from inside: shade it as facing the camera

    float pixelRadius = params.radius * 0.5 * params.projScaleY * size.y / -P.z;
    if (pixelRadius < 1.5) return 1.0; // too far away for the search to reach a neighbouring pixel
    pixelRadius = min(pixelRadius, AO_MAX_PIXEL_RADIUS);

    // Two decorrelated per-pixel noise values: a rotation of the slice pattern and a jitter of the
    // tap distances. The blur pass below averages the resulting noise away.
    float noiseSlice = fract(0.5 + dot(float2(pixel), float2(0.7548776662, 0.5698402910)) + params.noiseSeed);
    float noiseStep = fract(52.9829189 * fract(dot(float2(pixel), float2(0.06711056, 0.00583715))) + params.noiseSeed * 0.7548776662);

    // Samples fade out toward the edge of the radius so an occluder entering/leaving the search
    // sphere doesn't pop.
    constexpr float falloffRange = 0.615;
    float falloffMul = -1.0 / (falloffRange * params.radius);
    float falloffAdd = (1.0 - falloffRange) / falloffRange + 1.0;
    float minS = 1.3 / pixelRadius; // keep even the nearest tap out of the centre pixel

    if (int(params.mode + 0.5) == AO_MODE_SSAO) {
        return max(pow(aoClassicSSAO(pixel, P, N, size, noiseStep, depthTexture, params), params.strength), 0.03);
    }

    float visibility = 0.0;
    for (int slice = 0; slice < AO_SLICES; slice++) {
        float phi = (float(slice) + noiseSlice) / float(AO_SLICES) * PI;
        float cosPhi = cos(phi), sinPhi = sin(phi);
        float2 omega = float2(cosPhi, -sinPhi); // pixel space: +y is down, so flip against view-space y
        float3 directionVec = float3(cosPhi, sinPhi, 0.0);

        // The slice's plane contains V and directionVec. Project the normal into it: the angle
        // between that projection and V ("n") is what bounds the visible arc on each side.
        float3 orthoDirectionVec = directionVec - dot(directionVec, V) * V;
        float3 axisVec = normalize(cross(directionVec, V));
        float3 projectedNormal = N - axisVec * dot(N, axisVec);
        float projectedNormalLength = length(projectedNormal);
        float signNorm = sign(dot(orthoDirectionVec, projectedNormal));
        float cosNorm = saturate(dot(projectedNormal, V) / max(projectedNormalLength, 1e-5));
        float n = signNorm * acos(cosNorm);

        // Start each side at the tangent plane's horizon (nothing occludes below it), then raise it to
        // the highest occluder found marching outward.
        float lowHorizonCos0 = cos(n + PI * 0.5);
        float lowHorizonCos1 = cos(n - PI * 0.5);
        float horizonCos0 = lowHorizonCos0;
        float horizonCos1 = lowHorizonCos1;

        for (int tap = 0; tap < AO_STEPS; tap++) {
            float s = (float(tap) + noiseStep) / float(AO_STEPS);
            s *= s; // more taps near the centre, where occlusion changes fastest
            s = s + minS - s * minS;
            float2 offset = round(omega * (s * pixelRadius));

            float2 samplePixel0 = float2(pixel) + offset;
            float2 samplePixel1 = float2(pixel) - offset;

            if (all(samplePixel0 >= 0.0) && all(samplePixel0 < size)) {
                float d = depthTexture.read(uint2(samplePixel0));
                if (d < 1.0) {
                    float3 delta = aoViewPosition(floor(samplePixel0) + 0.5, d, size, params) - P;
                    float dist = length(delta);
                    float shc = dot(delta, V) / max(dist, 1e-5);
                    shc = mix(lowHorizonCos0, shc, saturate(dist * falloffMul + falloffAdd));
                    horizonCos0 = max(horizonCos0, shc);
                }
            }
            if (all(samplePixel1 >= 0.0) && all(samplePixel1 < size)) {
                float d = depthTexture.read(uint2(samplePixel1));
                if (d < 1.0) {
                    float3 delta = aoViewPosition(floor(samplePixel1) + 0.5, d, size, params) - P;
                    float dist = length(delta);
                    float shc = dot(delta, V) / max(dist, 1e-5);
                    shc = mix(lowHorizonCos1, shc, saturate(dist * falloffMul + falloffAdd));
                    horizonCos1 = max(horizonCos1, shc);
                }
            }
        }

        // Horizon angles, clamped to the hemisphere around the normal, integrated analytically
        // (cosine-weighted) over the visible arc between them.
        float h0 = -acos(clamp(horizonCos1, -1.0, 1.0));
        float h1 = acos(clamp(horizonCos0, -1.0, 1.0));
        h0 = n + clamp(h0 - n, -PI * 0.5, PI * 0.5);
        h1 = n + clamp(h1 - n, -PI * 0.5, PI * 0.5);
        float iarc0 = (cosNorm + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) * 0.25;
        float iarc1 = (cosNorm + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) * 0.25;
        visibility += projectedNormalLength * (iarc0 + iarc1);
    }

    visibility = saturate(visibility / float(AO_SLICES));
    return max(pow(visibility, params.strength), 0.03);
}

// Separable 7-tap bilateral blur (run horizontally, then vertically): averages away the per-pixel
// noise aoFragmentMain leaves, but weights each tap down when its depth differs from the centre's,
// so the occlusion of a near object doesn't bleed onto the far surface behind its silhouette.
fragment float aoBlurFragmentMain(PostProcessVertexOut in [[stage_in]],
                                  texture2d<float> aoTexture [[texture(0)]],
                                  depth2d<float> depthTexture [[texture(1)]],
                                  constant float2& direction [[buffer(0)]]) {
    int2 pixel = int2(in.position.xy);
    int2 maxPixel = int2(aoTexture.get_width() - 1, aoTexture.get_height() - 1);
    // An AO texel stands for the full-res pixel it was computed at (pixelScale times its coordinates).
    int scale = int(float(depthTexture.get_width()) / float(aoTexture.get_width()) + 0.5);
    int2 maxDepthPixel = int2(depthTexture.get_width() - 1, depthTexture.get_height() - 1);
    float centerRaw = depthTexture.read(uint2(min(pixel * scale, maxDepthPixel)));
    if (centerRaw >= 1.0) return 1.0;
    float centerDepth = aoLinearDepth(centerRaw);

    float sum = 0.0;
    float weightSum = 0.0;
    for (int i = -3; i <= 3; i++) {
        int2 p = clamp(pixel + int2(direction) * i, int2(0), maxPixel);
        float rawDepth = depthTexture.read(uint2(min(p * scale, maxDepthPixel)));
        float spatial = exp(-float(i * i) / 8.0);
        // Sky taps count as infinitely far away (weight 0); otherwise the weight falls off linearly
        // to 0 at a 4% relative depth difference.
        float depthWeight = rawDepth >= 1.0 ? 0.0
            : saturate(1.0 - abs(aoLinearDepth(rawDepth) - centerDepth) / (0.04 * centerDepth));
        float w = spatial * depthWeight;
        sum += aoTexture.read(uint2(p)).r * w;
        weightSum += w;
    }
    return sum / max(weightSum, 1e-4);
}

// Brings the (blurred, possibly lower-resolution) AO buffer up to full resolution for the scene pass to
// read: for each full-res pixel, a bilinear blend of the four nearest AO texels, each weighted down by how
// far its own depth is from this pixel's - so AO computed on a near surface doesn't bleed onto a far one
// across a silhouette (a plain bilinear upsample would smear a dark halo there). At scale 1 it reduces to
// a copy. An AO texel's depth is that of the full-res pixel it was computed at, as in aoFragmentMain.
fragment float aoUpsampleFragmentMain(PostProcessVertexOut in [[stage_in]],
                                      texture2d<float> aoTexture [[texture(0)]],
                                      depth2d<float> depthTexture [[texture(1)]]) {
    uint2 pixel = uint2(in.position.xy);
    float rawDepth = depthTexture.read(pixel);
    if (rawDepth >= 1.0) return 1.0;

    float scale = float(depthTexture.get_width()) / float(aoTexture.get_width());
    int2 maxTexel = int2(aoTexture.get_width() - 1, aoTexture.get_height() - 1);
    int2 maxDepthPixel = int2(depthTexture.get_width() - 1, depthTexture.get_height() - 1);
    float2 lowPosition = float2(pixel) / scale; // AO texel h was computed at full-res pixel h * scale
    int2 base = int2(floor(lowPosition));
    float2 fraction = lowPosition - float2(base);
    float centerDepth = aoLinearDepth(rawDepth);

    float sum = 0.0, weightSum = 0.0;
    for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
            int2 texel = clamp(base + int2(dx, dy), int2(0), maxTexel);
            float tapRaw = depthTexture.read(uint2(min(int2(float2(texel) * scale + 0.5), maxDepthPixel)));
            float bilinear = (dx == 0 ? 1.0 - fraction.x : fraction.x) * (dy == 0 ? 1.0 - fraction.y : fraction.y);
            // Same depth tolerance as the blur pass (weight 0 at a 4% relative difference), with a small floor
            // so a pixel with no similar neighbour still gets something rather than dividing by zero.
            float depthWeight = tapRaw >= 1.0 ? 1e-3
                : max(saturate(1.0 - abs(aoLinearDepth(tapRaw) - centerDepth) / (0.04 * centerDepth)), 1e-3);
            float w = bilinear * depthWeight;
            sum += aoTexture.read(uint2(texel)).r * w;
            weightSum += w;
        }
    }
    return sum / max(weightSum, 1e-6);
}

// --- Screen-space reflections. Two full-screen passes after the whole scene (opaque, sky and glass) has
// been drawn: ssrTraceFragmentMain, run at reduced resolution, finds what each ray hits, and
// ssrCompositeFragmentMain upsamples that and applies it at full resolution. The scene pass leaves a small
// G-buffer (see SceneFragmentOut): each pixel's shading normal, roughness and a specular weight. For every
// reflective pixel the trace marches the mirror
// direction through the depth buffer in screen space; where the ray finds a surface, the color there
// (read from a mip-chained copy of the finished frame, blurrier for rougher surfaces) replaces the
// environment-map reflection that fragmentMain already added, and where it finds nothing (the ray
// leaves the screen, passes behind something, or hits a back face) the environment reflection
// stands. The result is written additively into the HDR target as (traced - environment) * weight, so
// the two blend without the scene pass having to know whether SSR will succeed.

struct SSRParams {
    float4x4 viewMatrix;      // world -> view
    float projScaleX;         // the projection matrix's [0][0] and [1][1] - project/reconstruct view positions
    float projScaleY;
    float strength;           // Scene::ssrStrength
    float maxDistance;        // Scene::ssrMaxDistance, world units
    float thickness;          // Scene::ssrThickness, world units
    float environmentIntensity;
    float noiseSeed;          // 0..1, different every frame under TAA so it averages the noise away (0 = a fixed pattern)
    float pixelScale;         // full-res pixels per trace pixel (2 = the trace runs at half resolution)
};

#define SSR_MAX_STEPS 80          // depth-buffer taps per ray (the stride widens to cover long rays)
#define SSR_REFINE_STEPS 6        // bisection steps that pin a hit down inside the coarse step that found it
#define SSR_MAX_ROUGHNESS 0.75    // past this the lobe is far too wide for one ray: leave it to the environment map
#define SSR_MAX_HIT_RADIANCE 32.0 // caps hot spots (a lamp seen in a mirror) so they can't sparkle at low mips

// View-space point -> pixel position.
static float2 ssrProject(float3 viewPos, float2 size, constant SSRParams& params) {
    float2 ndc = float2(viewPos.x * params.projScaleX, viewPos.y * params.projScaleY) / -viewPos.z;
    return float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * size;
}

// View-space position of a pixel centre at the given hardware depth.
static float3 ssrViewPosition(float2 pixelCenter, float rawDepth, float2 size, constant SSRParams& params) {
    float2 ndc = float2(pixelCenter.x / size.x * 2.0 - 1.0, 1.0 - pixelCenter.y / size.y * 2.0);
    float z = aoLinearDepth(rawDepth);
    return float3(ndc.x * z / params.projScaleX, ndc.y * z / params.projScaleY, -z);
}

// Pass 1, run at (usually) half resolution: marches one ray per trace pixel, from the surface at the
// full-res pixel that trace pixel starts at, and writes what it found as (color, confidence) - the hit's
// radiance and how much to trust it, 0 for a miss. Everything per-pixel-exact (the reflectance weight, the
// environment it replaces) waits for the composite pass, which has the full-res G-buffer.
fragment float4 ssrTraceFragmentMain(PostProcessVertexOut in [[stage_in]],
                                     depth2d<float> depthTexture [[texture(0)]],
                                     texture2d<float> normalRoughnessTexture [[texture(1)]],
                                     texture2d<float> specularWeightTexture [[texture(2)]],
                                     texture2d<float> sceneColor [[texture(3)]],
                                     constant SSRParams& params [[buffer(0)]],
                                     sampler colorSampler [[sampler(0)]]) {
    uint2 pixel = min(uint2(floor(in.position.xy)) * uint(params.pixelScale + 0.5),
                      uint2(depthTexture.get_width() - 1, depthTexture.get_height() - 1));
    float rawDepth = depthTexture.read(pixel);
    if (rawDepth >= 1.0) return float4(0.0); // sky

    float3 weight = specularWeightTexture.read(pixel).rgb;
    float4 normalRoughness = normalRoughnessTexture.read(pixel);
    float roughness = normalRoughness.w;
    float roughnessFade = 1.0 - smoothstep(SSR_MAX_ROUGHNESS - 0.3, SSR_MAX_ROUGHNESS, roughness);
    if (max(weight.r, max(weight.g, weight.b)) < 0.002 || roughnessFade <= 0.0) return float4(0.0);

    float2 size = float2(depthTexture.get_width(), depthTexture.get_height());
    float3 P = ssrViewPosition(float2(pixel) + 0.5, rawDepth, size, params);
    float3 V = normalize(-P);

    // Shading normal in view space; a back face seen from inside is shaded as if it faced the camera.
    float3x3 viewRotation = float3x3(params.viewMatrix[0].xyz, params.viewMatrix[1].xyz, params.viewMatrix[2].xyz);
    float3 N = normalize(viewRotation * normalRoughness.xyz);
    if (dot(N, V) < 0.0) N = -N;
    float3 R = reflect(-V, N);
    float3 worldR = transpose(viewRotation) * R; // a rotation's inverse is its transpose

    // A mirror direction that points back at the camera can only hit things behind it, which the depth
    // buffer never saw - fade those out (the environment map takes over).
    float facing = 1.0 - smoothstep(0.1, 0.6, R.z);
    if (facing <= 0.0) return float4(0.0);

    // Segment to trace: from just off the surface along R, clipped to the near/far planes (a ray that
    // crosses the camera plane has no projection) and then to the screen rectangle, so the step budget
    // is spent on pixels that can actually be tested.
    float3 P0 = P + N * (0.01 + 0.004 * -P.z);
    float tMax = params.maxDistance;
    if (R.z > 0.0) tMax = min(tMax, (-CAMERA_NEAR_PLANE * 1.05 - P0.z) / R.z);
    if (R.z < 0.0) tMax = min(tMax, (-CAMERA_FAR_PLANE * 0.99 - P0.z) / R.z);
    if (tMax <= 0.0) return float4(0.0);
    float3 P1 = P0 + R * tMax;

    float2 p0 = ssrProject(P0, size, params);
    float2 p1 = ssrProject(P1, size, params);
    float k0 = 1.0 / -P0.z; // 1/depth is linear across the screen, so it (not depth) is what gets stepped
    float k1 = 1.0 / -P1.z;

    float2 delta = p1 - p0;
    float tClip = 1.0;
    if (delta.x > 0.0) tClip = min(tClip, (size.x - 1.0 - p0.x) / delta.x);
    else if (delta.x < 0.0) tClip = min(tClip, -p0.x / delta.x);
    if (delta.y > 0.0) tClip = min(tClip, (size.y - 1.0 - p0.y) / delta.y);
    else if (delta.y < 0.0) tClip = min(tClip, -p0.y / delta.y);
    tClip = clamp(tClip, 0.0, 1.0);
    p1 = p0 + delta * tClip;
    k1 = mix(k0, k1, tClip);

    float2 span = abs(p1 - p0);
    float pixelLength = max(span.x, span.y);
    if (pixelLength < 1.5) return float4(0.0);
    int steps = int(min(float(SSR_MAX_STEPS), ceil(pixelLength)));

    // Per-pixel offset of the sample positions, so the stride's banding turns into fine noise.
    float jitter = fract(52.9829189 * fract(dot(float2(pixel), float2(0.06711056, 0.00583715))) + params.noiseSeed);

    bool hit = false;
    float hitT = 0.0;
    float hitSceneZ = 0.0;
    float prevT = 0.0;
    for (int i = 0; i < steps; i++) {
        float t = (float(i) + 1.0 + jitter) / (float(steps) + 1.0);
        float2 pp = mix(p0, p1, t);
        float rayZ = 1.0 / mix(k0, k1, t);
        float rayZPrev = 1.0 / mix(k0, k1, prevT);

        float sceneRaw = depthTexture.read(uint2(pp));
        if (sceneRaw < 1.0) {
            float sceneZ = aoLinearDepth(sceneRaw);
            // The ray's depth span over this step overlaps the slab [surface, surface + thickness].
            if (max(rayZ, rayZPrev) >= sceneZ && min(rayZ, rayZPrev) <= sceneZ + params.thickness) {
                // Bisect inside the step to land on the crossing instead of its far end.
                float lo = prevT, hi = t;
                for (int r = 0; r < SSR_REFINE_STEPS; r++) {
                    float mid = 0.5 * (lo + hi);
                    float2 pm = mix(p0, p1, mid);
                    float zm = 1.0 / mix(k0, k1, mid);
                    float raw = depthTexture.read(uint2(pm));
                    if (raw < 1.0 && zm >= aoLinearDepth(raw)) hi = mid; else lo = mid;
                }
                float2 ph = mix(p0, p1, hi);
                float rawHit = depthTexture.read(uint2(ph));
                if (rawHit < 1.0) {
                    float zHit = aoLinearDepth(rawHit);
                    if (1.0 / mix(k0, k1, hi) - zHit <= params.thickness) {
                        hit = true;
                        hitT = hi;
                        hitSceneZ = zHit;
                        break;
                    }
                }
            }
        }
        prevT = t;
    }
    if (!hit) return float4(0.0);

    float2 hitPixel = mix(p0, p1, hitT);
    float2 hitUV = hitPixel / size;

    // Confidence: fade out everything that makes a hit untrustworthy - close to the screen border (the
    // ray is about to leave the known image), far away, a back face, a rough surface, a grazing mirror.
    float2 edge = min(hitUV, 1.0 - hitUV);
    float edgeFade = smoothstep(0.0, 0.1, min(edge.x, edge.y));
    float3 hitView = ssrViewPosition(floor(hitPixel) + 0.5, depthTexture.read(uint2(hitPixel)), size, params);
    float distanceFade = 1.0 - smoothstep(0.6 * params.maxDistance, params.maxDistance, length(hitView - P));
    float3 hitNormal = normalRoughnessTexture.read(uint2(hitPixel)).xyz;
    float backFade = 1.0 - smoothstep(0.0, 0.3, dot(hitNormal, worldR));
    float confidence = params.strength * roughnessFade * facing * edgeFade * distanceFade * backFade;
    if (confidence <= 0.0) return float4(0.0);

    // Rougher reflections are blurrier, and blurrier the farther the reflected thing is: pick the mip
    // whose texel spans the lobe's footprint at the hit.
    float footprint = length(hitPixel - (float2(pixel) + 0.5)) * roughness * roughness * 0.6;
    float lod = clamp(log2(max(footprint, 1.0)), 0.0, float(sceneColor.get_num_mip_levels() - 1));
    float3 traced = min(sceneColor.sample(colorSampler, hitUV, level(lod)).rgb, SSR_MAX_HIT_RADIANCE);

    return float4(traced, confidence);
}

// Pass 2, full resolution: takes the trace result (bilaterally upsampled from its own resolution - see
// aoUpsampleFragmentMain for why not plain bilinear) and applies it to each pixel exactly: swap the
// environment radiance the scene pass put into this pixel for the traced one, scaled by the same reflectance
// weight, and add the difference into the HDR image.
fragment float4 ssrCompositeFragmentMain(PostProcessVertexOut in [[stage_in]],
                                         depth2d<float> depthTexture [[texture(0)]],
                                         texture2d<float> normalRoughnessTexture [[texture(1)]],
                                         texture2d<float> specularWeightTexture [[texture(2)]],
                                         texture2d<float> sceneColor [[texture(3)]],
                                         texturecube<float> prefilterMap [[texture(4)]],
                                         texture2d<float> traceTexture [[texture(5)]],
                                         constant SSRParams& params [[buffer(0)]],
                                         sampler envSampler [[sampler(0)]]) {
    uint2 pixel = uint2(in.position.xy);
    float rawDepth = depthTexture.read(pixel);
    if (rawDepth >= 1.0) return float4(0.0); // sky
    float3 weight = specularWeightTexture.read(pixel).rgb;
    if (max(weight.r, max(weight.g, weight.b)) < 0.002) return float4(0.0);

    float2 size = float2(depthTexture.get_width(), depthTexture.get_height());
    float2 traceSize = float2(traceTexture.get_width(), traceTexture.get_height());
    float scale = size.x / traceSize.x;
    float2 lowPosition = float2(pixel) / scale; // trace texel h was traced from full-res pixel h * scale
    int2 base = int2(floor(lowPosition));
    float2 fraction = lowPosition - float2(base);
    float centerDepth = aoLinearDepth(rawDepth);

    // Color is averaged over the texels that hit (weighted by their confidence), confidence over all of them:
    // a pixel bordered by hits and misses gets the hits' color at a fractional strength, not a darkened one.
    float3 tracedSum = float3(0.0);
    float confidenceSum = 0.0, weightSum = 0.0;
    for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
            int2 texel = clamp(base + int2(dx, dy), int2(0), int2(traceSize) - 1);
            float tapRaw = depthTexture.read(uint2(min(int2(float2(texel) * scale + 0.5), int2(size) - 1)));
            float bilinear = (dx == 0 ? 1.0 - fraction.x : fraction.x) * (dy == 0 ? 1.0 - fraction.y : fraction.y);
            float depthWeight = tapRaw >= 1.0 ? 1e-3
                : max(saturate(1.0 - abs(aoLinearDepth(tapRaw) - centerDepth) / (0.04 * centerDepth)), 1e-3);
            float w = bilinear * depthWeight;
            float4 tap = traceTexture.read(uint2(texel));
            tracedSum += tap.rgb * tap.a * w;
            confidenceSum += tap.a * w;
            weightSum += w;
        }
    }
    if (confidenceSum <= 1e-5) return float4(0.0);
    float confidence = confidenceSum / weightSum;
    float3 traced = tracedSum / confidenceSum;

    // This pixel's own mirror direction (as the trace pass derived it) to look the environment up.
    float4 normalRoughness = normalRoughnessTexture.read(pixel);
    float roughness = normalRoughness.w;
    float3 P = ssrViewPosition(float2(pixel) + 0.5, rawDepth, size, params);
    float3 V = normalize(-P);
    float3x3 viewRotation = float3x3(params.viewMatrix[0].xyz, params.viewMatrix[1].xyz, params.viewMatrix[2].xyz);
    float3 N = normalize(viewRotation * normalRoughness.xyz);
    if (dot(N, V) < 0.0) N = -N;
    float3 worldR = transpose(viewRotation) * reflect(-V, N); // a rotation's inverse is its transpose

    // The environment radiance the scene pass already put into this pixel (same lookup as fragmentMain).
    float envLod = roughness * float(prefilterMap.get_num_mip_levels() - 1);
    float3 environment = prefilterMap.sample(envSampler, worldR, level(envLod)).rgb * params.environmentIntensity;

    // Swap environment for traced, scaled like the environment term was. Never subtract more than is
    // there (float rounding between this reconstruction and the scene pass could push a dark pixel
    // slightly negative, which the tone mapper's pow() turns into NaN).
    float3 correction = confidence * (traced - environment) * weight;
    correction = max(correction, -sceneColor.read(pixel).rgb);
    return float4(correction, 0.0);
}

// --- Temporal anti-aliasing. The scene is rendered with a different sub-pixel camera offset every frame
// (see Main.cpp's jitter), so each frame samples the geometry at slightly different positions inside
// every pixel. This full-screen pass, run on the finished HDR frame (after SSR), reprojects last
// frame's result to where each pixel was, and blends it with the current frame - accumulating those
// samples turns jagged edges into properly anti-aliased ones, and averages away the per-frame noise the
// AO and SSR passes leave. The catch is that history can be wrong (something moved, something was
// uncovered), so it is clipped into the colour range of the current frame's 3x3 neighbourhood before it
// is blended: history that disagrees with what is on screen now is pulled toward it instead of ghosting.
//
// Reprojection is from depth with the camera's own motion only (there is no per-object velocity buffer):
// exact for the static scene, and an object that moves relative to it leaves a trail that the clip
// shortens to a few frames.

struct TAAParams {
    float4x4 reprojectionMatrix; // previous frame's view-proj * inverse(this frame's), both un-jittered
    float feedback;              // Scene::taaFeedback, the most history the blend keeps
    float reset;                 // 1 = no usable history (first frame, resize, TAA just enabled): pass the frame through
};

// HDR colour to/from a bounded range (x / (1 + max channel)), so a few very bright pixels can't dominate
// the neighbourhood statistics or the blend and make the result flicker.
static float3 taaCompress(float3 c) {
    c = clamp(c, 0.0, 60000.0);
    return c / (1.0 + max(c.r, max(c.g, c.b)));
}
static float3 taaUncompress(float3 c) {
    return c / max(1.0 - max(c.r, max(c.g, c.b)), 1e-4);
}

// Catmull-Rom filtered read (5 bilinear taps, the corner taps dropped): plain bilinear history reads
// blur the image a little more every frame; this one keeps it sharp.
static float3 taaSampleCatmullRom(texture2d<float> tex, sampler s, float2 uv, float2 size) {
    float2 pos = uv * size;
    float2 center = floor(pos - 0.5) + 0.5;
    float2 f = pos - center;
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);
    float2 w12 = w1 + w2;
    float2 tc12 = (center + w2 / w12) / size;
    float2 tc0 = (center - 1.0) / size;
    float2 tc3 = (center + 2.0) / size;

    float wTop = w12.x * w0.y, wLeft = w0.x * w12.y, wMid = w12.x * w12.y, wRight = w3.x * w12.y, wBottom = w12.x * w3.y;
    float3 result = tex.sample(s, float2(tc12.x, tc0.y)).rgb * wTop
                  + tex.sample(s, float2(tc0.x, tc12.y)).rgb * wLeft
                  + tex.sample(s, tc12).rgb * wMid
                  + tex.sample(s, float2(tc3.x, tc12.y)).rgb * wRight
                  + tex.sample(s, float2(tc12.x, tc3.y)).rgb * wBottom;
    return max(result / (wTop + wLeft + wMid + wRight + wBottom), 0.0);
}

fragment float4 taaFragmentMain(PostProcessVertexOut in [[stage_in]],
                                texture2d<float> currentTexture [[texture(0)]],
                                texture2d<float> historyTexture [[texture(1)]],
                                depth2d<float> depthTexture [[texture(2)]],
                                constant TAAParams& params [[buffer(0)]],
                                sampler linearSampler [[sampler(0)]]) {
    int2 pixel = int2(in.position.xy);
    int2 maxPixel = int2(currentTexture.get_width() - 1, currentTexture.get_height() - 1);
    float2 size = float2(currentTexture.get_width(), currentTexture.get_height());

    float3 currentLinear = currentTexture.read(uint2(pixel)).rgb;
    if (params.reset > 0.5) return float4(currentLinear, 1.0);

    // The current frame's 3x3 neighbourhood, in compressed space: its mean and spread say what colours
    // this pixel can plausibly have.
    float3 current = taaCompress(currentLinear);
    float3 sum = float3(0.0), sumSquares = float3(0.0);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float3 c = (dx == 0 && dy == 0)
                ? current
                : taaCompress(currentTexture.read(uint2(clamp(pixel + int2(dx, dy), int2(0), maxPixel))).rgb);
            sum += c;
            sumSquares += c * c;
        }
    }
    float3 mean = sum / 9.0;
    float3 deviation = sqrt(max(sumSquares / 9.0 - mean * mean, 0.0));

    // Where this pixel was last frame: its depth back to a world point (through the un-jittered
    // matrices), which last frame's camera then projected somewhere else.
    float2 uv = (float2(pixel) + 0.5) / size;
    float4 previousClip = params.reprojectionMatrix * float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0,
                                                             depthTexture.read(uint2(pixel)), 1.0);
    if (previousClip.w <= 0.0) return float4(currentLinear, 1.0);
    float2 previousNDC = previousClip.xy / previousClip.w;
    float2 previousUV = float2(previousNDC.x * 0.5 + 0.5, 0.5 - previousNDC.y * 0.5);
    if (any(previousUV < 0.0) || any(previousUV > 1.0)) return float4(currentLinear, 1.0); // came from off-screen

    float3 history = taaCompress(taaSampleCatmullRom(historyTexture, linearSampler, previousUV, size));

    // Variance clipping: pull the history colour to the edge of the box mean +/- 1.5 sigma (along the
    // line from the box centre) when it lies outside - it can't be trusted to be the same surface then.
    float3 extent = 1.5 * deviation + 1e-4;
    float3 offset = history - mean;
    float outside = max(abs(offset.x) / extent.x, max(abs(offset.y) / extent.y, abs(offset.z) / extent.z));
    if (outside > 1.0) history = mean + offset / outside;

    // Less history while things are moving fast on screen: the reprojection is least exact there.
    float motionPixels = length((previousUV - uv) * size);
    float historyWeight = params.feedback * (1.0 - 0.5 * saturate(motionPixels / 8.0));

    return float4(taaUncompress(mix(current, history, historyWeight)), 1.0);
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

// --- Environment: sky background + the compute kernels that precompute the image-based lighting
// data (see Environment.cpp, which dispatches them once per environment, not per frame) ---

struct SkyParams {
    float4x4 invViewProj;
    float4 cameraPosition;
    float intensity;
};

// Same full-screen triangle as postProcessVertexMain, but at the far plane (z = 1) so the sky only
// shows where the scene pass left the depth buffer at its cleared value - drawn after the objects
// with depth test LessEqual / no depth write (see Main.cpp), so it costs nothing where geometry is.
vertex PostProcessVertexOut skyVertexMain(uint vertexID [[vertex_id]]) {
    PostProcessVertexOut out;
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    out.uv = uv;
    out.position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 1.0, 1.0);
    return out;
}

fragment float4 skyFragmentMain(PostProcessVertexOut in [[stage_in]],
                                constant SkyParams& params [[buffer(0)]],
                                texturecube<float> skyCube [[texture(0)]],
                                sampler envSampler [[sampler(2)]]) {
    // Unproject this pixel's far-plane point to world space; its direction from the camera is what
    // the cube map is looked up with. Only the direction matters, so the sky never parallaxes.
    float4 ndc = float4(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, 1.0, 1.0);
    float4 world = params.invViewProj * ndc;
    float3 dir = normalize(world.xyz / world.w - params.cameraPosition.xyz);
    return float4(skyCube.sample(envSampler, dir, level(0)).rgb * params.intensity, 1.0);
}

// Maps a cube face + [-1,1] face-local coordinate (u right, v down, in texture-row order) to the
// direction it represents - the standard cube-map face table, which is what the hardware uses when
// the finished cube is sampled by direction.
static float3 cubeDirection(uint face, float2 uv) {
    switch (face) {
        case 0:  return normalize(float3( 1.0, -uv.y, -uv.x)); // +X
        case 1:  return normalize(float3(-1.0, -uv.y,  uv.x)); // -X
        case 2:  return normalize(float3( uv.x,  1.0,  uv.y)); // +Y
        case 3:  return normalize(float3( uv.x, -1.0, -uv.y)); // -Y
        case 4:  return normalize(float3( uv.x, -uv.y,  1.0)); // +Z
        default: return normalize(float3(-uv.x, -uv.y, -1.0)); // -Z
    }
}

static float2 cubeFaceUV(uint2 pixel, uint size) {
    return (float2(pixel) + 0.5) / float(size) * 2.0 - 1.0;
}

// Built-in environment used when no .hdr file is chosen: a gradient sky with a small, very bright sun
// and a dim ground, so the default scene has directional ambient light and something to reflect
// without needing any asset on disk. Radiance is in linear HDR units like everything else.
static float3 proceduralSky(float3 dir) {
    const float3 horizonColor = float3(0.80, 0.85, 0.92);
    const float3 zenithColor = float3(0.18, 0.38, 0.85);
    const float3 groundColor = float3(0.16, 0.14, 0.12);
    const float3 sunDirection = normalize(float3(0.5, 0.45, 0.6));

    float h = dir.y;
    float3 sky = mix(horizonColor, zenithColor, powr(saturate(h), 0.6));
    float3 color = mix(sky, groundColor, smoothstep(0.0, -0.15, h));

    float sunCos = dot(dir, sunDirection);
    color += float3(1.0, 0.85, 0.6) * (powr(saturate(sunCos), 48.0) * 0.5);            // soft glow
    color += float3(1.0, 0.9, 0.7) * (150.0 * smoothstep(0.9994, 0.9998, sunCos));   // ~2 degree disc
    return color;
}

kernel void proceduralSkyToCubeKernel(texturecube<float, access::write> dst [[texture(0)]],
                                      uint3 gid [[thread_position_in_grid]]) {
    uint size = dst.get_width();
    if (gid.x >= size || gid.y >= size) return;
    float3 dir = cubeDirection(gid.z, cubeFaceUV(gid.xy, size));
    dst.write(float4(proceduralSky(dir), 1.0), gid.xy, gid.z);
}

// Resamples an equirectangular (lat-long) HDR panorama into a cube map. Image row 0 = straight up.
kernel void equirectToCubeKernel(texture2d<float> src [[texture(0)]],
                                 texturecube<float, access::write> dst [[texture(1)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    uint size = dst.get_width();
    if (gid.x >= size || gid.y >= size) return;
    float3 dir = cubeDirection(gid.z, cubeFaceUV(gid.xy, size));
    float2 uv = float2(atan2(dir.z, dir.x) / (2.0 * PI) + 0.5, acos(clamp(dir.y, -1.0, 1.0)) / PI);
    constexpr sampler eqSampler(filter::linear, s_address::repeat, t_address::clamp_to_edge);
    dst.write(float4(src.sample(eqSampler, uv, level(0)).rgb, 1.0), gid.xy, gid.z);
}

// Diffuse irradiance: for every output direction N, the cosine-weighted average of the environment
// over the hemisphere around N (already divided by PI, so the shader only multiplies by albedo). A
// brute-force angular sweep - fine for a one-time bake at 32x32 - reading a low mip of the source so
// a tiny bright sun contributes its energy smoothly instead of as a few noisy taps.
kernel void irradianceKernel(texturecube<float> src [[texture(0)]],
                             texturecube<float, access::write> dst [[texture(1)]],
                             uint3 gid [[thread_position_in_grid]]) {
    uint size = dst.get_width();
    if (gid.x >= size || gid.y >= size) return;
    float3 N = cubeDirection(gid.z, cubeFaceUV(gid.xy, size));
    float3 up = abs(N.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 right = normalize(cross(up, N));
    up = cross(N, right);

    constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
    constexpr float step = 0.05;
    float3 sum = float3(0.0);
    float count = 0.0;
    for (float phi = 0.0; phi < 2.0 * PI; phi += step) {
        for (float theta = 0.0; theta < 0.5 * PI; theta += step) {
            float3 t = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            float3 dir = t.x * right + t.y * up + t.z * N;
            sum += src.sample(s, dir, level(5.0)).rgb * cos(theta) * sin(theta);
            count += 1.0;
        }
    }
    dst.write(float4(PI * sum / count, 1.0), gid.xy, gid.z);
}

static float radicalInverse(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

// Low-discrepancy 2D point set for Monte Carlo integration.
static float2 hammersley(uint i, uint count) {
    return float2(float(i) / float(count), radicalInverse(i));
}

// Picks a half vector around N distributed like the GGX lobe, so samples land where the specular
// highlight actually has energy instead of being wasted uniformly over the sphere.
static float3 importanceSampleGGX(float2 Xi, float3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    float3 H = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    float3 up = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);
    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

// Specular prefilter: one dispatch per mip of the output cube, each at that mip's roughness. Uses the
// usual N = V = R assumption (so the highlight is isotropic - it loses the stretched grazing-angle
// reflections, an accepted trade-off of the split-sum method). Every tap reads the source at a mip
// chosen from the sample's solid angle, which removes the fireflies a tiny bright sun would cause.
kernel void prefilterKernel(texturecube<float> src [[texture(0)]],
                            texturecube<float, access::write> dst [[texture(1)]],
                            constant float& roughness [[buffer(0)]],
                            uint3 gid [[thread_position_in_grid]]) {
    uint size = dst.get_width();
    if (gid.x >= size || gid.y >= size) return;
    float3 N = cubeDirection(gid.z, cubeFaceUV(gid.xy, size));
    float3 V = N;

    constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
    constexpr uint sampleCount = 256;
    float srcSize = float(src.get_width());

    // A perfectly smooth mip (roughness 0) is just the environment itself: the GGX lobe collapses to
    // a single direction, and its normal-distribution term is 0/0 = NaN there. Read the source mip
    // whose resolution matches this output's so the copy isn't aliased by point-sampling mip 0.
    if (roughness < 1e-3) {
        dst.write(float4(src.sample(s, N, level(log2(srcSize / float(size)))).rgb, 1.0), gid.xy, gid.z);
        return;
    }

    float texelSolidAngle = 4.0 * PI / (6.0 * srcSize * srcSize);

    float3 color = float3(0.0);
    float weight = 0.0;
    for (uint i = 0; i < sampleCount; i++) {
        float3 H = importanceSampleGGX(hammersley(i, sampleCount), N, roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);
        float NdotL = dot(N, L);
        if (NdotL <= 0.0) continue;

        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);
        float D = distributionGGX(NdotH, roughness);
        float pdf = D * NdotH / (4.0 * HdotV) + 1e-4;
        float sampleSolidAngle = 1.0 / (float(sampleCount) * pdf + 1e-4);
        float mip = roughness == 0.0 ? 0.0 : 0.5 * log2(sampleSolidAngle / texelSolidAngle);

        color += src.sample(s, L, level(mip)).rgb * NdotL;
        weight += NdotL;
    }
    dst.write(float4(color / max(weight, 1e-4), 1.0), gid.xy, gid.z);
}

// Geometry term with the IBL remap of k (a/2, not the direct-lighting (a+1)^2/8 used above).
static float geometrySchlickGGXIBL(float NdotX, float roughness) {
    float k = (roughness * roughness) / 2.0;
    return NdotX / (NdotX * (1.0 - k) + k);
}

// BRDF lookup table: x = NdotV, y = roughness -> (scale, bias) applied to F0 in fragmentMain. It only
// depends on the BRDF, not on any environment, so it's baked once for the whole app.
kernel void brdfLUTKernel(texture2d<float, access::write> dst [[texture(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint size = dst.get_width();
    if (gid.x >= size || gid.y >= size) return;
    float NdotV = max((float(gid.x) + 0.5) / float(size), 1e-3);
    float roughness = (float(gid.y) + 0.5) / float(size);

    float3 V = float3(sqrt(1.0 - NdotV * NdotV), 0.0, NdotV);
    float3 N = float3(0.0, 0.0, 1.0);

    constexpr uint sampleCount = 1024;
    float A = 0.0, B = 0.0;
    for (uint i = 0; i < sampleCount; i++) {
        float3 H = importanceSampleGGX(hammersley(i, sampleCount), N, roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);
        float NdotL = saturate(L.z);
        float NdotH = saturate(H.z);
        float VdotH = saturate(dot(V, H));
        if (NdotL <= 0.0) continue;

        float G = geometrySchlickGGXIBL(NdotV, roughness) * geometrySchlickGGXIBL(NdotL, roughness);
        float G_Vis = (G * VdotH) / max(NdotH * NdotV, 1e-4);
        float Fc = powr(1.0 - VdotH, 5.0);
        A += (1.0 - Fc) * G_Vis;
        B += Fc * G_Vis;
    }
    dst.write(float4(A / float(sampleCount), B / float(sampleCount), 0.0, 1.0), gid);
}
