#include "Scene.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace {

constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;

simd::float4x4 scaleMatrix(simd::float3 s) {
    return simd_matrix(
        simd_make_float4(s.x, 0.0f, 0.0f, 0.0f),
        simd_make_float4(0.0f, s.y, 0.0f, 0.0f),
        simd_make_float4(0.0f, 0.0f, s.z, 0.0f),
        simd_make_float4(0.0f, 0.0f, 0.0f, 1.0f)
    );
}

// Same handedness/layout as the rotation matrices in Uniforms.cpp
simd::float4x4 rotationMatrixX(float radians) {
    float c = cosf(radians), s = sinf(radians);
    return simd_matrix(
        simd_make_float4(1.0f, 0.0f, 0.0f, 0.0f),
        simd_make_float4(0.0f, c,    s,    0.0f),
        simd_make_float4(0.0f, -s,   c,    0.0f),
        simd_make_float4(0.0f, 0.0f, 0.0f, 1.0f)
    );
}

simd::float4x4 rotationMatrixY(float radians) {
    float c = cosf(radians), s = sinf(radians);
    return simd_matrix(
        simd_make_float4(c,    0.0f, -s,   0.0f),
        simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f),
        simd_make_float4(s,    0.0f, c,    0.0f),
        simd_make_float4(0.0f, 0.0f, 0.0f, 1.0f)
    );
}

simd::float4x4 rotationMatrixZ(float radians) {
    float c = cosf(radians), s = sinf(radians);
    return simd_matrix(
        simd_make_float4(c,    s,    0.0f, 0.0f),
        simd_make_float4(-s,   c,    0.0f, 0.0f),
        simd_make_float4(0.0f, 0.0f, 1.0f, 0.0f),
        simd_make_float4(0.0f, 0.0f, 0.0f, 1.0f)
    );
}

simd::float4x4 translationMatrix(simd::float3 t) {
    return simd_matrix(
        simd_make_float4(1.0f, 0.0f, 0.0f, 0.0f),
        simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f),
        simd_make_float4(0.0f, 0.0f, 1.0f, 0.0f),
        simd_make_float4(t.x,  t.y,  t.z,  1.0f)
    );
}

simd::float4x4 objectRotationMatrix(const SceneObject& obj) {
    return rotationMatrixZ(obj.rotationDegrees[2] * kDegToRad)
         * rotationMatrixY(obj.rotationDegrees[1] * kDegToRad)
         * rotationMatrixX(obj.rotationDegrees[0] * kDegToRad);
}

const char* lightTypeName(LightType type) {
    switch (type) {
        case LightType::Directional: return "directional";
        case LightType::Spot:        return "spot";
        case LightType::Area:        return "area";
        case LightType::Point:       default: return "point";
    }
}

LightType parseLightType(const std::string& name) {
    if (name == "directional") return LightType::Directional;
    if (name == "spot")        return LightType::Spot;
    if (name == "area")        return LightType::Area;
    return LightType::Point;
}

const char* toneMapOperatorName(ToneMapOperator op) {
    switch (op) {
        case ToneMapOperator::Clamp:      return "clamp";
        case ToneMapOperator::Reinhard:   return "reinhard";
        case ToneMapOperator::Uncharted2: return "uncharted2";
        case ToneMapOperator::ACES:       default: return "aces";
    }
}

const char* ambientOcclusionModeName(AmbientOcclusionMode mode) {
    return mode == AmbientOcclusionMode::SSAO ? "ssao" : "gtao";
}

AmbientOcclusionMode parseAmbientOcclusionMode(const std::string& name) {
    return name == "ssao" ? AmbientOcclusionMode::SSAO : AmbientOcclusionMode::GTAO;
}

ToneMapOperator parseToneMapOperator(const std::string& name) {
    if (name == "clamp")      return ToneMapOperator::Clamp;
    if (name == "reinhard")   return ToneMapOperator::Reinhard;
    if (name == "uncharted2") return ToneMapOperator::Uncharted2;
    return ToneMapOperator::ACES;
}

} // namespace

simd::float4x4 objectModelMatrix(const SceneObject& obj) {
    simd::float4x4 rotation = objectRotationMatrix(obj);
    simd::float3 position = simd_make_float3(obj.position[0], obj.position[1], obj.position[2]);
    simd::float3 scale = simd_make_float3(obj.scale[0], obj.scale[1], obj.scale[2]);
    return translationMatrix(position) * rotation * scaleMatrix(scale);
}

simd::float3 objectForward(const SceneObject& obj) {
    simd::float4 f = objectRotationMatrix(obj) * simd_make_float4(0.0f, 0.0f, -1.0f, 0.0f);
    return simd_make_float3(f.x, f.y, f.z);
}

simd::float3 objectRight(const SceneObject& obj) {
    simd::float4 r = objectRotationMatrix(obj) * simd_make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    return simd_make_float3(r.x, r.y, r.z);
}

simd::float3 objectUp(const SceneObject& obj) {
    simd::float4 u = objectRotationMatrix(obj) * simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    return simd_make_float3(u.x, u.y, u.z);
}

// File grammar - one block per object, in order:
//   exposure <v>            (global, not tied to any object - see Scene::exposure)
//   tonemap clamp|reinhard|aces|uncharted2  (global - see Scene::toneMapOperator)
//   vignette <v>            (global - see Scene::vignetteStrength)
//   chromaticaberration <v> (global - see Scene::chromaticAberrationStrength)
//   filmgrain <v>           (global - see Scene::filmGrainStrength)
//   sharpen <v>             (global - see Scene::sharpenStrength)
//   colorgrading <saturation> <contrast>    (global - see Scene::colorGradingSaturation/Contrast)
//   bloom <threshold> <intensity>           (global - see Scene::bloomThreshold/Intensity)
//   dof <focusDistance> <focusRange> <strength>  (global - see Scene::dofFocusDistance/Range/Strength)
//   motionblur <strength>   (global - see Scene::motionBlurStrength)
//   lensflare <strength>    (global - see Scene::lensFlareStrength)
//   ambientocclusion <strength> <radius> [gtao|ssao]   (global - see Scene::ambientOcclusionStrength/
//                            Radius/Mode; the mode is optional, absent = gtao)
//   ssr <strength> <maxDistance> <thickness>   (global - see Scene::ssrStrength/ssrMaxDistance/
//                            ssrThickness)
//   taa <feedback>   (global - see Scene::taaFeedback)
//   camera <x> <y> <z> <yaw> <pitch>   (global - see Scene::cameraPosition/Yaw/Pitch; radians)
//   environment <intensity> <showSky 0|1>   (global - see Scene::environmentIntensity/showSky)
//   environmentmap <name>   (global - .hdr file stem under environment/, see Scene::environment;
//                            absent = kDefaultEnvironment, so old scene files load unchanged)
//   object <name>          (a Cube or Mesh) or  light <name>          (a Light)
//   position <x> <y> <z>
//   rotation <x> <y> <z>    (degrees; Cube/Mesh always, Light only for Directional/Spot/Area)
//   scale <x> <y> <z>       (Cube/Mesh only)
//   material <r> <g> <b> <metallic> <roughness> <ao> <useTextures 0|1>   (Cube/Mesh only - see
//                            Material in Scene.hpp; absent = defaults, so old scene files load unchanged)
//   texture <set name>      (Cube/Mesh only - folder name under texture/, see Material::textureSet;
//                            absent = kDefaultTextureSet, so old scene files load unchanged)
//   meshpath <path>         (Mesh only - presence of this line is what makes an "object" block a
//                            Mesh rather than a Cube, so old scene files without it still load as
//                            Cube unchanged)
//   lighttype point|directional|spot|area   (Light only)
//   color <r> <g> <b>       (Light only)
//   intensity <v>           (Light only)
//   spotangles <inner> <outer>              (Light, Spot only, degrees)
//   areasize <width> <height>               (Light, Area only)
// Lines starting with '#' and blank lines are ignored.
bool saveScene(const Scene& scene, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        fprintf(stderr, "Failed to open %s for writing\n", path.c_str());
        return false;
    }

    out << "# renderer-metal scene file v1\n";
    if (scene.hasCameraPose) {
        out << "camera " << scene.cameraPosition[0] << " " << scene.cameraPosition[1] << " " << scene.cameraPosition[2]
            << " " << scene.cameraYaw << " " << scene.cameraPitch << "\n";
    }
    out << "exposure " << scene.exposure << "\n";
    out << "tonemap " << toneMapOperatorName(scene.toneMapOperator) << "\n";
    out << "vignette " << scene.vignetteStrength << "\n";
    out << "chromaticaberration " << scene.chromaticAberrationStrength << "\n";
    out << "filmgrain " << scene.filmGrainStrength << "\n";
    out << "sharpen " << scene.sharpenStrength << "\n";
    out << "colorgrading " << scene.colorGradingSaturation << " " << scene.colorGradingContrast << "\n";
    out << "bloom " << scene.bloomThreshold << " " << scene.bloomIntensity << "\n";
    out << "dof " << scene.dofFocusDistance << " " << scene.dofFocusRange << " " << scene.dofStrength << "\n";
    out << "motionblur " << scene.motionBlurStrength << "\n";
    out << "lensflare " << scene.lensFlareStrength << "\n";
    out << "ambientocclusion " << scene.ambientOcclusionStrength << " " << scene.ambientOcclusionRadius << " "
        << ambientOcclusionModeName(scene.ambientOcclusionMode) << "\n";
    out << "ssr " << scene.ssrStrength << " " << scene.ssrMaxDistance << " " << scene.ssrThickness << "\n";
    out << "taa " << scene.taaFeedback << "\n";
    out << "environment " << scene.environmentIntensity << " " << (scene.showSky ? 1 : 0) << "\n";
    out << "environmentmap " << scene.environment << "\n";
    for (const auto& obj : scene.objects) {
        out << (obj.type == SceneObjectType::Light ? "light " : "object ") << obj.name << "\n";
        out << "position " << obj.position[0] << " " << obj.position[1] << " " << obj.position[2] << "\n";
        if (obj.type == SceneObjectType::Light) {
            out << "lighttype " << lightTypeName(obj.lightType) << "\n";
            out << "color " << obj.color[0] << " " << obj.color[1] << " " << obj.color[2] << "\n";
            out << "intensity " << obj.intensity << "\n";
            if (obj.lightType != LightType::Point) {
                out << "rotation " << obj.rotationDegrees[0] << " " << obj.rotationDegrees[1] << " " << obj.rotationDegrees[2] << "\n";
            }
            if (obj.lightType == LightType::Spot) {
                out << "spotangles " << obj.spotInnerDegrees << " " << obj.spotOuterDegrees << "\n";
            }
            if (obj.lightType == LightType::Area) {
                out << "areasize " << obj.areaSize[0] << " " << obj.areaSize[1] << "\n";
            }
        } else {
            out << "rotation " << obj.rotationDegrees[0] << " " << obj.rotationDegrees[1] << " " << obj.rotationDegrees[2] << "\n";
            out << "scale " << obj.scale[0] << " " << obj.scale[1] << " " << obj.scale[2] << "\n";
            const Material& m = obj.material;
            out << "material " << m.albedo[0] << " " << m.albedo[1] << " " << m.albedo[2] << " "
                << m.metallic << " " << m.roughness << " " << m.ao << " " << (m.useTextures ? 1 : 0) << "\n";
            out << "texture " << m.textureSet << "\n";
            if (obj.type == SceneObjectType::Mesh) {
                out << "meshpath " << obj.meshPath << "\n";
            }
        }
    }
    return true;
}

bool loadScene(Scene& scene, const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "Failed to open %s for reading\n", path.c_str());
        return false;
    }

    Scene loaded;
    bool haveCurrent = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string keyword;
        ss >> keyword;

        if (keyword == "exposure") {
            ss >> loaded.exposure;
        } else if (keyword == "tonemap") {
            std::string opName;
            ss >> opName;
            loaded.toneMapOperator = parseToneMapOperator(opName);
        } else if (keyword == "vignette") {
            ss >> loaded.vignetteStrength;
        } else if (keyword == "chromaticaberration") {
            ss >> loaded.chromaticAberrationStrength;
        } else if (keyword == "filmgrain") {
            ss >> loaded.filmGrainStrength;
        } else if (keyword == "sharpen") {
            ss >> loaded.sharpenStrength;
        } else if (keyword == "colorgrading") {
            ss >> loaded.colorGradingSaturation >> loaded.colorGradingContrast;
        } else if (keyword == "bloom") {
            ss >> loaded.bloomThreshold >> loaded.bloomIntensity;
        } else if (keyword == "dof") {
            ss >> loaded.dofFocusDistance >> loaded.dofFocusRange >> loaded.dofStrength;
        } else if (keyword == "motionblur") {
            ss >> loaded.motionBlurStrength;
        } else if (keyword == "lensflare") {
            ss >> loaded.lensFlareStrength;
        } else if (keyword == "ambientocclusion") {
            std::string modeName;
            ss >> loaded.ambientOcclusionStrength >> loaded.ambientOcclusionRadius >> modeName;
            loaded.ambientOcclusionMode = parseAmbientOcclusionMode(modeName);
        } else if (keyword == "camera") {
            float x, y, z, yaw, pitch;
            if (ss >> x >> y >> z >> yaw >> pitch) {
                loaded.hasCameraPose = true;
                loaded.cameraPosition[0] = x;
                loaded.cameraPosition[1] = y;
                loaded.cameraPosition[2] = z;
                loaded.cameraYaw = yaw;
                loaded.cameraPitch = pitch;
            }
        } else if (keyword == "ssr") {
            ss >> loaded.ssrStrength >> loaded.ssrMaxDistance >> loaded.ssrThickness;
        } else if (keyword == "taa") {
            ss >> loaded.taaFeedback;
        } else if (keyword == "environment") {
            int showSky = 1;
            ss >> loaded.environmentIntensity >> showSky;
            loaded.showSky = showSky != 0;
        } else if (keyword == "environmentmap") {
            std::getline(ss, loaded.environment);
            size_t start = loaded.environment.find_first_not_of(' ');
            loaded.environment = (start == std::string::npos) ? kDefaultEnvironment : loaded.environment.substr(start);
        } else if (keyword == "object" || keyword == "light") {
            SceneObject obj;
            obj.type = (keyword == "light") ? SceneObjectType::Light : SceneObjectType::Cube;
            std::getline(ss, obj.name);
            size_t start = obj.name.find_first_not_of(' ');
            obj.name = (start == std::string::npos) ? "" : obj.name.substr(start);
            loaded.objects.push_back(obj);
            haveCurrent = true;
        } else if (haveCurrent && keyword == "position") {
            float* p = loaded.objects.back().position;
            ss >> p[0] >> p[1] >> p[2];
        } else if (haveCurrent && keyword == "rotation") {
            float* r = loaded.objects.back().rotationDegrees;
            ss >> r[0] >> r[1] >> r[2];
        } else if (haveCurrent && keyword == "scale") {
            float* s = loaded.objects.back().scale;
            ss >> s[0] >> s[1] >> s[2];
        } else if (haveCurrent && keyword == "material") {
            Material& m = loaded.objects.back().material;
            int useTextures = 1;
            ss >> m.albedo[0] >> m.albedo[1] >> m.albedo[2] >> m.metallic >> m.roughness >> m.ao >> useTextures;
            m.useTextures = useTextures != 0;
        } else if (haveCurrent && keyword == "texture") {
            std::string& set = loaded.objects.back().material.textureSet;
            std::getline(ss, set);
            size_t start = set.find_first_not_of(' ');
            set = (start == std::string::npos) ? kDefaultTextureSet : set.substr(start);
        } else if (haveCurrent && keyword == "color") {
            float* c = loaded.objects.back().color;
            ss >> c[0] >> c[1] >> c[2];
        } else if (haveCurrent && keyword == "intensity") {
            ss >> loaded.objects.back().intensity;
        } else if (haveCurrent && keyword == "lighttype") {
            std::string typeName;
            ss >> typeName;
            loaded.objects.back().lightType = parseLightType(typeName);
        } else if (haveCurrent && keyword == "spotangles") {
            ss >> loaded.objects.back().spotInnerDegrees >> loaded.objects.back().spotOuterDegrees;
        } else if (haveCurrent && keyword == "areasize") {
            float* a = loaded.objects.back().areaSize;
            ss >> a[0] >> a[1];
        } else if (haveCurrent && keyword == "meshpath") {
            SceneObject& obj = loaded.objects.back();
            obj.type = SceneObjectType::Mesh;
            std::getline(ss, obj.meshPath);
            size_t start = obj.meshPath.find_first_not_of(' ');
            obj.meshPath = (start == std::string::npos) ? "" : obj.meshPath.substr(start);
        }
    }

    scene = std::move(loaded);
    return true;
}
