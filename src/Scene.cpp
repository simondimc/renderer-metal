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
//   object <name>          (a Cube or Mesh) or  light <name>          (a Light)
//   position <x> <y> <z>
//   rotation <x> <y> <z>    (degrees; Cube/Mesh always, Light only for Directional/Spot/Area)
//   scale <x> <y> <z>       (Cube/Mesh only)
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

        if (keyword == "object" || keyword == "light") {
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
