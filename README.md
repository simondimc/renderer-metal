# renderer-metal

## Goals

### Basics
- [x] Triangle
- [x] Cube
- [x] Textures
- [x] Lighting
- [x] Interactive Camera

### Engine
- [x] Frame-Resource Synchronization
- [x] Loading Assets
- [x] Scene Editor
- [x] Offscreen Rendering & Post-Processing
- Entity Component System (ECS)
- Job System / Task-Based Parallelism
- Hot Reloading (Shaders/Assets)
- Multithreaded Command Buffer Recording

### Rendering Pipeline
- Deferred Shading
- Clustered/Tiled Rendering
- Temporal Anti-Aliasing (TAA)
- [x] HDR & Tone Mapping

### Post-Processing
- [x] Bloom
- [x] Vignette
- [x] Chromatic Aberration
- [x] Film Grain
- [x] Color Grading (Saturation/Contrast - not LUT-based yet)
- [x] Depth of Field
- [x] Motion Blur
- [x] Sharpening
- [x] Lens Flare
- [x] Outline / Edge Detection

### Materials & Shading
- [x] Linear Color Space & Gamma Correction
- [x] Physically Based Rendering (PBR)
- [x] Normal Mapping
- Bump/Parallax Mapping
- [x] Image-Based Lighting (IBL)
- Anisotropic Shading
- [x] Transmission & Volume (glass)
- [x] Emissive Materials
- [x] Alpha Modes (Mask/Blend)
- [x] Separate Ambient Occlusion Texture

### Global Illumination
- Ambient Occlusion (SSAO/HBAO/GTAO)
- Screen-Space Reflections (SSR)
- Precomputed/Baked GI (Lightmaps, Light Probes)
- Voxel Cone Tracing / ReSTIR

### Animation & Characters
- Skeletal Animation & GPU Skinning
- Morph Targets / Blend Shapes
- Inverse Kinematics

### Realism
- Heightmap-Based Terrain & Tessellation
- Instanced Rendering

### Optimization
- Spatial Partitioning
- Frustum Culling
- Occlusion Culling (Hi-Z / GPU-Driven)
- Level of Detail (LOD)
- GPU-Driven Rendering / Indirect Draw Calls

### Advanced
- [x] Shadow Mapping
- Volumetric Atmosphere & Sky
- Water
- Physics Engine
- Cloth Simulation & Rendering
- Subsurface Scattering (Human Skin)
- Ray Tracing
- Path Tracing
