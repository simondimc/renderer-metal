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
- [x] Hot Reloading (Shaders)
- Hot Reloading (Assets)
- Multithreaded Command Buffer Recording

### Rendering Pipeline
- [x] Clustered Forward Lighting
- Visibility Buffer
- [x] Temporal Anti-Aliasing (TAA)
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
- [x] Ambient Occlusion
- [x] Screen-Space Reflections (SSR)
- Precomputed/Baked GI (Lightmaps, Light Probes)
- Voxel Cone Tracing / ReSTIR

### Animation & Characters
- Skeletal Animation & GPU Skinning
- Morph Targets / Blend Shapes
- Inverse Kinematics

### Realism
- Heightmap-Based Terrain & Tessellation

### Optimization
- [x] Frustum Culling
- [x] Instanced Rendering
- GPU-Driven Rendering / Indirect Draw Calls
- Occlusion Culling (Hi-Z)
- Level of Detail (LOD)
- Meshlets / Mesh Shaders
- Spatial Partitioning

### Advanced
- [x] Shadow Mapping
- Volumetric Atmosphere & Sky
- Water
- Physics Engine
- Cloth Simulation & Rendering
- Subsurface Scattering (Human Skin)
- Ray Tracing
- Path Tracing
