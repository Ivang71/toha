# Voxel Engine Technical Specification

## Overview

GPU compute-based voxel engine with 50km render distance. World defined by a **cellular function** — a mathematical function that classifies each voxel cell as solid/empty. No disk streaming, purely procedural.

---

## Core Concept: The Cellular Function

The entire world is defined by one function:

```
cell(ivec3 position) → {EMPTY, SOLID, material}
```

This function is:
- **Deterministic** — same input always produces same output
- **Stateless** — no external data needed
- **Local** — only depends on the cell position

Everything else (rendering, LOD, bounds) builds on top of this.

---

## World Generation

### Base Terrain (Heightfield + 3D Noise)

```
For cell at (x, y, z):

1. Compute surface height at (x, z):
   height = baseHeight 
          + fbm2D(x, z, scale=0.001, octaves=5) * 200    // Mountains
          + fbm2D(x, z, scale=0.01, octaves=3) * 20       // Hills
          + fbm2D(x, z, scale=0.1, octaves=2) * 2         // Bumps

2. Classify:
   if (y > height) → EMPTY (air)
   if (y > height - 1) → SOLID (grass)
   if (y > height - 4) → SOLID (dirt)
   else → SOLID (stone)
```

### Caves (3D Noise Carving)

```
For underground cells:

caveNoise = fbm3D(x, y, z, scale=0.02, octaves=3)

if (caveNoise > 0.6) → EMPTY (cave)
```

### Ores (3D Noise + Depth Rules)

```
For stone cells:

depth = surfaceHeight - y

coalNoise = fbm3D(x, y, z, scale=0.1, octaves=2, seed=100)
if (coalNoise > 0.75 && depth > 5) → COAL

ironNoise = fbm3D(x, y, z, scale=0.08, octaves=2, seed=200)
if (ironNoise > 0.8 && depth > 20 && depth < 80) → IRON

// ... similar for gold, diamond, etc.
```

### Water

```
WATER_LEVEL = 64

For EMPTY cells:
if (y < WATER_LEVEL) → WATER
```

### Complete Cellular Function

```
Material cellFunction(ivec3 pos) {
    float surfaceH = terrainHeight(pos.xz)
    
    // Above surface
    if (pos.y > surfaceH) {
        if (pos.y < WATER_LEVEL) return WATER
        return AIR
    }
    
    // Cave check
    if (caveNoise(pos) > CAVE_THRESHOLD) return AIR
    
    // Material by depth
    float depth = surfaceH - pos.y
    
    if (depth < 1) return GRASS
    if (depth < 4) return DIRT
    
    // Ore checks (stone layer)
    if (isCoal(pos, depth)) return COAL
    if (isIron(pos, depth)) return IRON
    if (isGold(pos, depth)) return GOLD
    if (isDiamond(pos, depth)) return DIAMOND
    
    return STONE
}
```

---

## Rendering: Hierarchical Ray Marching

### The Problem

50km = 50,000 cells. Can't check every cell along every ray.

### The Solution: Hierarchical Bounds

At coarse LOD levels, we can **bound** the noise without evaluating it fully:

```
LOD 0: 1m cells    (actual voxels)
LOD 1: 2m cells
LOD 2: 4m cells
LOD 3: 8m cells
LOD 4: 16m cells
LOD 5: 32m cells
LOD 6: 64m cells
LOD 7: 128m cells
```

For multi-octave noise `N = Σ amplitude[i] * noise(p * freq[i])`:
- At coarse LOD, only evaluate low-frequency octaves
- Bound contribution from skipped high-frequency octaves

### Cell Classification

For each cell at given LOD:

```
EMPTY:     max_possible_density < threshold  → skip entire cell
SOLID:     min_possible_density > threshold  → find exit or descend
UNCERTAIN: bounds straddle threshold         → descend to finer LOD
```

### Ray March Algorithm

```
trace(ray_origin, ray_direction):
    lod = MAX_LOD
    pos = ray_origin
    
    for step in 0..MAX_STEPS:
        cell = floor(pos / cell_size(lod))
        bounds = computeBounds(cell, lod)
        
        if bounds.max < threshold:
            # EMPTY - skip cell with DDA
            pos = exitCell(pos, dir, cell, lod)
            # Try to ascend to coarser LOD
            if canAscend(pos, lod): lod++
            
        else if bounds.min > threshold:
            # SOLID
            if lod == 0:
                return HIT(pos, cell)
            lod--  # Descend to find exact voxel
            
        else:
            # UNCERTAIN
            if lod == 0:
                # Evaluate actual cell function
                if cellFunction(cell).isSolid:
                    return HIT(pos, cell)
                pos = exitCell(pos, dir, cell, 0)
            else:
                lod--  # Descend
    
    return MISS
```

### DDA (Digital Differential Analyzer)

Exact cell-to-cell stepping on regular grid:

```
exitCell(pos, dir, cell, lod):
    cell_size = 1 << lod
    cell_min = cell * cell_size
    cell_max = cell_min + cell_size
    
    # Time to exit on each axis
    t_exit.x = (dir.x > 0 ? cell_max.x - pos.x : pos.x - cell_min.x) / |dir.x|
    t_exit.y = ...
    t_exit.z = ...
    
    t = min(t_exit.x, t_exit.y, t_exit.z)
    return pos + dir * (t + epsilon)
```

### Noise Bounds Computation

For FBM noise with known amplitudes:

```
computeBounds(cell, lod):
    center = (cell + 0.5) * cell_size(lod)
    
    # Evaluate only octaves that matter at this scale
    eval_octaves = TOTAL_OCTAVES - lod
    base_value = evaluateFBM(center, eval_octaves)
    
    # Sum amplitudes of skipped octaves
    max_deviation = sum(amplitude[i] for i in eval_octaves..TOTAL_OCTAVES)
    
    return Bounds(base_value - max_deviation, base_value + max_deviation)
```

---

## Terrain Height Bounds (Special Case)

For heightfield terrain, we can bound more efficiently:

```
heightBounds(cell_xz, lod):
    center = (cell_xz + 0.5) * cell_size(lod)
    
    h = terrainHeight(center)  # Evaluate at coarse scale
    max_dev = heightMaxDeviation(lod)
    
    return (h - max_dev, h + max_dev)
```

Cell classification for terrain:

```
cell_y_min = cell.y * cell_size
cell_y_max = cell_y_min + cell_size
height_bounds = heightBounds(cell.xz, lod)

if cell_y_min > height_bounds.max → EMPTY (above terrain)
if cell_y_max < height_bounds.min → SOLID (below terrain) 
else → UNCERTAIN
```

---

## Performance

### Step Counts

| Distance | LOD | Cell Size | Max Cells |
|----------|-----|-----------|-----------|
| 0-128m   | 0-1 | 1-2m      | ~100      |
| 128m-1km | 2-3 | 4-8m      | ~100      |
| 1-10km   | 4-5 | 16-32m    | ~300      |
| 10-50km  | 6-7 | 64-128m   | ~400      |

**Typical total: 40-100 steps per ray**

### Optimizations

1. **Temporal reprojection** — reuse last frame's hits as starting estimates
2. **Early ray termination** — max distance, behind solid
3. **Shared memory caching** — cache bounds for adjacent rays
4. **Variable rate shading** — fewer rays for distant pixels

---

## Vegetation (Trees, Grass)

### Tree Placement

Trees are placed via cellular function at larger scale:

```
hasTree(cell_xz):
    tree_cell = floor(cell_xz / TREE_SPACING)
    hash = hash2D(tree_cell)
    
    if hash.x > forestDensity(cell_xz): return false
    if terrainHeight(cell_xz) < WATER_LEVEL: return false
    if terrainSlope(cell_xz) > 0.5: return false
    
    return true
```

### Tree Rendering by Distance

| Distance | Representation |
|----------|----------------|
| 0-100m   | Voxel tree shape (evaluated per cell) |
| 100-2km  | Simplified canopy volume |
| 2km+     | Color tint on terrain |

### Forest Density Function

```
forestDensity(pos_xz):
    base = fbm2D(pos_xz, scale=0.0005, octaves=3)
    
    # Reduce on slopes
    base *= (1 - terrainSlope(pos_xz))
    
    # Reduce at high altitude
    base *= smoothstep(TREE_LINE + 200, TREE_LINE, terrainHeight(pos_xz))
    
    return clamp(base, 0, 1)
```

---

## Water

### Detection

```
isWater(pos):
    return pos.y < WATER_LEVEL && terrainHeight(pos.xz) < WATER_LEVEL
```

### Rendering

1. Check water plane intersection before terrain march
2. If hit water: apply reflection, refraction, waves
3. Waves via animated 2D noise on water surface normal

---

## Materials

```
enum Material {
    AIR, WATER,
    GRASS, DIRT, STONE,
    COAL, IRON, GOLD, DIAMOND,
    WOOD, LEAVES,
    SAND, GRAVEL
}
```

Each material has:
- Base color
- Roughness (for specular)
- Emission (for glowing ores)

---

## Lighting

### Ambient Occlusion

For voxels, AO is simple — sample neighboring cells:

```
ao(hit_pos, normal):
    occlusion = 0
    for offset in AO_SAMPLE_OFFSETS:
        sample_pos = hit_pos + normal + offset
        if cellFunction(sample_pos).isSolid:
            occlusion += 1
    return 1 - occlusion / NUM_SAMPLES
```

### Shadows

Cast shadow ray toward sun:
- Use same hierarchical march (cheaper with early termination)
- Binary result: lit or shadowed

### Sky Light

Ambient light from sky hemisphere:
- Stronger when normal points up
- Tinted blue

---

## Atmosphere

Critical for 50km scale perception:

```
applyAtmosphere(color, distance, ray_dir):
    # Rayleigh scattering (blue haze)
    rayleigh = 1 - exp(-distance * 0.00003)
    color = mix(color, SKY_BLUE, rayleigh * 0.7)
    
    # Mie scattering (white haze)
    mie = 1 - exp(-distance * 0.00008)
    color = mix(color, HAZE_WHITE, mie * 0.3)
    
    return color
```

---

## Shader Structure

```
shaders/
├── noise.glsl           # FBM, simplex, hash functions
├── terrain.glsl         # Height function, slope, biomes
├── cell.glsl            # The cellular function
├── bounds.glsl          # Noise bounds computation
├── march.glsl           # Hierarchical ray march + DDA
├── material.glsl        # Material colors, properties
├── lighting.glsl        # AO, shadows, sky
├── atmosphere.glsl      # Distance fog, scattering
├── water.glsl           # Water rendering
└── main.comp            # Entry point, ray generation
```

---

## Constants

```
// World
WATER_LEVEL = 64
TREE_LINE = 200
SNOW_LINE = 280

// Noise
TERRAIN_OCTAVES = 6
CAVE_OCTAVES = 4
CAVE_THRESHOLD = 0.6

// Rendering
MAX_STEPS = 200
MAX_DISTANCE = 50000
MAX_LOD = 7

// Atmosphere
RAYLEIGH_DENSITY = 0.00003
MIE_DENSITY = 0.00008
```

---

## Summary

1. **World** = cellular function: `ivec3 → Material`
2. **Rendering** = hierarchical ray march with noise bounds
3. **LOD** = implicit via octave evaluation depth
4. **No storage** = pure math, infinite world
5. **Performance** = 40-100 steps per ray for 50km

