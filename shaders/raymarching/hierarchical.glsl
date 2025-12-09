#ifndef TOHA_HIERARCHICAL_GLSL
#define TOHA_HIERARCHICAL_GLSL

#include "raymarching/dda.glsl"
#include "raymarching/bounds.glsl"
#include "world/world.glsl"

struct Hit {
    int hit;
    vec3 pos;
    ivec3 cell;
    float distance;
};

Hit Miss() {
    Hit h;
    h.hit = 0;
    h.pos = vec3(0.0);
    h.cell = ivec3(0);
    h.distance = MAX_DISTANCE;
    return h;
}

bool canAscendLOD(vec3 pos, int lod) {
    if (lod >= MAX_LOD) return false;
    float cellSize = float(1 << (lod + 1));
    vec3 fractPart = fract(pos / cellSize);
    return all(greaterThan(fractPart, vec3(0.2))) && all(lessThan(fractPart, vec3(0.8)));
}

Hit hierarchicalMarch(vec3 ro, vec3 rd) {
    int lod = MAX_LOD;
    vec3 pos = ro;
    for (int step = 0; step < MAX_STEPS; ++step) {
        float cellSize = float(1 << lod);
        ivec3 cell = ivec3(floor(pos / cellSize));
        NoiseBounds bounds = computeBounds(cell, lod);
        if (bounds.max < BLOCK_THRESHOLD) {
            pos = ddaExit(pos, rd, cell, cellSize);
            if (canAscendLOD(pos, lod)) lod++;
        } else if (bounds.min > BLOCK_THRESHOLD) {
            if (lod == 0) {
                Hit h;
                h.hit = 1;
                h.pos = pos;
                h.cell = cell;
                h.distance = length(pos - ro);
                return h;
            }
            lod--;
        } else {
            if (lod == 0) {
                float d = worldDensity(pos);
                if (d > BLOCK_THRESHOLD) {
                    Hit h;
                    h.hit = 1;
                    h.pos = pos;
                    h.cell = cell;
                    h.distance = length(pos - ro);
                    return h;
                }
                pos = ddaExit(pos, rd, cell, 1.0);
            } else {
                lod--;
            }
        }
        if (length(pos - ro) > MAX_DISTANCE) break;
    }
    return Miss();
}

#endif

