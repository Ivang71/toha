#ifndef TOHA_MATERIALS_GLSL
#define TOHA_MATERIALS_GLSL

#include "world/noise.glsl"

const int MAT_METAL_LIGHT = 0;
const int MAT_METAL_DARK = 1;
const int MAT_RUST = 2;
const int MAT_GRIME = 3;
const int MAT_PIPE = 4;
const int MAT_GRATING = 5;

int getMaterial(vec3 p, ivec3 cell) {
    float rust = fbm(p * 0.02, 3);
    float grime = fbm(p * 0.1 + 100.0, 2);
    if (rust > 0.7) return MAT_RUST;
    if (grime > 0.65) return MAT_GRIME;
    float r = hash11(float(cell.x * 12 + cell.y * 31 + cell.z * 7));
    if (r > 0.8) return MAT_GRATING;
    if (r > 0.6) return MAT_PIPE;
    return r > 0.35 ? MAT_METAL_DARK : MAT_METAL_LIGHT;
}

vec3 materialColor(int mat) {
    if (mat == MAT_RUST) return vec3(0.52, 0.25, 0.12);
    if (mat == MAT_GRIME) return vec3(0.15, 0.18, 0.2);
    if (mat == MAT_PIPE) return vec3(0.35, 0.4, 0.45);
    if (mat == MAT_GRATING) return vec3(0.25, 0.28, 0.3);
    if (mat == MAT_METAL_DARK) return vec3(0.18, 0.2, 0.22);
    return vec3(0.55, 0.6, 0.65);
}

#endif

