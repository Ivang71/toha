#ifndef TOHA_BOUNDS_GLSL
#define TOHA_BOUNDS_GLSL

#include "world/noise.glsl"

struct NoiseBounds {
    float min;
    float max;
};

const int TOTAL_OCTAVES = 6;

float evaluateOctaves(vec3 p, int octaves) {
    float v = 0.0;
    float a = 0.5;
    float f = 1.0;
    for (int i = 0; i < octaves; ++i) {
        v += a * noise3D(p * f);
        f *= 2.0;
        a *= 0.5;
    }
    return v;
}

NoiseBounds computeBounds(ivec3 cell, int lod) {
    float cellSize = float(1 << lod);
    vec3 center = (vec3(cell) + 0.5) * cellSize;
    int evalOctaves = max(1, TOTAL_OCTAVES - lod);
    float baseValue = evaluateOctaves(center, evalOctaves);
    float maxDeviation = sumAmplitudes(evalOctaves, TOTAL_OCTAVES);
    NoiseBounds b;
    b.min = baseValue - maxDeviation;
    b.max = baseValue + maxDeviation;
    return b;
}

#endif

