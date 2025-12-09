#ifndef TOHA_DDA_GLSL
#define TOHA_DDA_GLSL

vec3 ddaExit(vec3 pos, vec3 rd, ivec3 cell, float cellSize) {
    vec3 cellMin = vec3(cell) * cellSize;
    vec3 cellMax = cellMin + cellSize;
    vec3 tMax;
    tMax.x = (rd.x > 0.0 ? cellMax.x - pos.x : pos.x - cellMin.x) / max(abs(rd.x), 1e-6);
    tMax.y = (rd.y > 0.0 ? cellMax.y - pos.y : pos.y - cellMin.y) / max(abs(rd.y), 1e-6);
    tMax.z = (rd.z > 0.0 ? cellMax.z - pos.z : pos.z - cellMin.z) / max(abs(rd.z), 1e-6);
    float tExit = min(min(tMax.x, tMax.y), tMax.z);
    return pos + rd * (tExit + 1e-3);
}

#endif

