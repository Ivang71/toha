#ifndef TOHA_ATMOSPHERE_GLSL
#define TOHA_ATMOSPHERE_GLSL

vec3 applyAtmosphere(vec3 color, float distance, vec3 rd, vec3 sunDir) {
    float rayleigh = 1.0 - exp(-distance * RAYLEIGH_DENSITY);
    vec3 rayleighColor = vec3(0.5, 0.7, 1.0);
    float mie = 1.0 - exp(-distance * MIE_DENSITY);
    float sunAngle = max(0.0, dot(rd, sunDir));
    float g = 0.6;
    float denom = 1.0 + g * g - 2.0 * g * sunAngle;
    float miePhase = (1.0 - g * g) / (4.0 * 3.14159 * denom * sqrt(denom));
    vec3 mieColor = vec3(0.8, 0.85, 0.9) * (1.0 + miePhase * 0.5);
    color = mix(color, rayleighColor, rayleigh * 0.6);
    color = mix(color, mieColor, mie * 0.4);
    return color;
}

#endif

