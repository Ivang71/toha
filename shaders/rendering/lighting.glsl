#ifndef TOHA_LIGHTING_GLSL
#define TOHA_LIGHTING_GLSL

vec3 lighting(vec3 p, vec3 n, vec3 camPos, int mat) {
    vec3 lightDir = normalize(vec3(0.45, 0.8, 0.25));
    float diff = clamp(dot(n, lightDir), 0.0, 1.0);
    vec3 base = materialColor(mat);
    float grime = clamp(fbm(p * 0.5, 2) * 0.5 + 0.5, 0.4, 1.0);
    vec3 view = normalize(camPos - p);
    vec3 h = normalize(view + lightDir);
    float spec = pow(clamp(dot(n, h), 0.0, 1.0), 32.0) * 0.1;
    return base * (0.2 + 0.8 * diff) * grime + spec;
}

#endif

