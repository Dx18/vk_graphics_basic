#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"
#include "unpack_attributes.h"


layout(location = 0) in vec4 vPosNorm;
layout(location = 1) in vec4 vTexCoordAndTang;

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
} params;


layout (location = 0 ) out VS_OUT
{
    vec3 wPos;
    vec3 wNorm;
    vec3 wTangent;
    vec2 texCoord;

} vOut;

layout(binding = 0, set = 0) uniform AppData
{
    UniformParams Params;
};

vec3 calculate_displacement(vec3 position) {
    return position + vec3(0.0, 0.0, sin(position.x * 30.0) * 0.05) * sin(Params.time);
}

mat3 calculate_displacement_jacobi_matrix(vec3 position) {
    float delta = 0.00001;

    vec3 x1 = calculate_displacement(position - vec3(delta, 0.0, 0.0));
    vec3 x2 = calculate_displacement(position + vec3(delta, 0.0, 0.0));
    vec3 y1 = calculate_displacement(position - vec3(0.0, delta, 0.0));
    vec3 y2 = calculate_displacement(position + vec3(0.0, delta, 0.0));
    vec3 z1 = calculate_displacement(position - vec3(0.0, 0.0, delta));
    vec3 z2 = calculate_displacement(position + vec3(0.0, 0.0, delta));

    return mat3(
        (x2 - x1) / (2.0 * delta),
        (y2 - y1) / (2.0 * delta),
        (z2 - z1) / (2.0 * delta)
    );
}

out gl_PerVertex { vec4 gl_Position; };
void main(void)
{
    vec3 positionInit = (params.mModel * vec4(vPosNorm.xyz, 1.0f)).xyz;
    vec3 normalInit = DecodeNormal(floatBitsToInt(vPosNorm.w));
    normalInit = normalize(mat3(transpose(inverse(params.mModel))) * normalInit);
    vec3 tangentInit = DecodeNormal(floatBitsToInt(vTexCoordAndTang.z));
    tangentInit = normalize(mat3(transpose(inverse(params.mModel))) * tangentInit);
    vec3 bitangentInit = cross(normalInit, tangentInit);

    vec3 position = calculate_displacement(positionInit);

    mat3 positionJ = calculate_displacement_jacobi_matrix(positionInit);

    vec3 tangent = normalize(positionJ * tangentInit);
    vec3 bitangent = normalize(positionJ * bitangentInit);
    vec3 normal = normalize(cross(tangent, bitangent));

    vOut.wPos     = position;
    vOut.wNorm    = normal;
    vOut.wTangent = tangent;
    vOut.texCoord = vTexCoordAndTang.xy;

    gl_Position   = params.mProjView * vec4(position, 1.0);
}
