#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "unpack_attributes.h"

layout (location = 0) in vec4 vPosNorm;
layout (location = 1) in vec4 vTexCoordAndTang;

layout (location = 0) out VS_OUT
{
    vec3 wPos;
    vec3 wNorm;
    vec3 wTangent;
    vec2 texCoord;
} vOut;

layout (push_constant) uniform params_t
{
    mat4 model;
} params;

layout (std140, binding = 0) uniform viewParams_t
{
   mat4 view;
   mat4 projection;
   mat4 viewInv;
   mat4 projectionInv;
   mat4 viewProjection;
   mat4 viewProjectionInv;
   vec4 cameraPosition;
   uvec2 targetSize;
} viewParams;

void main()
{
    const vec4 wNorm = vec4(DecodeNormal(floatBitsToInt(vPosNorm.w)),         0.0f);
    const vec4 wTang = vec4(DecodeNormal(floatBitsToInt(vTexCoordAndTang.z)), 0.0f);

    vOut.wPos     = (params.model * vec4(vPosNorm.xyz, 1.0f)).xyz;
    vOut.wNorm    = normalize(mat3(transpose(inverse(params.model))) * wNorm.xyz);
    vOut.wTangent = normalize(mat3(transpose(inverse(params.model))) * wTang.xyz);
    vOut.texCoord = vTexCoordAndTang.xy;

    gl_Position   = viewParams.viewProjection * vec4(vOut.wPos, 1.0);
}