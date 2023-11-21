#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "light.h"

layout (location = 0) in vec3 vPos;

layout (location = 0) out VS_OUT
{
    LightInfo light;
} vOut;

layout (std430, binding = 2) buffer lightData
{
  LightInfo lights[];
};

layout (std140, binding = 3) uniform viewParams_t
{
   mat4 view;
   mat4 projection;
   mat4 viewInv;
   mat4 projectionInv;
   mat4 viewProjection;
   vec4 cameraPosition;
   uvec2 targetSize;
} viewParams;

void main()
{
    LightInfo light = lights[gl_InstanceIndex];

    float scale = light.parameters.w;

    gl_Position = viewParams.viewProjection * vec4(
        vPos * scale + light.position.xyz, 1.0
    );

    vOut.light = light;
}