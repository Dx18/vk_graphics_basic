#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "unpack_attributes.h"

layout (location = 0) in vec4 vPosNorm;
layout (location = 1) in vec4 vTexCoordAndTang;

layout (location = 0) out VS_OUT
{
  vec4 vPosNorm;
  vec4 vTexCoordAndTang;
} vOut;

out gl_PerVertex { vec4 gl_Position; };

void main(void)
{
    vOut.vPosNorm = vPosNorm;
    vOut.vTexCoordAndTang = vTexCoordAndTang;
}
