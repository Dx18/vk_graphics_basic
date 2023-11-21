#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

layout (location = 0) out vec4 out_albedoDepth;
layout (location = 1) out vec4 out_normal;

layout (location = 0) in VS_OUT
{
    vec3 wPos;
    vec3 wNorm;
    vec3 wTangent;
    vec2 texCoord;
} fIn;

void main()
{
    out_albedoDepth = vec4(1.0, 1.0, 1.0, gl_FragCoord.z);
    out_normal = vec4(normalize(fIn.wNorm), 0.0);
}