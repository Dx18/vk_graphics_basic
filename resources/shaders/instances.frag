#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

layout(location = 0) out vec4 out_fragColor;

layout (location = 0 ) in VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec3 wTangent;
  vec2 texCoord;
} surf;

void main()
{
  vec3 lightDirection = normalize(vec3(0.3, -0.4, 0.8));
  vec3 normal = normalize(surf.wNorm);

  float light = 0.2 + 0.8 * max(0.0, dot(lightDirection, normal));

  out_fragColor = vec4(vec3(0.4, 0.5, 0.6) * light, 1.0f);
}