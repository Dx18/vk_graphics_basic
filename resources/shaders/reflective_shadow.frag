#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"

layout(location = 0) out vec4 out_fragColor;

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
    vec4 objectColor;
} params;

layout (location = 0 ) in VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec3 wTangent;
  vec2 texCoord;
} surf;

layout(binding = 0, set = 0) uniform AppData
{
  UniformParams Params;
};

layout (binding = 1) uniform sampler2D shadowMap;
layout (binding = 2) uniform sampler2D shadowMapNormal;
layout (binding = 3) uniform sampler2D shadowMapFlux;

layout (binding = 4, std430) buffer SamplePositions {
  vec2 samplePositions[800];
};

struct RSMPixelData
{
  vec3 position;
  vec3 normal;
  vec3 flux;
};

RSMPixelData getRSMPixelData(vec2 coord)
{
  vec4 positionH = inverse(Params.lightMatrix) * vec4(coord * 2.0 - 1.0, textureLod(shadowMap, coord, 0).r, 1.0);
  vec3 position = positionH.xyz / positionH.w;

  vec3 normal = textureLod(shadowMapNormal, coord, 0).rgb;
  vec3 flux = textureLod(shadowMapFlux, coord, 0).rgb;

  return RSMPixelData(position, normal, flux);
}

void main()
{
  const vec4 posLightClipSpace = Params.lightMatrix*vec4(surf.wPos, 1.0f); // 
  const vec3 posLightSpaceNDC  = posLightClipSpace.xyz/posLightClipSpace.w;    // for orto matrix, we don't need perspective division, you can remove it if you want; this is general case;
  const vec2 shadowTexCoord    = posLightSpaceNDC.xy*0.5f + vec2(0.5f, 0.5f);  // just shift coords from [-1,1] to [0,1]               
    
  const bool  outOfView = (shadowTexCoord.x < 0.0001f || shadowTexCoord.x > 0.9999f || shadowTexCoord.y < 0.0091f || shadowTexCoord.y > 0.9999f);
  const float shadow    = ((posLightSpaceNDC.z < textureLod(shadowMap, shadowTexCoord, 0).x + 0.001f) || outOfView) ? 1.0f : 0.0f;

  vec3 sampleSum = vec3(0.0);

  RSMPixelData pixelData = getRSMPixelData(shadowTexCoord);

  for (int i = 0; i < 800; ++i)
  {
    vec2 offset = samplePositions[i];

    vec2 currShadowTexCoord = shadowTexCoord + offset;

    RSMPixelData currPixelData = getRSMPixelData(currShadowTexCoord);

    vec3 positionOffset = pixelData.position - currPixelData.position;
    sampleSum += currPixelData.flux * dot(offset, offset) *
        max(0.0, dot(currPixelData.normal, positionOffset)) *
        max(0.0, dot(pixelData.normal, -positionOffset)) /
        pow(dot(positionOffset, positionOffset), 2.0);
  }
  
  vec3 lightDir   = normalize(Params.lightPos - surf.wPos);
  vec3 color = (max(dot(surf.wNorm, lightDir), 0.0) * shadow + vec3(0.1)) * params.objectColor.rgb;
  out_fragColor = vec4(color + 0.4 * sampleSum, 1.0);
}
