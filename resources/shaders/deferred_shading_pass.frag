#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "light.h"

layout (location = 0) in VS_OUT
{
    LightInfo light;
} fIn;

layout (location = 0) out vec4 out_fragColor;

layout (binding = 0) uniform sampler2D albedoDepthMap;
layout (binding = 1) uniform sampler2D normalMap;

layout (std140, binding = 3) uniform viewParams_t
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
    vec2 uv = gl_FragCoord.xy / vec2(viewParams.targetSize);

    vec4 albedoDepth = texture(albedoDepthMap, uv);
    vec3 albedo = albedoDepth.rgb;
    vec4 positionH = viewParams.viewProjectionInv *
        vec4(uv * 2.0 - 1.0, albedoDepth.a, 1.0);
    vec3 position = positionH.xyz / positionH.w;

    vec3 normal = texture(normalMap, uv).xyz;

    vec3 lightVec = fIn.light.position.xyz - position;
    vec3 lightDir = normalize(lightVec);

    float effect;
    {
        float constant = fIn.light.parameters.x;
        float linear = fIn.light.parameters.y;
        float quadratic = fIn.light.parameters.z;

        float distSq = dot(lightVec, lightVec);
        float dist = sqrt(distSq);

        effect = 1.0 / (constant + linear * dist + quadratic * distSq);
    }

    float diffuse = max(0.0, dot(normal, lightDir));

    vec3 viewDir = normalize(viewParams.cameraPosition.xyz - position);

    float specular = pow(max(0.0, dot(reflect(-lightDir, normal), viewDir)), 4.0);

    out_fragColor = vec4(
        effect * (diffuse + specular) * albedo * fIn.light.color.rgb, 1.0
    );
}