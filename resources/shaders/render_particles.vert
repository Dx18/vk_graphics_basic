#version 430
#extension GL_GOOGLE_include_directive : require

#include "particle.h"

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
} params;

layout(std430, binding = 0) buffer Particles
{
    Particle particles[];
};

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 uv;

vec2 getVertexLocalPosition()
{
    return vec2[](
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
    )[gl_VertexIndex];
}

void main()
{
    Particle particle = particles[gl_InstanceIndex];

    vec3 position = particle.positionAndMass.xyz;
    float size = particle.velocityAndSize.w;
    if (particle.colorAndRemainingLifetime.w <= 0.0)
    {
        size = 0.0;
    }

    vec2 localPosition = getVertexLocalPosition();

    vec2 uvBegin = particle.uvBeginAndUVEnd.xy;
    vec2 uvEnd = particle.uvBeginAndUVEnd.zw;

    vec2 uvSize = uvEnd - uvBegin;

    if (uvSize.x >= uvSize.y)
    {
        localPosition.y *= uvSize.y / uvSize.x;
    }
    else
    {
        localPosition.x *= uvSize.x / uvSize.y;
    }

    vec4 cameraPositionH = inverse(params.mProjView) * vec4(0.0, 0.0, 0.0, 1.0);
    vec3 cameraPosition = cameraPositionH.xyz / cameraPositionH.w;
    
    vec3 dir = normalize(position - cameraPosition);
    vec3 dirX = cross(dir, vec3(0.0, 1.0, 0.0));
    vec3 dirY = cross(dirX, dir);
    
    gl_Position = params.mProjView * params.mModel * vec4(position + (dirX * localPosition.x + dirY * localPosition.y) * size, 1.0);

    fragColor = particle.colorAndRemainingLifetime.rgb;
    uv = uvBegin + (localPosition + 1.0) / 2.0 * (uvEnd - uvBegin);
}