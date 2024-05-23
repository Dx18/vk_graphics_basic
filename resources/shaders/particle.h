#ifndef VK_GRAPHICS_BASIC_PARTICLE_H
#define VK_GRAPHICS_BASIC_PARTICLE_H

struct Particle
{
    vec4 positionAndMass;
    vec4 velocityAndSize;
    vec4 uvBeginAndUVEnd;
    vec4 colorAndRemainingLifetime;
};

Particle constructParticle(vec3 position, vec3 velocity, vec2 uvBegin, vec2 uvEnd,
    vec3 color, float mass, float size, float remainingLifetime)
{
    return Particle(vec4(position, mass), vec4(velocity, size),
        vec4(uvBegin, uvEnd), vec4(color, remainingLifetime));
}

#endif