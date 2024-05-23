#version 430

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec4 out_fragColor;

layout(binding = 1, rgba8ui) readonly uniform uimage2D particleAtlas;

void main()
{
    ivec2 particleAtlasSize = ivec2(imageSize(particleAtlas));
    ivec2 pixelLocation = ivec2(uv * particleAtlasSize);

    uvec4 pixel = imageLoad(particleAtlas, pixelLocation);

    if (pixel.a < 255)
    {
        discard;
    }

    out_fragColor = vec4(pixel) / 255.0 * vec4(fragColor, 1.0);
}