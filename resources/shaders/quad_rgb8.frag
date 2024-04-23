#version 450

layout(location = 0) out vec4 color;

layout (binding = 0, rgba8ui) readonly uniform uimage2D colorTex;

layout (location = 0 ) in VS_OUT
{
  vec2 texCoord;
} surf;

void main()
{
  ivec2 size = imageSize(colorTex);
  ivec2 pixelLocation = clamp(ivec2((vec2(0.0, 1.0) + vec2(1.0, -1.0) * surf.texCoord) * vec2(size)), ivec2(0), size - 1);

  color = vec4(vec3(imageLoad(colorTex, pixelLocation).rgb) / 255.0, 1.0);
}