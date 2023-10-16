#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 color;

layout (binding = 0) uniform sampler2D colorTex;

layout (location = 0 ) in VS_OUT
{
  vec2 texCoord;
} surf;

const int windowSize = 3;

void sortColors(inout vec4 colors[windowSize * windowSize], int component)
{
  for (int i = 1; i < windowSize * windowSize; ++i)
  {
    vec4 color = colors[i];

    int j = i;
    while (j > 0 && color[component] < colors[j - 1][component])
    {
      colors[j] = colors[j - 1];
      --j;
    }

    colors[j] = color;
  }
}

void main()
{
  vec4 colors[windowSize * windowSize];

  for (int i = 0; i < windowSize; ++i)
  {
    for (int j = 0; j < windowSize; ++j)
    {
      colors[i * windowSize + j] = textureLod(
        colorTex, surf.texCoord - windowSize / 2 + vec2(j, i), 0
      );
    }
  }

  for (int c = 0; c < 4; ++c)
  {
    sortColors(colors, c);
    color[c] = colors[windowSize * windowSize / 2][c];
  }
}
