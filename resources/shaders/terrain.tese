#version 450

layout (quads) in;

layout (push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
} params;

layout (location = 0) out VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec3 wTangent;
  vec2 texCoord;
} surf;

layout (binding = 2) uniform sampler2D noiseImage;

float getHeight(vec2 position)
{
    return clamp(texture(noiseImage, position).x, 0.0, 1.0);
}

vec3 getNormal(vec2 position)
{
  const float EPS = 0.001;

  float heightXNeg = getHeight(position - vec2(EPS, 0.0));
  float heightXPos = getHeight(position + vec2(EPS, 0.0));
  float heightYNeg = getHeight(position - vec2(0.0, EPS));
  float heightYPos = getHeight(position + vec2(0.0, EPS));

  float gradX = (heightXPos - heightXNeg) / 2.0 / EPS;
  float gradY = (heightYPos - heightYNeg) / 2.0 / EPS;

  return -normalize(vec3(gradX, -1.0, gradY));
}

void main()
{
    vec3 position = vec3(gl_TessCoord.x, getHeight(gl_TessCoord.xy), gl_TessCoord.y);
    vec3 normal = getNormal(gl_TessCoord.xy);

    gl_Position = params.mProjView * params.mModel * vec4(position, 1.0);

    surf.wPos = (params.mModel * vec4(position, 1.0)).xyz;
    surf.wNorm = normalize((transpose(inverse(params.mModel)) * vec4(normal, 1.0)).xyz);
    surf.wTangent = vec3(0.0);
    surf.texCoord = vec2(0.0);
}