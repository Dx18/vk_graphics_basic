#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"
#include "unpack_attributes.h"

layout (triangles) in;
layout (triangle_strip, max_vertices = 27) out;

layout (location = 0) in VS_OUT
{
  vec4 vPosNorm;
  vec4 vTexCoordAndTang;
} gIn[];

struct GS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec3 wTangent;
  vec2 texCoord;
};

layout (location = 0) out GS_OUT gOut;

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
} params;

layout (binding = 0, set = 0) uniform AppData
{
  UniformParams Params;
};

GS_OUT calculateGOut(vec4 vPosNorm, vec4 vTexCoordAndTang)
{
  GS_OUT result;

  const vec4 wNorm = vec4(DecodeNormal(floatBitsToInt(vPosNorm.w)),         0.0f);
  const vec4 wTang = vec4(DecodeNormal(floatBitsToInt(vTexCoordAndTang.z)), 0.0f);

  result.wPos     = (params.mModel * vec4(vPosNorm.xyz, 1.0f)).xyz;
  result.wNorm    = normalize(mat3(transpose(inverse(params.mModel))) * wNorm.xyz);
  result.wTangent = normalize(mat3(transpose(inverse(params.mModel))) * wTang.xyz);
  result.texCoord = vTexCoordAndTang.xy;

  return result;
}

GS_OUT interpolateGOut(GS_OUT perVertexGOut[3], float alpha, float beta) {
  GS_OUT result;
  
  result.wPos = perVertexGOut[0].wPos +
    alpha * (perVertexGOut[1].wPos - perVertexGOut[0].wPos) +
    beta * (perVertexGOut[2].wPos - perVertexGOut[0].wPos);
  result.wNorm = perVertexGOut[0].wNorm +
    alpha * (perVertexGOut[1].wNorm - perVertexGOut[0].wNorm) +
    beta * (perVertexGOut[2].wNorm - perVertexGOut[0].wNorm);
  result.wTangent = perVertexGOut[0].wTangent +
    alpha * (perVertexGOut[1].wTangent - perVertexGOut[0].wTangent) +
    beta * (perVertexGOut[2].wTangent - perVertexGOut[0].wTangent);
  result.texCoord = perVertexGOut[0].texCoord +
    alpha * (perVertexGOut[1].texCoord - perVertexGOut[0].texCoord) +
    beta * (perVertexGOut[2].texCoord - perVertexGOut[0].texCoord);

  result.wNorm = normalize(result.wNorm);
  result.wTangent = normalize(result.wTangent);

  return result;
}

void emitBase(GS_OUT perVertexGOut[3], GS_OUT perMiddleVertexGOut[3], int index)
{
  int indexNext = (index + 1) % 3;

  gOut = perVertexGOut[index];
  gl_Position = params.mProjView * vec4(gOut.wPos, 1.0);
  EmitVertex();

  gOut = perVertexGOut[indexNext];
  gl_Position = params.mProjView * vec4(gOut.wPos, 1.0);
  EmitVertex();

  gOut = perMiddleVertexGOut[index];
  gl_Position = params.mProjView * vec4(gOut.wPos, 1.0);
  EmitVertex();

  gOut = perMiddleVertexGOut[indexNext];
  gl_Position = params.mProjView * vec4(gOut.wPos, 1.0);
  EmitVertex();

  EndPrimitive();
}

void emitSide(GS_OUT perMiddleVertexGOut[3], int index, float depth)
{
  int indexNext = (index + 1) % 3;
  vec3 sideVec = perMiddleVertexGOut[indexNext].wPos -
    perMiddleVertexGOut[index].wPos;

  gOut = perMiddleVertexGOut[index];
  vec3 normal = normalize(cross(sideVec, gOut.wNorm));
  gOut.wNorm = normal;
  gl_Position = params.mProjView * vec4(gOut.wPos, 1.0);
  EmitVertex();

  gOut = perMiddleVertexGOut[indexNext];
  vec3 normalNext = normalize(cross(sideVec, gOut.wNorm));
  gOut.wNorm = normalNext;
  gl_Position = params.mProjView * vec4(gOut.wPos, 1.0);
  EmitVertex();

  gOut = perMiddleVertexGOut[index];
  gOut.wPos += gOut.wNorm * depth;
  gOut.wNorm = normal;
  gl_Position = params.mProjView * vec4(gOut.wPos, 1.0);
  EmitVertex();

  gOut = perMiddleVertexGOut[indexNext];
  gOut.wPos += gOut.wNorm * depth;
  gOut.wNorm = normalNext;
  gl_Position = params.mProjView * vec4(gOut.wPos, 1.0);
  EmitVertex();

  EndPrimitive();
}

void emitCap(GS_OUT perMiddleVertexGOut[3], float depth)
{
  for (int i = 0; i < 3; ++i)
  {
    gOut = perMiddleVertexGOut[i];
    gOut.wPos += gOut.wNorm * depth;
    gl_Position = params.mProjView * vec4(gOut.wPos, 1.0);
    EmitVertex();
  }

  EndPrimitive();
}

void main()
{
  GS_OUT perVertexGOut[3];
  for (int i = 0; i < 3; ++i)
  {
    perVertexGOut[i] = calculateGOut(
      gIn[i].vPosNorm, gIn[i].vTexCoordAndTang
    );
  }

  GS_OUT perMiddleVertexGOut[3];
  for (int i = 0; i < 3; ++i)
  {
    perMiddleVertexGOut[i] = interpolateGOut(
      perVertexGOut, i == 1 ? 0.8 : 0.1, i == 2 ? 0.8 : 0.1
    );
  }

  float sinTime = sin(Params.time);
  float cosScaledTime = cos(10.0 * Params.time);
  float shift = sinTime + (abs(sinTime) - pow(sinTime, 4.0)) * cosScaledTime;

  float depth = min(
    min(
      length(perVertexGOut[1].wPos - perVertexGOut[0].wPos),
      length(perVertexGOut[2].wPos - perVertexGOut[1].wPos)
    ),
    length(perVertexGOut[0].wPos - perVertexGOut[2].wPos)
  ) * 0.1 * shift;

  for (int i = 0; i < 3; ++i)
  {
    emitBase(perVertexGOut, perMiddleVertexGOut, i);
  }

  for (int i = 0; i < 3; ++i)
  {
    emitSide(perMiddleVertexGOut, i, depth);
  }

  emitCap(perMiddleVertexGOut, depth);
}