#version 450

layout(location = 0) out vec4 out_fragColor;

void main()
{
  out_fragColor = vec4(gl_FragCoord.z, gl_FragCoord.z * gl_FragCoord.z, 0.0f, 1.0f);
}
