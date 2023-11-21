import os
import subprocess
import pathlib

if __name__ == '__main__':
    glslang_cmd = "glslangValidator"

    shader_list = ["g_buffer_pass.vert", "g_buffer_pass.frag", "deferred_shading_pass.vert", "deferred_shading_pass.frag"]

    for shader in shader_list:
        subprocess.run([glslang_cmd, "-V", shader, "-o", "{}.spv".format(shader)])

