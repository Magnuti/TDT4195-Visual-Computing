#version 460 core

in layout(location = 0) vec3 position;
in layout(location = 1) vec4 color;
in layout(location = 2) vec3 normal;
uniform layout(location = 3) mat4 matrix;

out layout(location = 0) vec4 outVertexColor;
out layout(location = 1) vec3 outVertexNormal;

void main()
{
    // Simply forward the data to the fragment shader with colors
    outVertexColor = color;
    outVertexNormal = normal;

    gl_Position = matrix * vec4(position, 1.0f);
}