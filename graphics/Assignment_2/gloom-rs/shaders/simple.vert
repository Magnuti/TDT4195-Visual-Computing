#version 460 core

in layout(location = 0) vec3 position;
in layout(location = 1) vec4 color;
uniform layout(location = 2) mat4 matrix;

out layout(location=0) vec4 outVertexColor;

void main()
{
    // Simply forward the data to the fragment shader with colors
    outVertexColor = color;

    gl_Position = matrix * vec4(position, 1.0f);
}