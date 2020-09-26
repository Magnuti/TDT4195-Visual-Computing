#version 460 core

in vec3 position;
in vec4 color;

// The out name must match the in name in .frag
out vec4 vertexColor;

void main()
{
    // Simply forward the data to the fragment shader with colors
    vertexColor = color;

    gl_Position = vec4(position, 1.0f);
}