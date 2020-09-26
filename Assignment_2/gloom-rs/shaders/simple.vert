#version 460 core

in vec3 position;
in vec4 color;

// The out name must match the in name in .frag
out vec4 vertexColor;

// {1, 0, 0, 0} is a column (read downwards)
mat4x4 matrix = {{1.4, 0.4, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0.7, 0, 1}};

void main()
{
    // Simply forward the data to the fragment shader with colors
    vertexColor = color;

    gl_Position = matrix * vec4(position, 1.0f);
}