#version 460 core

// The in name must match the out name in .vert
in vec4 vertexColor;

out vec4 color;

void main()
{
    color = vertexColor;
}