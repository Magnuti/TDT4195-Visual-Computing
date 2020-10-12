#version 460 core

in layout(location = 0) vec4 vertexColor;

out vec4 color;

void main()
{
    color = vertexColor;
}