#version 460 core

in layout(location = 0) vec4 vertexColor;
in layout(location = 1) vec3 vertexNormal;

out vec4 color;

void main()
{
    // color = vertexColor;
    color = vec4(vertexNormal.xyz, vertexColor.a);
}