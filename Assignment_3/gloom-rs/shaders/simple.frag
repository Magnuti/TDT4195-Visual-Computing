#version 460 core

in layout(location = 0) vec4 vertexColor;
in layout(location = 1) vec3 vertexNormal;

out vec4 color;

vec3 lightDirection = normalize(vec3(0.8, -0.5, 0.6));

void main()
{
    vec4 colorWithLight = vertexColor * max(0, dot(vertexNormal, -lightDirection));
    color = vec4(colorWithLight.xyz, vertexColor.a);
}