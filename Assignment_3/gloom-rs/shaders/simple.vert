#version 460 core

in layout(location = 0) vec3 position;
in layout(location = 1) vec4 color;
in layout(location = 2) vec3 normal;
uniform layout(location = 3) mat4 mvp_matrix;
uniform layout(location = 4) mat4 model_matrix;

out layout(location = 0) vec4 outVertexColor;
out layout(location = 1) vec3 outVertexNormal;

void main()
{
    // Simply forward the colors to the fragment shaders
    outVertexColor = color;

    // Forward the correct light
    outVertexNormal = normalize(mat3(model_matrix) * normal);;

    gl_Position = mvp_matrix * vec4(position, 1.0f);
}