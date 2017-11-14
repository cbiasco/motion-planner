#version 330 core

precision mediump float;

in vec3 in_position;
in vec3 in_color;
in vec3 in_normal;

out vec3 vnormal;
out vec3 vcolor;

uniform vec3 viewdir;
uniform mat4 view;
uniform mat4 projection;

uniform vec3 eye;
uniform int vertex_type;

void main()
{
	vec4 pos = projection * view * vec4(in_position,1.0);

	float dist = distance(eye, in_position);
	if (vertex_type > .5)
		dist *= .5;
	gl_PointSize = max(1.0, 50.0/dist);

	vnormal = in_normal;
	if (vertex_type > 2.5)
		vcolor = in_color;
	else if (vertex_type > 1.5)
		vcolor = vec3(0, 1, 0);
	else if (vertex_type > .5)
		vcolor = vec3(1, 0, 0);
	else
		vcolor = in_color;
	gl_Position = pos;
}
