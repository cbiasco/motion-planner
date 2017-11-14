#version 330 core

precision mediump float;

out vec4 out_fragcolor;

in vec3 vcolor;
in vec3 vnormal;

uniform vec3 viewdir;
uniform vec3 light;
uniform int drawing_points;

void main() {
	if (drawing_points > .5) {
		out_fragcolor = vec4(vcolor, 1);
	}
	else {
		vec3 L = -normalize(light);

		vec3 N;
		if (gl_FrontFacing)
			N = normalize(vnormal);
		else
			N = -normalize(vnormal);

		vec3 V = -normalize(viewdir);
		vec3 H = normalize(L + V);
		float ambient = .2;
		float diffuse = .4 * max(0.0, dot(N, L));
		float spec = .4 * pow(max(0.0, dot(N, H)), 4);
		vec4 phong = vec4(vcolor * ambient + vcolor * diffuse + vec3(1.0) * spec, 1.0);

		out_fragcolor = phong;
	}
}
