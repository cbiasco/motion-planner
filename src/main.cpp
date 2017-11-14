
// The loaders are included by glfw3 (glcorearb.h) if we are not using glew.
#ifdef USE_GLEW
#include <GL/glew.h>
#endif
#include <GLFW/glfw3.h>

// Includes
#include <omp.h>
#include "trimesh.hpp"
#include "shader.hpp"
#include <cstring> // memcpy
#include <random>
#include <algorithm>
#include <set>
#include <stdlib.h>
#include <omp.h>

// Constants
#define WIN_WIDTH 800
#define WIN_HEIGHT 800
#define RECT false
#define SPHERE true

#define PI 4.0*atan(1.0)
#define GRAVITY -9.8
#define MAXDT .25
#define MINHEIGHT 0.01
#define NUMPRESETS 6

#define DEBUG 0
#define USE_BVH 0
#define USE_OMP 1

using std::vector;
using std::string;
using std::set;
using std::max;
using std::min;

typedef Vec<2,float> Vec2f;

class Mat4x4 {
public:

	float m[16];

	Mat4x4(){ // Default: Identity
		m[0] = 1.f;  m[4] = 0.f;  m[8]  = 0.f;  m[12] = 0.f;
		m[1] = 0.f;  m[5] = 1.f;  m[9]  = 0.f;  m[13] = 0.f;
		m[2] = 0.f;  m[6] = 0.f;  m[10] = 1.f;  m[14] = 0.f;
		m[3] = 0.f;  m[7] = 0.f;  m[11] = 0.f;  m[15] = 1.f;
	}

	void make_identity(){
		m[0] = 1.f;  m[4] = 0.f;  m[8]  = 0.f;  m[12] = 0.f;
		m[1] = 0.f;  m[5] = 1.f;  m[9]  = 0.f;  m[13] = 0.f;
		m[2] = 0.f;  m[6] = 0.f;  m[10] = 1.f;  m[14] = 0.f;
		m[3] = 0.f;  m[7] = 0.f;  m[11] = 0.f;  m[15] = 1.f;
	}

	void print(){
		std::cout << m[0] << ' ' <<  m[4] << ' ' <<  m[8]  << ' ' <<  m[12] << "\n";
		std::cout << m[1] << ' ' <<   m[5] << ' ' <<  m[9]  << ' ' <<   m[13] << "\n";
		std::cout << m[2] << ' ' <<   m[6] << ' ' <<  m[10] << ' ' <<   m[14] << "\n";
		std::cout << m[3] << ' ' <<   m[7] << ' ' <<  m[11] << ' ' <<   m[15] << "\n";
	}

	void make_scale(float x, float y, float z){
		make_identity();
		m[0] = x; m[5] = y; m[10] = x;
	}
};

static inline const Vec3f operator*(const Mat4x4 &m, const Vec3f &v) {
	Vec3f r(m.m[0] * v[0] + m.m[4] * v[1] + m.m[8] * v[2],
		m.m[1] * v[0] + m.m[5] * v[1] + m.m[9] * v[2],
		m.m[2] * v[0] + m.m[6] * v[1] + m.m[10] * v[2]);
	return r;
}
static inline const Vec3f operator*(const double d, const Vec3f &v) {
	Vec3f r(v[0] * d, v[1] * d, v[2] * d);
	return r;
}
static inline const Vec3f operator*(const Vec3f &v, const double d) {
	Vec3f r(v[0] * d, v[1] * d, v[2] * d);
	return r;
}
static inline const Vec3f operator+(const Vec3f &v1, const Vec3f &v2) {
	Vec3f r(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
	return r;
}

float dot(Vec3f v1, Vec3f v2) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}


typedef struct {
	bool active = false;
	GLdouble prev_x;
	GLdouble prev_y;
} MouseInfo;

//
//	Global state variables
//
namespace Globals {
	float win_width, win_height; // window size
	float aspect;
	int vert_dim = 2;
	GLuint position_vbo[1], colors_vbo[1], faces_ibo[1], edges_ibo[1], normals_vbo[1], scene_vao;
	Vec3f lightDir;

	//  Model, view and projection matrices, initialized to the identity
	Mat4x4 model;
	Mat4x4 view;
	Mat4x4 projection;

	// Scene variables
	Vec3f eye;
	float near = .1;
	float far = 1000;
	float left = -.1;
	float right = .1;
	float top = .1;
	float bottom = -.1;
	Vec3f viewDir;
	Vec3f upDir;
	Vec3f rightDir;

	// Input variables
	bool key_w; // forward movement
	bool key_s; // backward movement
	bool key_d; // right strafing
	bool key_a; // left strafing
	bool key_e; // upward movement
	bool key_q; // downward movement
	bool key_lshift; // speed up

	bool key_num2;
	bool key_num4;
	bool key_num6;
	bool key_num8;
	bool key_up;
	bool key_down;
	bool key_left;
	bool key_right;

	double theta;
	double phi;
	Mat4x4 xRot;
	Mat4x4 yRot;

	MouseInfo mouse;
	mcl::Shader curShader;
	double curTime, prevTime;
	int preset = 0;
	bool pause = true, addTimeMultiplier = false, subTimeMultiplier = false;
	double timeMultiplier = 1;
	double movementSpeed = 0.1;
	GLFWwindow *activeWindow;
}

class Object {
public:
	Object() {
		pos = Vec3f();
	}

	Object(float x, float y) {
		pos = Vec3f(x, y, 0.0);
	}

	Object(Vec3f p) {
		pos = p;
	}

	virtual ~Object() {}

	virtual void setParams(float x) {
		params[0] = x;
		params[1] = x;
		params[2] = x;

		construct();
	}

	virtual void setParams(float x, float y) {
		params[0] = x;
		params[1] = y;
		params[2] = 0;

		construct();
	}

	virtual void setParams(float x, float y, float z) {
		params[0] = x;
		params[1] = y;
		params[2] = z;

		construct();
	}

	void setColor(float r, float g, float b) {
		setColor(Vec3f(r, g, b));
	}

	void setColor(Vec3f rgb) {
		color = Vec3f(max(0.0f, min(1.0f, rgb[0])), max(0.0f, min(1.0f, rgb[1])), max(0.0f, min(1.0f, rgb[2])));
		for (int i = 0; i < colors.size(); i++) {
			colors[i] = color;
		}
	}

	void moveBy(float x, float y, float z) {
		moveBy(Vec3f(x, y, z));
	}

	void moveBy(Vec3f t) {
		pos += t;
		for (int i = 0; i < vertices.size(); i++) {
			vertices[i] += t;
		}
	}

	void moveTo(float x, float y, float z) {
		moveTo(Vec3f(x, y, z));
	}

	void moveTo(Vec3f position) {
		Vec3f p = position - pos;
		pos += p;
		for (int i = 0; i < vertices.size(); i++) {
			vertices[i] += p;
		}
	}

	virtual Object *cObstacle(Vec3f inputParams) = 0;

	virtual bool collides(Vec3f p) = 0;

	virtual bool intersects(Vec3f origin, Vec3f ray) = 0;

	virtual void construct() = 0;

	void render() {
		using namespace Globals;

		glBindBuffer(GL_ARRAY_BUFFER, position_vbo[0]);
		if (!vertices.empty())
			glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), &vertices[0][0], GL_DYNAMIC_DRAW);
		else
			glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, colors_vbo[0]);
		if (!colors.empty())
			glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(colors[0]), &colors[0][0], GL_DYNAMIC_DRAW);
		else
			glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, normals_vbo[0]);
		if (!normals.empty())
			glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(normals[0]), &normals[0][0], GL_DYNAMIC_DRAW);
		else
			glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faces_ibo[0]);
		if (!faces.empty())
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.size() * sizeof(faces[0]), &faces[0], GL_STATIC_DRAW);
		else
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

		glDrawElements(GL_TRIANGLES, faces.size(), GL_UNSIGNED_INT, (GLvoid*) 0);
	}

	Vec3f pos;
	Vec3f params;
	Vec3f color = Vec3f(0, 0, 1);
	bool sphere;

	// Rendering values
	vector<Vec3f> vertices;
	vector<Vec3f> colors;
	vector<Vec3f> normals;
	vector<GLuint> faces;
};

class Sphere : public Object {
public:
	Sphere() : Object() {
		params[0] = 1.0;
		sphere = true;
		construct();
	}

	Sphere(float x, float y) : Object(Vec3f(x, y, 0.0)) {
		params[0] = 1.0;
		sphere = true;
		construct();
	}

	Sphere(float x, float y, float r) : Object(Vec3f(x, y, 0.0)) {
		params[0] = r;
		sphere = true;
		construct();
	}

	Sphere(float r) : Object() {
		params[0] = r;
		sphere = true;
		construct();
	}

	Sphere(Vec3f p, float r) : Object(p) {
		params[0] = r;
		sphere = true;
		construct();
	}

	virtual ~Sphere() {}

	void construct() {
		vertices.clear();
		colors.clear();
		normals.clear();
		faces.clear();

		int stacks = 20;
		int slices = 20;
		for (int i = 0; i < stacks+1; i++) {
			int nextStack = (i+1)%(stacks+1);
			double longitude = 2 * PI * i / stacks;
			//calculate the vertices and colors
			for (int j = 0; j < slices+1; j++) {
				int nextSlice = (j+1)%(slices+1);
				double colatitude = PI * j / slices;
				double x = pos[0] + params[0] * cos(longitude) * sin(colatitude);
				double y = pos[1] + params[0] * sin(longitude) * sin(colatitude);
				double z = pos[2] + params[0] * cos(colatitude);
				vertices.push_back(Vec3f(x, y, z));
				colors.push_back(color);
				normals.push_back(pos - vertices.back());

				faces.push_back(j + i*slices);
				faces.push_back(j + (nextStack)*slices);
				faces.push_back(nextSlice + (nextStack)*slices);

				faces.push_back(j + i*slices);
				faces.push_back(nextSlice + (nextStack)*slices);
				faces.push_back(nextSlice + i*slices);
			}
		}
	}

	Object *cObstacle(Vec3f inputParams) {
		Sphere *s = new Sphere(pos, params[0] + inputParams[0]);
		s->setColor(.5, .5, .5);

		return s;
	}

	bool collides(Vec3f p) {
		return (p - pos).len2() < params[0]*params[0];
	}

	bool intersects(Vec3f origin, Vec3f ray) {
		Vec3f objectRay = origin - pos;
		Vec3f pointRay = ray;
		pointRay.normalize();
		float magnitude = ray.len();

		float discriminant = pointRay.dot(objectRay)*pointRay.dot(objectRay) - (objectRay.len2() - params[0]*params[0]);

		if (discriminant < 0)
			return false;

		float distance1 = -pointRay.dot(objectRay) - sqrt(discriminant);
		float distance2 = -pointRay.dot(objectRay) + sqrt(discriminant);
		
		if (distance1 < magnitude && distance2 < magnitude && distance1 > 0 && distance2 > 0)
			return true;
		
		return false;
	}

	// params[0] is radius
};

class Rectangle : public Object {
public:
	Rectangle() : Object() {
		params[0] = 1.0;
		params[1] = 1.0;
		sphere = false;
		construct();
	}

	Rectangle(float x, float y) : Object(Vec3f(x, y, 0.0)) {
		params[0] = 1.0;
		params[1] = 1.0;
		sphere = false;
		construct();
	}

	Rectangle(float x, float y, float w, float h) : Object(Vec3f(x, y, 0.0)) {
		params[0] = w;
		params[1] = h;
		sphere = false;
		construct();
	}

	Rectangle(Vec3f p) : Object(p) {
		params[0] = 1.0;
		params[1] = 1.0;
		sphere = false;
		construct();
	}

	Rectangle(Vec3f p, float w, float h) : Object(p) {
		params[0] = w;
		params[1] = h;
		sphere = false;
		construct();
	}

	virtual ~Rectangle() {}

	void construct() {
		vertices.clear();
		colors.clear();
		normals.clear();
		faces.clear();

		// Front face
		vertices.push_back(pos + Vec3f(params[0]/2, -params[1]/2, pos[2] + 1));
		vertices.push_back(pos + Vec3f(-params[0]/2, -params[1]/2, pos[2] + 1));
		vertices.push_back(pos + Vec3f(params[0]/2, params[1]/2, pos[2] + 1));
		vertices.push_back(pos + Vec3f(-params[0]/2, params[1]/2, pos[2] + 1));

		for (int i = 0; i < 4; i++)
			normals.push_back(Vec3f(0, 0, 1));

		// Left face
		vertices.push_back(pos + Vec3f(-params[0]/2, -params[1]/2, pos[2] + 1));
		vertices.push_back(pos + Vec3f(-params[0]/2, -params[1]/2, pos[2] - 1));
		vertices.push_back(pos + Vec3f(-params[0]/2, params[1]/2, pos[2] + 1));
		vertices.push_back(pos + Vec3f(-params[0]/2, params[1]/2, pos[2] - 1));

		for (int i = 0; i < 4; i++)
			normals.push_back(Vec3f(-1, 0, 0));

		// Back face
		vertices.push_back(pos + Vec3f(-params[0]/2, -params[1]/2, pos[2] - 1));
		vertices.push_back(pos + Vec3f(params[0]/2, -params[1]/2, pos[2] - 1));
		vertices.push_back(pos + Vec3f(-params[0]/2, params[1]/2, pos[2] - 1));
		vertices.push_back(pos + Vec3f(params[0]/2, params[1]/2, pos[2] - 1));

		for (int i = 0; i < 4; i++)
			normals.push_back(Vec3f(0, 0, -1));

		// Right face
		vertices.push_back(pos + Vec3f(params[0]/2, -params[1]/2, pos[2] - 1));
		vertices.push_back(pos + Vec3f(params[0]/2, -params[1]/2, pos[2] + 1));
		vertices.push_back(pos + Vec3f(params[0]/2, params[1]/2, pos[2] - 1));
		vertices.push_back(pos + Vec3f(params[0]/2, params[1]/2, pos[2] + 1));

		for (int i = 0; i < 4; i++)
			normals.push_back(Vec3f(1, 0, 0));

		// Top face
		vertices.push_back(pos + Vec3f(params[0]/2, params[1]/2, pos[2] + 1));
		vertices.push_back(pos + Vec3f(-params[0]/2, params[1]/2, pos[2] + 1));
		vertices.push_back(pos + Vec3f(params[0]/2, params[1]/2, pos[2] - 1));
		vertices.push_back(pos + Vec3f(-params[0]/2, params[1]/2, pos[2] - 1));

		for (int i = 0; i < 4; i++)
			normals.push_back(Vec3f(0, 1, 0));

		// Bottom face
		vertices.push_back(pos + Vec3f(params[0]/2, -params[1]/2, pos[2] - 1));
		vertices.push_back(pos + Vec3f(-params[0]/2, -params[1]/2, pos[2] - 1));
		vertices.push_back(pos + Vec3f(params[0]/2, -params[1]/2, pos[2] + 1));
		vertices.push_back(pos + Vec3f(-params[0]/2, -params[1]/2, pos[2] + 1));

		for (int i = 0; i < 4; i++)
			normals.push_back(Vec3f(0, -1, 0));


		for (int i = 0; i < 24; i++)
			colors.push_back(color);

		for (int i = 0; i < 6; i++) {
			faces.push_back(i*4);
			faces.push_back(i*4 + 2);
			faces.push_back(i*4 + 3);

			faces.push_back(i*4);
			faces.push_back(i*4 + 3);
			faces.push_back(i*4 + 1);
		}
	}

	Object *cObstacle(Vec3f inputParams) {
		Rectangle *r = new Rectangle(pos, params[0] + inputParams[0], params[1] + inputParams[1]);
		r->setColor(.5, .5, .5);

		return r;
	}

	bool collides(Vec3f p) {
		return p[0] > pos[0]-params[0]/2 && p[0] < pos[0]+params[0]/2 &&
				p[1] > pos[1]-params[1]/2 && p[1] < pos[1]+params[1]/2;
	}

	// Code adapted from Alejo's rectangle intersection solution at http://stackoverflow.com/a/293052
	bool intersects(Vec3f origin, Vec3f ray) {
		Vec3f topLeft, topRight, bottomLeft, bottomRight, start, end;
		topLeft = Vec3f(pos[0] - params[0]/2, pos[1] + params[1]/2, 0);
		topRight = Vec3f(pos[0] + params[0]/2, pos[1] + params[1]/2, 0);
		bottomLeft = Vec3f(pos[0] - params[0]/2, pos[1] - params[1]/2, 0);
		bottomRight = Vec3f(pos[0] + params[0]/2, pos[1] - params[1]/2, 0);
		
		start = origin;
		end = origin + ray;

		float tl, tr, bl, br;
		tl = (end[1]-start[1])*topLeft[0] + (start[0]-end[0])*topLeft[1]
			+ (end[0]*start[1] - start[0]*end[1]);
		tr = (end[1]-start[1])*topRight[0] + (start[0]-end[0])*topRight[1]
			+ (end[0]*start[1] - start[0]*end[1]);
		bl = (end[1]-start[1])*bottomLeft[0] + (start[0]-end[0])*bottomLeft[1]
			+ (end[0]*start[1] - start[0]*end[1]);
		br = (end[1]-start[1])*bottomRight[0] + (start[0]-end[0])*bottomRight[1]
			+ (end[0]*start[1] - start[0]*end[1]);

		if ((tl > 0 && tr > 0 && bl > 0 && br > 0) || !(tl > 0 || tr > 0 || bl > 0 || br > 0))
			return false;

		if ((start[0] > topRight[0] && end[0] > topRight[0]) ||
			(start[0] < bottomLeft[0] && end[0] < bottomLeft[0]) ||
			(start[1] > topRight[1] && end[1] > topRight[1]) ||
			(start[1] < bottomLeft[1] && end[1] < bottomLeft[1]))
			return false;

		return true;
	}

	// width is params[0]
	// height is params[1]
};

class PRM {
public:
	PRM() {}

	PRM(Object *agent, vector<Object *> obstacles) {
		generateCObstacles(agent, obstacles);
	}

	virtual ~PRM() {
		clearCObstacles();
	}

	void clearCObstacles() {
		for (int i = 0; i < cObstacles.size(); i++) {
			if (cObstacles[i])
				delete cObstacles[i];
		}
		cObstacles.clear();
	}

	void generateCObstacles(Object *agent, vector<Object *> obstacles) {
		clearCObstacles();

		for (int i = 0; i < obstacles.size(); i++) {
			cObstacles.push_back(obstacles[i]->cObstacle(agent->params));
		}
	}

	void addCObstacle(Object *agent, Object *obstacle) {
		cObstacles.push_back(obstacle->cObstacle(agent->params));
	}

	bool cCollides(Vec3f point) {
		for (int i = 0; i < cObstacles.size(); i++) {
			if (cObstacles[i]->collides(point)) {
				return true;
			}
		}
		return false;
	}

	bool cIntersects(Vec3f src, Vec3f dst) {
		Vec3f vector;
		for (int i = 0; i < cObstacles.size(); i++) {
			vector = dst - src;
			if (cObstacles[i]->intersects(src, vector)) {
				return true;
			}
		}
		return false;
	}

	// DEPRECATED - Use setBounds to manually place the bounds
	//				(bounds can also be expanded by start and goal points)
	void updateBounds() {
		for (int i = 0; i < cObstacles.size(); i++) {
			if (cObstacles[i]->pos[0] < minBound[0])
				minBound[0] = cObstacles[i]->pos[0];
			else if (cObstacles[i]->pos[0] > maxBound[0])
				maxBound[0] = cObstacles[i]->pos[0];
			if (cObstacles[i]->pos[1] < minBound[1])
				minBound[1] = cObstacles[i]->pos[1];
			else if (cObstacles[i]->pos[1] > maxBound[1])
				maxBound[1] = cObstacles[i]->pos[1];
		}

		if (start >= 0) {
			if (points[start][0] < minBound[0])
				minBound[0] = points[start][0];
			else if (points[start][0] > maxBound[0])
				maxBound[0] = points[start][0];
			if (points[start][1] < minBound[1])
				minBound[1] = points[start][1];
			else if (points[start][1] > maxBound[1])
				maxBound[1] = points[start][1];
		}

		if (goal >= 0) {
			if (points[goal][0] < minBound[0])
				minBound[0] = points[goal][0];
			else if (points[goal][0] > maxBound[0])
				maxBound[0] = points[goal][0];
			if (points[goal][1] < minBound[1])
				minBound[1] = points[goal][1];
			else if (points[goal][1] > maxBound[1])
				maxBound[1] = points[goal][1];
		}
	}

	Vec3f sampleSpace() {
		// would like to use RAND_MAX, but for some reason it is undefined
		float x, y;
		x = (float(rand() % 10000)/10000)*(maxBound[0] - minBound[0]);
		y = (float(rand() % 10000)/10000)*(maxBound[1] - minBound[1]);
		return Vec3f(x + minBound[0], y + minBound[1], 0);
	}

	void addPoint(Vec3f point) {
		points.push_back(point);
		edges.push_back(vector<int>());
		colors.push_back(Vec3f());
		normals.push_back(Vec3f(0, 0, 0));

		for (int i = 0; i < points.size() - 1; i++) {
			if (!cIntersects(point, points[i])) {
				edges[i].push_back(points.size() - 1);
				edges[points.size() - 1].push_back(i);
			}
		}
	}

	void generatePRM(int n = -1) {
		points.clear();
		edges.clear();
		path.clear();

		Vec3f point;
		int numPoints;

		if (n < 0) {
			float xDim = maxBound[0] - minBound[0];
			float yDim = maxBound[1] - minBound[1];
			numPoints = xDim*yDim/8; // arbitrary scaling factor
		}
		else {
			numPoints = n;
		}

		for (int i = 0; i < numPoints; i++) {
			point = sampleSpace();
			
			if (cCollides(point))
				continue;

			addPoint(point);
		}

		if (start >= 0 && !cCollides(startVec)) {
			start = points.size();
			addPoint(startVec);
		}

		if (goal >= 0 && !cCollides(goalVec)) {
			goal = points.size();
			addPoint(goalVec);
		}
	}

	bool setStart(Vec3f s) {
		if (cCollides(s))
			return false;

		if (s[0] < minBound[0])
			minBound[0] = s[0];
		else if (s[0] > maxBound[0])
			maxBound[0] = s[0];
		if (s[1] < minBound[1])
			minBound[1] = s[1];
		else if (s[1] > maxBound[1])
			maxBound[1] = s[1];

		start = points.size();
		startVec = s;
		addPoint(s);

		return true;
	}

	bool setGoal(Vec3f g) {
		if (cCollides(g))
			return false;

		if (g[0] < minBound[0])
			minBound[0] = g[0];
		else if (g[0] > maxBound[0])
			maxBound[0] = g[0];
		if (g[1] < minBound[1])
			minBound[1] = g[1];
		else if (g[1] > maxBound[1])
			maxBound[1] = g[1];

		goal = points.size();
		goalVec = g;
		addPoint(g);

		return true;
	}

	void setBounds(Vec3f min, Vec3f max) {
		minBound = min;
		maxBound = max;

		if (start >= 0 && (points[start][0] < minBound[0] || points[start][0] > maxBound[0] ||
						   points[start][1] < minBound[1] || points[start][1] > maxBound[1])) {
			start = -1;
		}

		if (goal >= 0 && (points[goal][0] < minBound[0] || points[goal][0] > maxBound[0] ||
						  points[goal][1] < minBound[1] || points[goal][1] > maxBound[1])) {
			goal = -1;
		}
	}

	bool findPath(bool Astar = true) {
		if (start == -1 || goal == -1) {
			return false;
		}

		path.clear();
		drawnPath.clear();

		vector<float> gScore;
		vector<float> fScore;
		vector<int> cameFrom;

		for (int i = 0; i < points.size(); i++) {
			gScore.push_back(-1);
			fScore.push_back(-1);
			cameFrom.push_back(-1);
		}

		set<int> closed;
		set<int> open;
		open.insert(start);

		gScore[start] = 0;
		if (Astar) fScore[start] = (points[goal] - points[start]).len();

		int current, vertex;
		float min;
		while (open.size() > 0) {

			current = -1;
			min = -1;
			for (set<int>::iterator i = open.begin(); i != open.end(); i++) {
				vertex = *i;
				if (Astar && (min < 0 || (fScore[vertex] >= 0 && fScore[vertex] < min))) {
					min = fScore[vertex];
					current = vertex;
				}
				else if (!Astar && (min < 0 || (gScore[vertex] >= 0 && gScore[vertex] < min))) {
					min = gScore[vertex];
					current = vertex;
				}
			}

			if (current == goal) {
				vector<int> finalPath;
				finalPath.push_back(current);
				while (cameFrom[current] != -1) {
					current = cameFrom[current];
					finalPath.push_back(current);
				}
				for (int i = 0; i < finalPath.size(); i++) {
					path.push_back(finalPath[finalPath.size() - 1 - i]);
				}
				generateDrawnPath();
				return true;
			}

			open.erase(current);
			closed.insert(current);
			for (int i = 0; i < edges[current].size(); i++) {
				int neighbor = edges[current][i];
				if (closed.find(neighbor) != closed.end())
					continue;

				float temp_gScore = gScore[current] + (points[neighbor] - points[current]).len();
				if (open.find(neighbor) == open.end()) {
					open.insert(neighbor);
				}
				else if (temp_gScore >= gScore[neighbor])
					continue;

				cameFrom[neighbor] = current;
				gScore[neighbor] = temp_gScore;
				if (Astar) fScore[neighbor] = gScore[neighbor] + (points[goal] - points[neighbor]).len();
			}
		}
		
		// Failed to find path to goal
		generateDrawnPath();
		return false;
	}

	// only works with spheres currently
	bool reachedGoal(Object *agent, int index, float speed) {
		if (index == path.size() - 1 && (points[goal] - agent->pos).len2() < speed*speed)
			return true;
		return false;
	}

	bool nextPointInPathIsVisible(Object *agent, int index) {
		if (index > path.size() - 1) {
			return false;
		}

		if (cIntersects(agent->pos, points[path[index]])) {
			return false;
		}

		return true;
	}

	Vec3f getPointInPath(int index) {
		if (index < 0 || index > path.size() - 1) {
			std::cout << "PRM: Cannot access point out of path vector bounds" << std::endl;
			return Vec3f();
		}
		return points[path[index]];
	}

	void updateEdges() {
		edgeIdx.clear();

		for (int i = 0; i < edges.size(); i++) {
			for (int j = 0; j < edges[i].size(); j++) {
				edgeIdx.push_back(i);
				edgeIdx.push_back(edges[i][j]);
			}
		}
	}

	void generateDrawnPath() {
		drawnPath.clear();
		drawnPoints.clear();
		drawnColors.clear();
		drawnNormals.clear();
		if (path.size() == 0)
			return;

		double width = .05;
		Vec3f vertex, rightDir;
		Vec3f green = Vec3f(0, 1, 0);
		Vec3f normal = Vec3f(0, 0, 1);
		
		// First point
		vertex = points[path[0]];
		rightDir = (points[path[1]] - vertex).cross(normal);
		rightDir.normalize();

		// vertices
		drawnPoints.push_back(vertex + rightDir*width + Vec3f(0, 0, .02));
		drawnPoints.push_back(vertex - rightDir*width + Vec3f(0, 0, .02));

		// colors
		drawnColors.push_back(green);
		drawnColors.push_back(green);

		// normals
		drawnNormals.push_back(normal);
		drawnNormals.push_back(normal);

		// faces
		drawnPath.push_back(0);
		drawnPath.push_back(2);
		drawnPath.push_back(3);

		drawnPath.push_back(0);
		drawnPath.push_back(3);
		drawnPath.push_back(1);

		for (int i = 1; i < path.size() - 1; i++) {
			vertex = points[path[i]];
			rightDir = (points[path[i+1]] - points[path[i-1]]).cross(normal);
			rightDir.normalize();

			// vertices
			drawnPoints.push_back(vertex + rightDir*width + Vec3f(0, 0, .02));
			drawnPoints.push_back(vertex - rightDir*width + Vec3f(0, 0, .02));

			// colors
			drawnColors.push_back(green);
			drawnColors.push_back(green);

			// normals
			drawnNormals.push_back(normal);
			drawnNormals.push_back(normal);

			// faces
			drawnPath.push_back(i*2);
			drawnPath.push_back((i+1)*2);
			drawnPath.push_back((i+1)*2 + 1);
			
			drawnPath.push_back(i*2);
			drawnPath.push_back((i+1)*2 + 1);
			drawnPath.push_back(i*2 + 1);
		}

		// Last point
		vertex = points[path.back()];
		rightDir = (points[path[path.size() - 2]] - vertex).cross(-1*normal);
		rightDir.normalize();

		// vertices
		drawnPoints.push_back(vertex + rightDir*width + Vec3f(0, 0, .02));
		drawnPoints.push_back(vertex - rightDir*width + Vec3f(0, 0, .02));

		// colors
		drawnColors.push_back(green);
		drawnColors.push_back(green);

		// normals
		drawnNormals.push_back(normal);
		drawnNormals.push_back(normal);
	}

	void render() {
		using namespace Globals;

		glBindBuffer(GL_ARRAY_BUFFER, position_vbo[0]);
		if (!points.empty())
			glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(points[0]), &points[0][0], GL_DYNAMIC_DRAW);
		else
			glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, colors_vbo[0]);
		if (!colors.empty())
			glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(colors[0]), &colors[0][0], GL_DYNAMIC_DRAW);
		else
			glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, normals_vbo[0]);
		if (!normals.empty())
			glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(normals[0]), &normals[0][0], GL_DYNAMIC_DRAW);
		else
			glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edges_ibo[0]);
		if (!edgeIdx.empty())
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, edgeIdx.size() * sizeof(edgeIdx[0]), &edgeIdx[0], GL_STATIC_DRAW);
		else
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

		// Draw points
		glUniform1i(curShader.uniform("drawing_points"), 1);
		glDrawArrays(GL_POINTS, 0, points.size());

		// Draw start and goal
		if (start >= 0) {
			glUniform1i(curShader.uniform("vertex_type"), 2);
			glDrawArrays(GL_POINTS, start, 1);
		}
		if (goal >= 0) {
			glUniform1i(curShader.uniform("vertex_type"), 1);
			glDrawArrays(GL_POINTS, goal, 1);
		}
		glUniform1i(curShader.uniform("vertex_type"), 0);
		glUniform1i(curShader.uniform("drawing_points"), 0);

		// Draw edges
		glDrawElements(GL_LINES, edgeIdx.size(), GL_UNSIGNED_INT, (GLvoid*) 0);
	}

	void renderPath() {
		using namespace Globals;

		glBindBuffer(GL_ARRAY_BUFFER, position_vbo[0]);
		if (!drawnPoints.empty())
			glBufferData(GL_ARRAY_BUFFER, drawnPoints.size() * sizeof(drawnPoints[0]), &drawnPoints[0][0], GL_DYNAMIC_DRAW);
		else
			glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, colors_vbo[0]);
		if (!drawnColors.empty())
			glBufferData(GL_ARRAY_BUFFER, drawnColors.size() * sizeof(drawnColors[0]), &drawnColors[0][0], GL_DYNAMIC_DRAW);
		else
			glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, normals_vbo[0]);
		if (!drawnNormals.empty())
			glBufferData(GL_ARRAY_BUFFER, drawnNormals.size() * sizeof(drawnNormals[0]), &drawnNormals[0][0], GL_DYNAMIC_DRAW);
		else
			glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edges_ibo[0]);
		if (!drawnPath.empty())
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, drawnPath.size() * sizeof(drawnPath[0]), &drawnPath[0], GL_DYNAMIC_DRAW);
		else
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);
		
		// Draw path
		glDrawElements(GL_TRIANGLES, drawnPath.size(), GL_UNSIGNED_INT, (GLvoid*) 0);

	}

	void renderCObstacles() {
		using namespace Globals;

		for (int i = 0; i < cObstacles.size(); i++) {
			cObstacles[i]->render();
		}
	}

	vector<Object *> cObstacles;
	vector<Vec3f> points;
	vector<vector<int>> edges;
	vector<int> path;
	int start = -1;
	int goal = -1;
	Vec3f startVec;
	Vec3f goalVec;
	Vec3f minBound;
	Vec3f maxBound;

	// Rendering values
	vector<Vec3f> drawnPoints;
	vector<int> drawnPath;
	vector<Vec3f> drawnColors;
	vector<Vec3f> drawnNormals;
	vector<Vec3f> colors;
	vector<Vec3f> normals;
	vector<GLuint> edgeIdx;

};

class Scene {
public:
	Scene() {
		agent = new Sphere();
		setAgentSize(1.0);
		setAgentColor(1, 0, 0);

		setStart(-10, -10);
		setGoal(10, 10);

		updatePRM();
	}

	virtual ~Scene() {
		if (agent)
			delete agent;

		if (currentObject)
			delete currentObject;

		for (int i = 0; i < objects.size(); i++) {
			delete objects[i];
		}
		objects.clear();
	}

	void changeShape() {
		if (placingObject || placingStart || placingGoal) return;

		if (agent)
			delete agent;

		hasPath = false;
		isGenerated = false;
		simulationStarted = false;
		simulationComplete = false;
		Globals::pause = true;
		agentIsSphere = !agentIsSphere;
		if (agentIsSphere)
			agent = new Sphere();
		else
			agent = new Rectangle();
		agent->setColor(1, 0, 0);

		prm.clearCObstacles();
		for (int i = 0; i < objects.size(); i++) {
			delete objects[i];
		}
		objects.clear();
		setStart(-10, -10);
		setGoal(10, 10);

		updatePRM();
	}

	void setAgentSize(float x) {
		if (simulationStarted) return;

		agent->setParams(x);

		prm.generatePRM();
		prm.updateEdges();
	}

	void setAgentSize(float x, float y) {
		if (simulationStarted) return;

		agent->setParams(x, y);

		prm.generatePRM();
		prm.updateEdges();
	}

	void addSphere(float x, float y, float r) {
		if (simulationStarted) return;

		objects.push_back(new Sphere(x, y, r));
		prm.addCObstacle(agent, objects.back());

		updatePRM();
	}

	void addRectangle(float x, float y, float w, float h) {
		if (simulationStarted) return;

		objects.push_back(new Rectangle(x, y, w, h));
		prm.addCObstacle(agent, objects.back());

		updatePRM();
	}

	void addCurrentObject() {
		objects.push_back(currentObject);
		prm.addCObstacle(agent, objects.back());

		updatePRM();

		currentObject = NULL;
	}

	void placeObject() {
		if (placingStart) {
			placeStart();
			return;
		}
		else if (placingGoal) {
			placeGoal();
			return;
		}

		if (placingObject) {
			currentObject->setColor(Vec3f(0, 0, 1));
			addCurrentObject();
			placingObject = false;
			return;
		}

		if (agentIsSphere) {
			currentObject = new Sphere(0, 0, 1.0);
		}
		else {
			currentObject = new Rectangle(0, 0, 1.0, 1.0);
		}

		currentObject->setColor(Vec3f(0, 1, 0));
		placingObject = true;
	}

	void placeStart() {
		if (placingObject || placingGoal) return;

		if (placingStart) {
			setStart(currentPoint);
			placingStart = false;
			return;
		}

		currentPoint = Vec3f(0, 0, .2);
		placingStart = true;
	}

	void placeGoal() {
		if (placingObject || placingStart) return;

		if (placingGoal) {
			setGoal(currentPoint);
			placingGoal = false;
			return;
		}

		currentPoint = Vec3f(0, 0, .2);
		placingGoal = true;
	}

	void cancelPlacing() {
		if (placingObject) {
			delete currentObject;
			placingObject = false;
		}
		else if (placingStart || placingGoal) {
			placingStart = false;
			placingGoal = false;
		}
	}

	void moveObject(Vec3f v) {
		if (placingObject) {
			currentObject->moveBy(v);
		}
		else if (placingStart || placingGoal) {
			currentPoint += v;
		}
	}

	void changeObjectSize1(float dx) {
		if (simulationStarted) return;

		if (placingStart || placingGoal) return;
		else if (placingObject) {
			currentObject->setParams(max(.1f, currentObject->params[0] + dx), currentObject->params[1]);
		}
		else {
			agent->setParams(max(.1f, agent->params[0] + dx), agent->params[1]);
		}
	}

	void changeObjectSize2(float dy) {
		if (simulationStarted) return;

		if (placingStart || placingGoal) return;
		else if (placingObject) {
			if (agentIsSphere)
				currentObject->setParams(max(.1f, currentObject->params[0] + dy), currentObject->params[1]);
			else
				currentObject->setParams(currentObject->params[0], max(.1f, currentObject->params[1] + dy));
		}
		else {
			if (agentIsSphere)
				agent->setParams(max(.1f, agent->params[0] + dy), agent->params[1]);
			else
				agent->setParams(agent->params[0], max(.1f, agent->params[1] + dy));
		}
	}

	void updateCObstacles() {
		if (placingObject || placingStart || placingGoal) return;

		prm.clearCObstacles();
		prm.generateCObstacles(agent, objects);
		updatePRM();
	}

	void setAgentColor(float r, float g, float b) {
		setAgentColor(Vec3f(r, g, b));
	}

	void setAgentColor(Vec3f rgb) {
		agent->setColor(rgb);
	}

	void testPathfinding() {
		if (simulationStarted) return;

		if (!isGenerated) {
			std::cout << "SCENE: Cannot test pathfinding - PRM is not generated" << std::endl;
			return;
		}

		double startTime;

		// UCS
		startTime = glfwGetTimerValue();
		if (!prm.findPath(false)) {
			std::cout << "SCENE: Cannot test pathfinding - no path exists" << std::endl;
			return;
		}
		std::cout << "UCS pathfinding time: " <<
			(double)(glfwGetTimerValue() - startTime)/glfwGetTimerFrequency() <<
			"s" << std::endl;

		// A*
		startTime = glfwGetTimerValue();
		if (!prm.findPath(true)) {
			std::cout << "SCENE: Cannot test pathfinding - A* is broken" << std::endl;
			return;
		}
		std::cout << "A*  pathfinding time: " <<
			(double)(glfwGetTimerValue() - startTime)/glfwGetTimerFrequency() <<
			"s" << std::endl;
	}

	void updatePRM(int n = -1) {
		if (simulationStarted) return;

		prm.generatePRM(n);
		prm.updateEdges();

		isGenerated = true;
		hasPath = prm.findPath();
	}

	void setStart(float x, float y) {
		setStart(Vec3f(x, y, 0.0));
	}

	void setStart(Vec3f start) {
		if (simulationStarted) return;

		prm.setStart(start);
		prm.updateEdges();

		agent->moveTo(start);
		curTargetInPath = 0;

		simulationComplete = false;
		
		if (isGenerated)
			hasPath = prm.findPath();
	}

	void setGoal(float x, float y) {
		setGoal(Vec3f(x, y, 0.0));
	}

	void setGoal(Vec3f goal) {
		if (simulationStarted) return;

		prm.setGoal(goal);
		prm.updateEdges();

		simulationComplete = false;

		if (isGenerated)
			hasPath = prm.findPath();
	}

	void setBounds(Vec3f min, Vec3f max) {
		prm.setBounds(min, max);
	}

	bool simulate(double dt) {
		if (placingObject || placingStart || placingGoal) {
			std::cout << "SCENE: Cannot simulate while placing object." << std::endl;
			return false;
		}
		if (simulationComplete) {
			std::cout << "SCENE: Simulation must be reset." << std::endl;
			return false;
		}
		if (!isGenerated) {
			std::cout << "SCENE: PRM has not been updated." << std::endl;
			return false;
		}
		if (!hasPath) {
			std::cout << "SCENE: Scene does not have a path." << std::endl;
			return false;
		}

		simulationStarted = true;

		if (prm.reachedGoal(agent, curTargetInPath, agentSpeed*dt)) {
			simulationComplete = true;
			std::cout << "Simulation has finished." << std::endl;

			agent->pos = prm.getPointInPath(curTargetInPath-1);

			return true;
		}

		if (prm.nextPointInPathIsVisible(agent, curTargetInPath+1)) {
			curTargetInPath++;
			agentDirection = prm.getPointInPath(curTargetInPath) - agent->pos;
			agentDirection.normalize();
		}

		agent->moveBy(agentDirection*agentSpeed*dt);
		
		return true;
	}

	void render() {
		prm.render();
		prm.renderPath();
		agent->render();

		if (!renderCObstacles)
		for (int i = 0; i < objects.size(); i++) {
			objects[i]->render();
		}
		else
			prm.renderCObstacles();

		renderCurrentObject();
	}

	void renderCurrentObject() {
		if (placingObject) {
			currentObject->render();
		}
		else if (placingStart || placingGoal) {
			using namespace Globals;
			Vec3f color;

			if (placingStart) {
				color = Vec3f(0, 1, 1);
			}
			else {
				color = Vec3f(1, 0, 1);
			}
			glBindBuffer(GL_ARRAY_BUFFER, position_vbo[0]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3f), &currentPoint, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glBindBuffer(GL_ARRAY_BUFFER, colors_vbo[0]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3f), &color, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glBindBuffer(GL_ARRAY_BUFFER, normals_vbo[0]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(Vec3f), &Vec3f(0, 0, 0), GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glUniform1i(curShader.uniform("drawing_points"), 1);
			glUniform1i(curShader.uniform("vertex_type"), 3);
			glDrawArrays(GL_POINTS, 0, 1);
			glUniform1i(curShader.uniform("drawing_points"), 0);
			glUniform1i(curShader.uniform("vertex_type"), 0);
		}
	}

	Object *agent;
	vector<Object *> objects;
	PRM prm;

	bool agentIsSphere = true;
	int curTargetInPath; // index of point in PRM's path vector that is being moved towards
	Vec3f agentDirection;
	float agentSpeed = 7.0;

	bool hasPath = false;
	bool isGenerated = false;
	bool simulationStarted = false;
	bool simulationComplete = false;

	Object *currentObject;
	Vec3f currentPoint;
	bool placingObject = false;
	bool placingStart = false;
	bool placingGoal = false;
	bool renderCObstacles = false;
};

namespace Globals {
	Scene s;
}

void updateViewProjection() {
	using namespace Globals;

	// Calculate the orthogonal axes based on the viewing parameters
	Vec3f n = viewDir * (-1.f / viewDir.len());
	Vec3f u = upDir.cross(n);
	u.normalize();
	Vec3f v = n.cross(u);

	// Calculate the translation based on the new axes
	float dx = -(eye.dot(u));
	float dy = -(eye.dot(v));
	float dz = -(eye.dot(n));

	// Fill in the matrix
	view.m[0] = u[0];	view.m[4] = u[1];	view.m[8] = u[2];	view.m[12] = dx;
	view.m[1] = v[0];	view.m[5] = v[1];	view.m[9] = v[2];	view.m[13] = dy;
	view.m[2] = n[0];	view.m[6] = n[1];	view.m[10] = n[2];	view.m[14] = dz;
	view.m[3] = 0;		view.m[7] = 0;		view.m[11] = 0;		view.m[15] = 1;
}

//
//	Callbacks
//
static void error_callback(int error, const char* description){ fprintf(stderr, "Error: %s\n", description); }

// function that is called whenever a mouse or trackpad button press event occurs
static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwGetCursorPos(window, &Globals::mouse.prev_x, &Globals::mouse.prev_y);
		Globals::mouse.active = true;
	}
}

// Function to rotate the viewing transform about the z axis
static const Mat4x4 rotateZ(float theta) {
	float t = theta*PI/180.f;

	Mat4x4 mat;
	mat.m[0] =  cos(t);		mat.m[4] = -sin(t);	mat.m[8] = 0.f;			mat.m[12] = 0.f;
	mat.m[1] = 	sin(t);		mat.m[5] = cos(t);	mat.m[9] = 0.f;			mat.m[13] = 0.f;
	mat.m[2] = 0.f;			mat.m[6] = 0.f;		mat.m[10] = 1.f;		mat.m[14] = 0.f;
	mat.m[3] = 0.f;			mat.m[7] = 0.f;		mat.m[11] = 0.f;		mat.m[15] = 1.f;

	return mat;
}

// Function to rotate the viewing transform about the y axis
static const Mat4x4 rotateY(float theta) {
	float t = theta*PI/180.f;
	
	Mat4x4 mat;
	mat.m[0] = cos(t);		mat.m[4] = 0.f;		mat.m[8] = sin(t);		mat.m[12] = 0.f;
	mat.m[1] = 0.f;			mat.m[5] = 1.f;		mat.m[9] = 0.f;			mat.m[13] = 0.f;
	mat.m[2] = -sin(t);		mat.m[6] = 0.f;		mat.m[10] = cos(t);		mat.m[14] = 0.f;
	mat.m[3] = 0.f;			mat.m[7] = 0.f;		mat.m[11] = 0.f;		mat.m[15] = 1.f;
	
	return mat;
}

// Function to rotate the viewing transform about the y axis
static const Mat4x4 rotateX(float phi) {
	float t = phi*PI/180.f;
	
	Mat4x4 mat;
	mat.m[0] = 1.f;		mat.m[4] = 0.f;		mat.m[8] = 0.f;			mat.m[12] = 0.f;
	mat.m[1] = 0.f;		mat.m[5] = cos(t);	mat.m[9] = -sin(t);		mat.m[13] = 0.f;
	mat.m[2] = 0.f;		mat.m[6] = sin(t);	mat.m[10] = cos(t);		mat.m[14] = 0.f;
	mat.m[3] = 0.f;		mat.m[7] = 0.f;		mat.m[11] = 0.f;		mat.m[15] = 1.f;
	
	return mat;
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
	using namespace Globals;
	if (!mouse.active)
		return;

	viewDir = Vec3f(0, 0, -1);
	upDir = Vec3f(0, 1, 0);
	
	/*
	if (xpos != mouse.prev_x) {	
		theta -= 0.2*(xpos - mouse.prev_x);
		yRot = rotateY(theta);
		mouse.prev_x = xpos;
	}

	if (ypos != mouse.prev_y) {
		phi -= 0.2*(ypos - mouse.prev_y);
		if (phi > 89)
			phi = 89;
		else if (phi < -89)
			phi = -89;
		xRot = rotateX(phi);
		mouse.prev_y = ypos;
	}
	*/
	// Changing view controls to suit the simulation better
	if (xpos != mouse.prev_x) {
		theta -= 0.2*(xpos - mouse.prev_x);
		yRot = rotateZ(theta);
		mouse.prev_x = xpos;
	}

	if (ypos != mouse.prev_y) {
		phi -= 0.2*(ypos - mouse.prev_y);
		if (phi > 180)
			phi = 180;
		else if (phi < 0)
			phi = 0;
		xRot = rotateX(phi);
		mouse.prev_y = ypos;
	}
	
	viewDir = xRot*viewDir;
	viewDir = yRot*viewDir;

	upDir = xRot*upDir;
	upDir = yRot*upDir;

	rightDir = upDir.cross(viewDir);
}


static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
	using namespace Globals;

	bool smooth;

	// Close on escape or Q
	if (action == GLFW_PRESS) {
		switch (key) {
			// Close on escape
		case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GL_TRUE); break;
			// Pause the simulation
		case GLFW_KEY_SPACE: if (pause) pause = false;
							 else pause = true;
							 std::cout << ((pause) ? "Paused" : "Unpaused") << std::endl;
							 break;

			// Movement keys trigger booleans to be processed during the graphics loop
			// Forward movement
		case GLFW_KEY_W: key_w = true; break;

			// Backward movement
		case GLFW_KEY_S: key_s = true; break;

			// Right strafing movement
		case GLFW_KEY_D: key_d = true; break;

			// Left strafing movement
		case GLFW_KEY_A: key_a = true; break;

			// Upward movement
		case GLFW_KEY_E: key_e = true; break;

			// Downward movement
		case GLFW_KEY_Q: key_q = true; break;

			// Speed up
		case GLFW_KEY_LEFT_SHIFT: key_lshift = true; break;

			// Release mouse
		case GLFW_KEY_KP_ENTER:
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			mouse.active = false;
			break;

			// Begin placing object
		case GLFW_KEY_ENTER:
			s.placeObject();
			break;

		case GLFW_KEY_HOME:
			s.placeStart();
			break;

		case GLFW_KEY_END:
			s.placeGoal();
			break;

		case GLFW_KEY_DELETE:
			s.cancelPlacing();
			break;

		case GLFW_KEY_KP_0:
			s.changeShape();
			break;

			// Increase/decrease object size
		case GLFW_KEY_KP_2: key_num2 = true; break;
		case GLFW_KEY_KP_4: key_num4 = true; break;
		case GLFW_KEY_KP_6: key_num6 = true; break;
		case GLFW_KEY_KP_8: key_num8 = true; break;
			// Move object
		case GLFW_KEY_UP: key_up = true; break;
		case GLFW_KEY_DOWN: key_down = true; break;
		case GLFW_KEY_LEFT: key_left = true; break;
		case GLFW_KEY_RIGHT: key_right = true; break;

			// Render the CObstacles instead of the regular obstacles
		case GLFW_KEY_TAB:
			s.renderCObstacles = !s.renderCObstacles;
			break;
		}

	}
	else if ( action == GLFW_RELEASE ) {
		switch ( key ) {
			// Movement keys trigger booleans to be processed during the graphics loop
			// Forward movement
		case GLFW_KEY_W: key_w = false; break;

			// Backward movement
		case GLFW_KEY_S: key_s = false; break;

			// Right strafing movement
		case GLFW_KEY_D: key_d = false; break;

			// Left strafing movement
		case GLFW_KEY_A: key_a = false; break;

			// Upward movement
		case GLFW_KEY_E: key_e = false; break;

			// Downward movement
		case GLFW_KEY_Q: key_q = false; break;

			// Speed up
		case GLFW_KEY_LEFT_SHIFT: key_lshift = false; break;

			// Increase/decrease object size
		case GLFW_KEY_KP_2: key_num2 = false; break;
		case GLFW_KEY_KP_4: key_num4 = false; break;
		case GLFW_KEY_KP_6: key_num6 = false; break;
		case GLFW_KEY_KP_8: key_num8 = false; break;
			// Move object
		case GLFW_KEY_UP: key_up = false; break;
		case GLFW_KEY_DOWN: key_down = false; break;
		case GLFW_KEY_LEFT: key_left = false; break;
		case GLFW_KEY_RIGHT: key_right = false; break;
		}
	}
}

void updatePerspectiveProjection() {
	using namespace Globals;

	for (int i = 0; i < 15; i++) {
		projection.m[i] = 0;
	}
	left = aspect * bottom;
	right = aspect * top;
	//diagonal values done first
	projection.m[0] = 2 * near / (right - left);
	projection.m[5] = 2 * near / (top - bottom);
	projection.m[10] = -(near + far) / (far - near);
	projection.m[15] = 0;
	//other values are then calculated.
	projection.m[8] = (right + left) / (right - left);
	projection.m[9] = (top + bottom) / (top - bottom);
	projection.m[14] = -2 * far*near / (far - near);
	projection.m[11] = -1;
}

static void framebuffer_size_callback(GLFWwindow* window, int width, int height){
	Globals::win_width = float(width);
	Globals::win_height = float(height);
    Globals::aspect = Globals::win_width/Globals::win_height;
	
    glViewport(0,0,width,height);

	// ToDo: update the perspective matrix according to the new window size
	updatePerspectiveProjection();
}

void init_scene();

//
//	Main
//
int main(int argc, char *argv[]){

	// Set up window
	GLFWwindow* window;
	glfwSetErrorCallback(&error_callback);

	// Initialize the window
	if( !glfwInit() ){ return EXIT_FAILURE; }

	// Ask for OpenGL 3.2
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	// Create the glfw window
	Globals::win_width = WIN_WIDTH;
	Globals::win_height = WIN_HEIGHT;
	window = glfwCreateWindow(int(Globals::win_width), int(Globals::win_height), "Motion Planning", NULL, NULL);
	if( !window ){ glfwTerminate(); return EXIT_FAILURE; }
	Globals::activeWindow = window;
	// Bind callbacks to the window
	glfwSetKeyCallback(window, &key_callback);
	glfwSetFramebufferSizeCallback(window, &framebuffer_size_callback);

	// Make current
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	// Initialize glew AFTER the context creation and before loading the shader.
	// Note we need to use experimental because we're using a modern version of opengl.
	#ifdef USE_GLEW
		glewExperimental = GL_TRUE;
		glewInit();
	#endif

	// Initialize the shader (which uses glew, so we need to init that first).
	// MY_SRC_DIR is a define that was set in CMakeLists.txt which gives
	// the full path to this project's src/ directory.
	std::stringstream ss; ss << MY_SRC_DIR << "shader.";
	Globals::curShader.init_from_files( ss.str()+"vert", ss.str()+"frag" );

	// Initialize the scene
	// IMPORTANT: Only call after gl context has been created
	init_scene();
	framebuffer_size_callback(window, int(Globals::win_width), int(Globals::win_height));

	// Enable the shader, this allows us to set uniforms and attributes
	Globals::curShader.enable();
	glBindVertexArray(Globals::scene_vao);

	// Initialize OpenGL
	glEnable(GL_DEPTH_TEST);
	glClearColor(1.0, 1.0, 1.0, 1.f);

	updatePerspectiveProjection();
	updateViewProjection();

	double timePassed = 0;
	double dt = 0;
	int frames = 0;
	double counter = 0;
	int seconds = 0;

	using namespace Globals;

	int framesPassed = 0;

	float SIZERATE = 6;
	float MOVERATE = 6;
	float dObjectSize1 = 0;
	float dObjectSize2 = 0;
	Vec3f objectMovement = Vec3f();
	bool holdPRMUpdate = false;

	// Game loop
	while (!glfwWindowShouldClose(window)) {

		framesPassed++;

		// Clear screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		prevTime = curTime;
		curTime = glfwGetTimerValue();
		timePassed = (double)(curTime - prevTime) / glfwGetTimerFrequency();
		
		// Prevents the simulation from destabilizing due to computation freeze (e.g. dragging the window)
		if (DEBUG && timePassed >= MAXDT) {
			std::cout << "Update cycle past " << MAXDT << "s, refreshing..." << std::endl;
			continue;
		}
		dt = timePassed*timeMultiplier;

		// SIMULATION UPDATE
		if (!pause && !s.simulate(dt)) {
			pause = true;
		}

		// INPUT PROCESSING
		if (key_lshift)
			movementSpeed += .01;
		else
			movementSpeed = .1;

		if (key_w) // Move the camera forward
			eye += viewDir*movementSpeed;
		if (key_s) // Move the camera backward
			eye += viewDir*(-movementSpeed);
		if (key_a) // Move the camera leftward
			eye += rightDir*movementSpeed;
		if (key_d) // Move the camera rightward
			eye += rightDir*(-movementSpeed);
		if (key_e) // Move the camera upward
			eye += upDir*movementSpeed;
		if (key_q) // Move the camera downward
			eye += upDir*(-movementSpeed);

		dObjectSize1 = 0;
		dObjectSize2 = 0;
		objectMovement = Vec3f();
		if (key_num2)
			dObjectSize2 -= SIZERATE;
		if (key_num4)
			dObjectSize1 -= SIZERATE;
		if (key_num6)
			dObjectSize1 += SIZERATE;
		if (key_num8)
			dObjectSize2 += SIZERATE;
		if (key_down)
			objectMovement[1] -= MOVERATE;
		if (key_left)
			objectMovement[0] -= MOVERATE;
		if (key_right)
			objectMovement[0] += MOVERATE;
		if (key_up)
			objectMovement[1] += MOVERATE;

		s.moveObject(objectMovement*dt);
		if (key_num2 || key_num4 || key_num6 || key_num8) {
			holdPRMUpdate = true;
			s.changeObjectSize1(dObjectSize1*dt);
			s.changeObjectSize2(dObjectSize2*dt);
		}
		else if (holdPRMUpdate) {
			holdPRMUpdate = false;
			s.updateCObstacles();
		}
		
		// Send updated info to the GPU
		updateViewProjection();

		// FRAME RATE DISPLAY
		frames++;
		counter += timePassed;
		if (counter >= 1.0) {
			//std::cout << "S" << seconds << " - ";
			//std::cout << "FPS: " << frames << std::endl;
			frames = 0;
			counter -= 1.0;
			seconds++;
		}

		// RENDERING
		//glUniformMatrix4fv( shader.uniform("model"), 1, GL_FALSE, model.m  ); // model transformation
		glUniformMatrix4fv(Globals::curShader.uniform("view"), 1, GL_FALSE, view.m); // viewing transformation
		glUniformMatrix4fv(Globals::curShader.uniform("projection"), 1, GL_FALSE, projection.m); // projection matrix
		glUniform3f(Globals::curShader.uniform("viewdir"), viewDir[0], viewDir[1], viewDir[2]);
		glUniform3f(Globals::curShader.uniform("light"), lightDir[0], lightDir[1], lightDir[2]);
		glUniform3f(curShader.uniform("eye"), eye[0], eye[1], eye[2]);

		s.render();

		// Finalize
		glfwSwapBuffers(window);
		glfwPollEvents();

		
	} // end game loop
	// Unbind
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	// Disable the shader, we're done using it
	Globals::curShader.disable();

	return EXIT_SUCCESS;
}


void init_scene(){
	using namespace Globals;

	// Define the keyboard callback function
	glfwSetKeyCallback(activeWindow, key_callback);
	// Define the mouse button callback function
	glfwSetMouseButtonCallback(activeWindow, mouse_button_callback);
	// Define the mouse motion callback function
	glfwSetCursorPosCallback(activeWindow, cursor_position_callback);
	viewDir = Vec3f(0, 0, -1);
	phi = 0;
	xRot = rotateX(phi);
	upDir = Vec3f(0, 1, 0);
	rightDir = Vec3f(-1, 0, 0);
	eye = Vec3f(0, 0, 12);
	lightDir = Vec3f(0, 0, -1);

	//s.testPathfinding();

	glGenVertexArrays(1, &scene_vao);
	glBindVertexArray(scene_vao);

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	glGenBuffers(1, position_vbo);
	// Create the buffer for colors
	glGenBuffers(1, colors_vbo);
	// Create buffer for normals
	glGenBuffers(1, normals_vbo);
	// Create buffer for faces
	glGenBuffers(1, faces_ibo);
	// Create the buffer for edges
	glGenBuffers(1, edges_ibo);

	//GLint glnormal = glGetAttribLocation(curShader.program_id, "normal");
	//particle position
	glEnableVertexAttribArray(curShader.attribute("in_position"));
	glBindBuffer(GL_ARRAY_BUFFER, position_vbo[0]);
	glVertexAttribPointer(curShader.attribute("in_position"), 3, GL_FLOAT, GL_FALSE, 0, 0);
	//particle color
	glEnableVertexAttribArray(curShader.attribute("in_color"));
	glBindBuffer(GL_ARRAY_BUFFER, colors_vbo[0]);
	glVertexAttribPointer(curShader.attribute("in_color"), 3, GL_FLOAT, GL_FALSE, 0, 0);
	//particle normal
	glEnableVertexAttribArray(curShader.attribute("in_normal"));
	glBindBuffer(GL_ARRAY_BUFFER, normals_vbo[0]);
	glVertexAttribPointer(curShader.attribute("in_normal"), 3, GL_FLOAT, GL_TRUE, 0, 0);

	glUniform3f(curShader.uniform("light"), lightDir[0], lightDir[1], lightDir[2]);
	glUniform3f(curShader.uniform("viewdir"), viewDir[0], viewDir[1], viewDir[2]);
	glUniform3f(curShader.uniform("eye"), eye[0], eye[1], eye[2]);
	glUniform1i(curShader.uniform("vertex_type"), 0);
	glUniform1i(curShader.uniform("drawing_points"), 0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//done setting data for VAO
	glBindVertexArray(0);
	curTime = glfwGetTimerValue();
}
