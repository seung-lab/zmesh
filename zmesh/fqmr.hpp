#ifndef __ZMESH_FQMR_HPP__
#define __ZMESH_FQMR_HPP__

/////////////////////////////////////////////
//
// Mesh Simplification
//
// (C) by Sven Forstmann in 2014
//
// License : MIT
// http://opensource.org/licenses/MIT
//
//https://github.com/sp4cerat/Fast-Quadric-Mesh-S;
// 5/2016: Chris Rorden created minimal version for OSX/Linux/Windows compile
// 3/2026: William Silversmith forked from https://github.com/Kramer84/pyfqmr-Fast-Quadric-Mesh-Reduction
//         dropped texture mapping and focused on memory pressure

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "utility.hpp"

using namespace zmesh::utility;

#define loopi(start_l,end_l) for (auto i = start_l; i < end_l; ++i)
#define loopj(start_l,end_l) for (auto j = start_l; j < end_l; ++j)
#define loopk(start_l,end_l) for (auto k = start_l; k < end_l; ++k)

namespace zmesh::fqmr {

constexpr size_t ZERO = 0;

double min(double v1, double v2) {
	return fmin(v1,v2);
}

class SymmetricMatrix {
	public:
	float m[10];

	SymmetricMatrix(float c=0) { loopi(0,10) m[i] = c;  }

	SymmetricMatrix(
		float m11, float m12, float m13, float m14,
					float m22, float m23, float m24,
								float m33, float m34,
											float m44
	) {
		m[0] = m11;  m[1] = m12;  m[2] = m13;  m[3] = m14;
								 m[4] = m22;  m[5] = m23;  m[6] = m24;
								 							m[7] = m33;  m[8] = m34;
																					 m[9] = m44;
	}

	// Make plane
	SymmetricMatrix(
		const float a, 
		const float b,
		const float c,
		const float d
	) {
		m[0] = a*a;  m[1] = a*b;  m[2] = a*c;  m[3] = a*d;
								 m[4] = b*b;  m[5] = b*c;  m[6] = b*d;
														  m[7] = c*c;  m[8] = c*d;
																					 m[9] = d*d;
	}

	double operator[](int c) const { return m[c]; }

	// Determinant
	double det(
		const int a11, const int a12, const int a13,
		const int a21, const int a22, const int a23,
		const int a31, const int a32, const int a33
	){
		// less likely to over or underflow if we alternate
		// additions and subtractions
		return static_cast<double>(
			(m[a11]*m[a22]*m[a33])
			- (m[a13]*m[a22]*m[a31])
			+ (m[a13]*m[a21]*m[a32])
			- (m[a11]*m[a23]*m[a32])
			+ (m[a12]*m[a23]*m[a31])
			- (m[a12]*m[a21]*m[a33])
		);
	}

	const SymmetricMatrix operator+(const SymmetricMatrix& n) const {
		return SymmetricMatrix(
			m[0]+n[0],   m[1]+n[1],   m[2]+n[2],   m[3]+n[3],
			m[4]+n[4],   m[5]+n[5],   m[6]+n[6],
			m[7]+n[7],   m[8]+n[8],
			m[9]+n[9]
		);
	}

	SymmetricMatrix& operator+=(const SymmetricMatrix& n) {
		m[0]+=n[0];   m[1]+=n[1];   m[2]+=n[2];   m[3]+=n[3];
		m[4]+=n[4];   m[5]+=n[5];   m[6]+=n[6];   m[7]+=n[7];
		m[8]+=n[8];   m[9]+=n[9];
		return *this;
	}
};
///////////////////////////////////////////

// Global Variables & Structures
struct Triangle { 
	uint32_t v[3];
	float err[4];
	bool deleted;
	bool dirty;
	Vec3<float> n;
};

struct Vertex { 
	Vec3<float> p;
	uint32_t tstart;
	uint32_t tcount;
	SymmetricMatrix q;
	bool border;
};

// Pairs triangle index and vertex slot (0/1/2) within that triangle.
// The two arrays are always the same length and indexed together,
// so they are kept as a single struct to make that invariant explicit.
struct RefList {
	std::vector<uint32_t> tris;  // triangle index
	std::vector<uint8_t>  verts; // vertex slot within that triangle (0, 1, or 2)

	void resize(size_t n) {
		tris.resize(n);
		verts.resize(n);
	}

	void reserve(size_t n) {
		tris.reserve(n);
		verts.reserve(n);
	}

	size_t size() const {
		return tris.size();
	}

	void push_back(uint32_t tri, uint8_t vert) {
		tris.push_back(tri);
		verts.push_back(vert);
	}
};

struct FqmrMesh {
	std::vector<Triangle> triangles;
	std::vector<Vertex> vertices;

	void set(
		const std::span<const float>& verts, 
		const std::span<const uint32_t>& faces
	){
		vertices.clear();
		triangles.clear();

		uint64_t N_vertices = verts.size() / 3;
		uint64_t N_faces = faces.size() / 3;
		
		vertices.reserve(N_vertices);
		triangles.reserve(N_faces);

		for (uint64_t i = 0; i < N_vertices * 3; i += 3) {
			Vertex v;
			v.p.x = verts[i];
			v.p.y = verts[i+1];
			v.p.z = verts[i+2];
			vertices.push_back(v);
		}
		for (uint64_t i = 0; i < N_faces * 3; i += 3) {
			Triangle t;
			t.v[0] = faces[i];
			t.v[1] = faces[i+1];
			t.v[2] = faces[i+2];
			triangles.push_back(t);
		}
	}

	void set(
		const float* verts, const uint64_t Nv,
		const uint32_t* faces, const uint64_t Nf
	) {
		set(
			std::span(verts, Nv * 3),
			std::span(faces, Nf * 3)
		);
	}

	std::vector<float> getVertices() const {
		std::vector<float> verts;
		const uint64_t N_vertices = vertices.size();
		verts.reserve(N_vertices * 3);
		for (uint64_t i = 0; i < N_vertices; i++) {
			verts.push_back(vertices[i].p.x);
			verts.push_back(vertices[i].p.y);
			verts.push_back(vertices[i].p.z);
		}
		return verts;
	}

	std::vector<uint32_t> getFaces() const {
		std::vector<uint32_t> faces;
		const uint64_t N_faces = triangles.size();
		faces.reserve(N_faces * 3);
		for (uint64_t i = 0; i < N_faces; i++) {
			faces.push_back(triangles[i].v[0]);
			faces.push_back(triangles[i].v[1]);
			faces.push_back(triangles[i].v[2]);
		}
		return faces;
	}

	std::vector<float> getNormals() {
		std::vector<float> normals;
		const uint64_t N_faces = triangles.size();
		normals.reserve(N_faces * 3);
		for (uint64_t i = 0; i < N_faces; i++) {
			normals.push_back(triangles[i].n.x);
			normals.push_back(triangles[i].n.y);
			normals.push_back(triangles[i].n.z);
		}
		return normals;
	}

	void clear() {
		triangles = std::vector<Triangle>();
		vertices = std::vector<Vertex>();
	}
};

// Error between vertex and Quadric
double vertex_error(
	const SymmetricMatrix& q,
	const double x,
	const double y,
	const double z
) {
	return (
		q[0]*x*x 
		+ 2*q[1]*x*y 
		+ 2*q[2]*x*z 
		+ 2*q[3]*x 
		+ q[4]*y*y
		+ 2*q[5]*y*z
		+ 2*q[6]*y
		+ q[7]*z*z 
		+ 2*q[8]*z 
		+ q[9]
	);
}

// Error for one edge
double calculate_error(
	const std::vector<Vertex>& vertices,
	uint32_t id_v1, 
	uint32_t id_v2, 
	Vec3<float> &p_result
) {
	// compute interpolated vertex
	SymmetricMatrix q = vertices[id_v1].q;
	q += vertices[id_v2].q;
	bool border = vertices[id_v1].border && vertices[id_v2].border;
	double error = 0;
	double det = q.det(0, 1, 2, 1, 4, 5, 2, 5, 7);
	if (det != 0 && !border) {
		// q_delta is invertible
		p_result.x = -1/det*(q.det(1, 2, 3, 4, 5, 6, 5, 7 , 8));  // vx = A41/det(q_delta)
		p_result.y =  1/det*(q.det(0, 2, 3, 1, 5, 6, 2, 7 , 8));  // vy = A42/det(q_delta)
		p_result.z = -1/det*(q.det(0, 1, 3, 1, 4, 6, 2, 5,  8));  // vz = A43/det(q_delta)
		error = vertex_error(q, p_result.x, p_result.y, p_result.z);
	}
	else {
		// det = 0 -> try to find best result
		Vec3<float> p1 = vertices[id_v1].p;
		Vec3<float> p2 = vertices[id_v2].p;
		Vec3<float> p3 = (p1+p2)/2;
		double error1 = vertex_error(q, p1.x,p1.y,p1.z);
		double error2 = vertex_error(q, p2.x,p2.y,p2.z);
		double error3 = vertex_error(q, p3.x,p3.y,p3.z);
		error = min(error1, min(error2, error3));
		if (error1 == error) p_result = p1;
		if (error2 == error) p_result = p2;
		if (error3 == error) p_result = p3;
	}
	return (double)error;
}


// compact triangles, compute edge error and build reference list
void update_mesh(
	std::vector<Triangle>& triangles,
	std::vector<Vertex>& vertices,
	RefList& refs,
	int iteration
) {
	if (iteration > 0) { // compact triangles
		int dst = 0;
		loopi(ZERO, triangles.size()) {
			if(!triangles[i].deleted) {
				triangles[dst++] = triangles[i];
			}
		}
		triangles.resize(dst);
	}

	// Init Reference ID list
	loopi(ZERO, vertices.size()) {
		vertices[i].tstart = 0;
		vertices[i].tcount = 0;
	}
	loopi(ZERO, triangles.size()) {
		Triangle &t = triangles[i];
		loopj(0,3) {
			vertices[t.v[j]].tcount++;
		}
	}

	uint32_t tstart = 0;
	loopi(ZERO, vertices.size()) {
		Vertex &v = vertices[i];
		v.tstart = tstart;
		tstart += v.tcount;
		v.tcount = 0;
	}

	// Write References
	refs.resize(triangles.size()*3);

	loopi(ZERO, triangles.size()) {
		Triangle &t = triangles[i];
		loopj(0,3) {
			Vertex &v = vertices[t.v[j]];
			refs.tris[v.tstart+v.tcount]  = i;
			refs.verts[v.tstart+v.tcount] = j;
			v.tcount++;
		}
	}

	// Identify boundary
	if (iteration == 0) {
		std::vector<uint32_t> vcount, vids;

		loopi(ZERO, vertices.size()) {
			Vertex &v = vertices[i];
			vcount.clear();
			vids.clear();
			
			for (uint32_t j = 0; j < v.tcount; j++) {
				uint32_t k = refs.tris[v.tstart+j];
				Triangle &t = triangles[k];
				loopk(0,3) {
					size_t ofs = 0;
					uint32_t id = t.v[k];
					while (ofs < vcount.size()) {
						if (vids[ofs] == id) {
							break;
						}
						ofs++;
					}
					if (ofs == vcount.size()) {
						vcount.push_back(1);
						vids.push_back(id);
					}
					else {
						vcount[ofs]++;
					}
				}
			}

			loopj(ZERO, vcount.size()) {
				vertices[vids[j]].border = (vcount[j] == 1);
			}
		}
	}

	//
	// Init Quadrics by Plane & Edge Errors
	//
	// required at the beginning ( iteration == 0 )
	// recomputing during the simplification is not required,
	// but mostly improves the result for closed meshes
	//
	if (iteration == 0) {
		loopi(ZERO, vertices.size()) {
			vertices[i].q = SymmetricMatrix(0.0);
		}

		loopi(ZERO, triangles.size()) {
			Triangle &t = triangles[i];
			Vec3<float> p[3];
			loopj(0,3) {
				p[j] = vertices[t.v[j]].p;
			}
			Vec3<float> n = (p[1]-p[0]).cross(p[2]-p[0]).hat();
			t.n = n;
			loopj(0,3) {
				vertices[t.v[j]].q += SymmetricMatrix(n.x,n.y,n.z,-n.dot(p[0]));
			}
		}
		loopi(ZERO, triangles.size()) {
			// Calc Edge Error
			Triangle &t = triangles[i];Vec3<float> p;
			loopj(0,3) t.err[j] = calculate_error(vertices, t.v[j], t.v[(j+1)%3], p);
			t.err[3] = min(t.err[0],min(t.err[1],t.err[2]));
		}
	}
}

// Finally compact mesh before exiting
void compact_mesh(
	std::vector<Triangle>& triangles,
	std::vector<Vertex>& vertices
) {
	int dst = 0;
	loopi(ZERO, vertices.size()) {
		vertices[i].tcount = 0;
	}
	loopi(ZERO, triangles.size()) {
		if(!triangles[i].deleted) {
			Triangle &t = triangles[i];
			triangles[dst++] = t;
			loopj(0,3) {
				vertices[t.v[j]].tcount = 1;
			}
		}
	}

	triangles.resize(dst);
	dst = 0;
	loopi(ZERO, vertices.size()) {
		if(vertices[i].tcount) {
			vertices[i].tstart=dst;
			vertices[dst].p=vertices[i].p;
			dst++;
		}
	}

	loopi(ZERO, triangles.size()) {
		Triangle &t = triangles[i];
		loopj(0,3)t.v[j] = vertices[t.v[j]].tstart;
	}
	vertices.resize(dst);
}

// Check if a triangle flips when this edge is removed
bool flipped(
	const std::vector<Triangle>& triangles,
	const std::vector<Vertex>& vertices,
	const RefList& refs,
	Vec3<float> p,
	uint32_t i0,
	uint32_t i1,
	Vertex& v0,
	Vertex& v1,
	std::vector<int>& deleted
) {
	for (uint32_t k = 0; k < v0.tcount; k++) {
		const Triangle &t = triangles[refs.tris[v0.tstart+k]];
		if (t.deleted) continue;

		auto s = refs.verts[v0.tstart+k];
		auto id1 = t.v[(s+1)%3];
		auto id2 = t.v[(s+2)%3];

		if (id1 == i1 || id2 == i1) { // delete ?
			deleted[k] = 1;
			continue;
		}
		Vec3<float> d1 = vertices[id1].p-p;
		d1 = d1.hat();
		Vec3<float> d2 = vertices[id2].p-p;
		d2 = d2.hat();
		
		if (fabs(d1.dot(d2)) > 0.999) {
			return true;
		}

		Vec3<float> n = d1.cross(d2).hat();
		deleted[k] = 0;
		if (n.dot(t.n) < 0.2) {
			return true;
		}
	}

	return false;
}

// Update triangle connections and edge error after a edge is collapsed
void update_triangles(
	std::vector<Triangle>& triangles,
	const std::vector<Vertex>& vertices, 
	RefList& refs,
	uint32_t i0,
	Vertex &v,
	std::vector<int> &deleted,
	int64_t &deleted_triangles
) {
	Vec3<float> p;
	for (uint32_t k = 0; k < v.tcount; k++) {
		auto tid     = refs.tris[v.tstart+k];
		auto tvertex = refs.verts[v.tstart+k];

		Triangle &t = triangles[tid];
		if (t.deleted) {
			continue;
		}
		else if (deleted[k]) {
			t.deleted = 1;
			deleted_triangles++;
			continue;
		}

		t.v[tvertex] = i0;
		t.dirty = 1;
		t.err[0] = calculate_error(vertices, t.v[0], t.v[1], p);
		t.err[1] = calculate_error(vertices, t.v[1], t.v[2], p);
		t.err[2] = calculate_error(vertices, t.v[2], t.v[0], p);
		t.err[3] = min(t.err[0], min(t.err[1],t.err[2]));
		
		refs.push_back(tid, tvertex);
	}
}

// Main simplification function
//
// target_count  : target nr. of triangles
// aggressiveness : sharpness to increase the threshold.
//                 5..8 are good numbers
//                 more iterations yield higher quality
//
int simplify_to_triangle_count(
	FqmrMesh& mesh,
	int64_t target_count, 
	int update_rate = 5, 
	double aggressiveness = 7,
	int max_iterations = 100,
	double alpha = 0.000000001,
	int K = 3,
	bool lossless = false,
	double threshold_lossless = 0.0001,
	bool preserve_border = false
) {
	RefList refs;

	auto& triangles = mesh.triangles;
	auto& vertices = mesh.vertices;

	loopi(ZERO, triangles.size()) {
		triangles[i].deleted = 0;
	}

	// main iteration loop
	int64_t deleted_triangles = 0;
	std::vector<int> deleted0, deleted1;
	int64_t triangle_count = triangles.size();

	int iteration;
	for (iteration = 0; iteration < max_iterations; iteration++) {
		if (triangle_count - deleted_triangles <= target_count) {
			break;
		}

		// update mesh once in a while
		if ((iteration % update_rate == 0) || lossless) {
			update_mesh(triangles, vertices, refs, iteration);
		}

		loopi(ZERO, triangles.size()) {
			triangles[i].dirty = 0;
		}

		//
		// All triangles with edges below the threshold will be removed
		//
		// The following numbers works well for most models.
		// If it does not, try to adjust the 3 parameters
		//
		double threshold = alpha * pow(double(iteration+K), aggressiveness);
		if (lossless) {
			threshold = threshold_lossless;
		}

		// remove vertices & mark deleted triangles
		loopi(ZERO, triangles.size()) {
			Triangle &t = triangles[i];
			if (t.err[3] > threshold) continue;
			if (t.deleted) continue;
			if (t.dirty) continue;

			loopj(0,3) {
				if (t.err[j] < threshold) {
					auto i0 = t.v[ j     ];
					Vertex &v0 = vertices[i0];
					auto i1 = t.v[(j+1)%3];
					Vertex &v1 = vertices[i1];

					// Border check //Added preserve_border method from issue 14
					if (preserve_border) {
						if (v0.border || v1.border) continue; // should keep border vertices
					}
					else if (v0.border != v1.border) {
						continue; // base behaviour
					}

					// Compute vertex to collapse to
					Vec3<float> p;
					calculate_error(vertices, i0,i1,p);
					deleted0.resize(v0.tcount); // normals temporarily
					deleted1.resize(v1.tcount); // normals temporarily
					// don't remove if flipped
					if (flipped(triangles,vertices,refs,p,i0,i1,v0,v1,deleted0)) {
						continue;
					}
					if (flipped(triangles,vertices,refs,p,i1,i0,v1,v0,deleted1)) {
						continue;
					}

					// not flipped, so remove edge
					v0.p = p;
					v0.q = v1.q + v0.q;
					auto tstart = refs.size();

					update_triangles(triangles,vertices,refs,i0,v0,deleted0,deleted_triangles);
					update_triangles(triangles,vertices,refs,i0,v1,deleted1,deleted_triangles);

					auto tcount = refs.size() - tstart;

					if (tcount <= v0.tcount) {
						// save ram
						if (tcount) {
							memcpy(&refs.tris[v0.tstart],  &refs.tris[tstart],  tcount * sizeof(uint32_t));
							memcpy(&refs.verts[v0.tstart], &refs.verts[tstart], tcount * sizeof(uint8_t));
						}
					}
					else {
						// append
						v0.tstart = tstart;
					}

					v0.tcount=tcount;
					break;
				}
			}

			// done?
			if (lossless && (deleted_triangles <= 0)) {
				break;
			} 
			else if (!lossless && (triangle_count-deleted_triangles <= target_count)) {
				break;
			}

			if (lossless) deleted_triangles = 0;
		}
	}

	compact_mesh(triangles, vertices);

	return iteration;
}

int simplify_below_threshold(
	FqmrMesh& mesh,
	double epsilon = 1e-3,
	int max_iterations = 9999,
	bool preserve_border = false
) {
	RefList refs;

	auto& triangles = mesh.triangles;
	auto& vertices = mesh.vertices;

	loopi(ZERO, triangles.size()) {
		triangles[i].deleted = 0;
	}
	// main iteration loop
	int64_t deleted_triangles = 0;
	std::vector<int> deleted0, deleted1;

	int iteration;
	for (iteration = 0; iteration < max_iterations; iteration++) {
		
		update_mesh(triangles, vertices, refs, iteration);
		
		loopi(ZERO, triangles.size()) {
			triangles[i].dirty = 0;
		}
		
		// All triangles with edges below the threshold will be removed
		//
		// The following numbers works well for most models.
		// If it does not, try to adjust the 3 parameters
		double threshold = epsilon;

		// remove vertices & mark deleted triangles
		loopi(ZERO, triangles.size()) {
			
			Triangle &t = triangles[i];
			if (t.err[3]>threshold) {
				continue;
			}
			else if (t.deleted) { 
				continue; 
			}
			else if (t.dirty) {
				continue;
			}

			loopj(0,3) {
				if (t.err[j] < threshold) {
					int i0=t.v[ j     ]; Vertex &v0 = vertices[i0];
					int i1=t.v[(j+1)%3]; Vertex &v1 = vertices[i1];

					// Border check //Added preserve_border method from issue 14 for lossless
					if (preserve_border) {
						if (v0.border || v1.border) continue; // should keep border vertices
					}
					else {
						if (v0.border != v1.border) continue; // base behaviour
					}

					// Compute vertex to collapse to
					Vec3<float> p;
					calculate_error(vertices, i0, i1, p);

					deleted0.resize(v0.tcount); // normals temporarily
					deleted1.resize(v1.tcount); // normals temporarily

					// don't remove if flipped
					if (flipped(triangles,vertices,refs,p,i0,i1,v0,v1,deleted0)) continue;
					if (flipped(triangles,vertices,refs,p,i1,i0,v1,v0,deleted1)) continue;

					// not flipped, so remove edge
					v0.p = p;
					v0.q = v1.q + v0.q;
					auto tstart = refs.size();

					update_triangles(triangles,vertices,refs,i0,v0,deleted0,deleted_triangles);
					update_triangles(triangles,vertices,refs,i0,v1,deleted1,deleted_triangles);

					auto tcount = refs.size() - tstart;

					if (tcount <= v0.tcount) {
						// save ram
						if (tcount) {
							memcpy(&refs.tris[v0.tstart],  &refs.tris[tstart],  tcount * sizeof(uint32_t));
							memcpy(&refs.verts[v0.tstart], &refs.verts[tstart], tcount * sizeof(uint8_t));
						}
					}
					else {
						v0.tstart = tstart; // append
					}

					v0.tcount = tcount;
					break;
				}
			}
		}
		
		if (deleted_triangles <= 0) {
			break;
		}
		
		deleted_triangles = 0;
	}
	compact_mesh(triangles, vertices);

	return iteration;
}

};
///////////////////////////////////////////

#undef loopi
#undef loopj
#undef loopk

#endif