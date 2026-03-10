#ifndef __ZMESH_CHUNK_MESH_HPP__
#define __ZMESH_CHUNK_MESH_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "utility.hpp"

using namespace zmesh::utility;

namespace zmesh::chunk_mesh {

Vec3<int32_t> zone2grid(int32_t zone, const Vec3<int32_t>& gs) {
    int32_t z = zone / gs.x / gs.y;
    int32_t y = (zone - gs.x * gs.y * z) / gs.x;
    int32_t x = zone - gs.x * (y + gs.y * z);
    return Vec3<int32_t>(x,y,z);
}

Vec3<float> intersect(int axis, float plane_offset, const Vec3<float> &p, const Vec3<float> &q) {
  float t = (plane_offset - p.get(axis)) / (q.get(axis) - p.get(axis));
  Vec3<float> result = p + (q - p) * t;
  result[axis] = plane_offset; // snap to grid
  return result;
}

std::vector<Triangle> divide_triangle(
  const int axis,
  const float plane_value,
  const Vec3<float>& v1,
  const Vec3<float>& v2,
  const Vec3<float>& v3
) {
    uint8_t above[3] = {};
    uint8_t below[3] = {};
    uint8_t equal[3] = {};
    Vec3<float> verts[3] = { v1, v2, v3 };

    constexpr float epsilon = 1e-5;
    int aboveCount = 0, belowCount = 0, equalCount = 0;

    uint8_t i = 0;
    for (auto& vertex : verts) {
      const float dist = vertex.get(axis) - plane_value;

      if (std::abs(dist) < epsilon) {
        equal[equalCount++] = i;
        vertex[axis] = plane_value; // snap to grid to avoid floating point errors
      }
      else if (dist > 0) {
        above[aboveCount++] = i;
      } 
      else {
        below[belowCount++] = i;
      }
      i++;
    }

    std::vector<Triangle> result;
    Vec3 i1, i2;

    // note: we are exploting the fact that zmesh gives us consistent winding
    // from the start. if the mesh is not consistent already, it will not be 
    // made consistent by this procedure.

    // note: if plane slices very close to a vertex but not within epsilon,
    // an ugly thin triangle will result. fix this later by setting some 
    // threshold for drawing two triangles.
    if (
         aboveCount == 3 
      || belowCount == 3 
      || equalCount >= 2 
      || (equalCount == 1 && (aboveCount == 2 || belowCount == 2))
    ) {
      result.emplace_back(v1, v2, v3);
    }
    else if (equalCount == 1) {
      int v1i, v2i, v3i;

      if (equal[0] == 0) {
        v1i = 0; v2i = 1; v3i = 2;
      } else if (equal[0] == 1) {
        v1i = 1; v2i = 2; v3i = 0;
      } else {
        v1i = 2; v2i = 0; v3i = 1;
      }

      const Vec3<float>& a = verts[v1i];
      const Vec3<float>& b = verts[v2i];
      const Vec3<float>& c = verts[v3i];

      i1 = intersect(axis, plane_value, b, c);
      result.emplace_back(a, b, i1);
      result.emplace_back(a, i1, c);
    }
    else if (aboveCount == 2) {
      const Vec3<float>& a = verts[below[0]];
      const Vec3<float>& b = verts[above[0]];
      const Vec3<float>& c = verts[above[1]];

      i1 = intersect(axis, plane_value, a, b);
      i2 = intersect(axis, plane_value, a, c);

      const Vec3<float> normal = (v2 - v1).cross(v3 - v1);
      const Vec3<float> subnormal = (b - a).cross(c - a);
      const bool ccw = (subnormal.dot(normal) > 0);

      if (ccw) {
        result.emplace_back(a, i1, i2);
        result.emplace_back(i1, b, i2);
        result.emplace_back(i2, b, c);
      }
      else {
        result.emplace_back(a, i2, i1);
        result.emplace_back(i2, b, i1);
        result.emplace_back(c, b, i2);
      }
    }
    else {
      const Vec3<float>& a = verts[above[0]];
      const Vec3<float>& b = verts[below[0]];
      const Vec3<float>& c = verts[below[1]];

      i1 = intersect(axis, plane_value, a, b);
      i2 = intersect(axis, plane_value, a, c);

      const Vec3<float> normal = (v2 - v1).cross(v3 - v1);
      const Vec3<float> subnormal = (b - a).cross(c - a);
      const bool ccw = (subnormal.dot(normal) > 0);

      if (ccw) {
        result.emplace_back(a, i1, i2);
        result.emplace_back(b, i2, i1);
        result.emplace_back(b, c, i2);
      }
      else {
        result.emplace_back(i1, a, i2);
        result.emplace_back(i1, i2, b);
        result.emplace_back(b, i2, c);
      }
    }

    return result;
}

auto divide_triangle(  
  const int axis,
  const float plane_value,
  const Triangle& tri
) {
  return divide_triangle(axis, plane_value, tri.v1, tri.v2, tri.v3);
}

// more elegant algorithmically, but not the fastest or simpliest
// division of the triangle into subtriangles
void resect_triangle_iterative(
  const float* vertices,
  const Vec3<float> minpt,
  const std::vector<int32_t>& zones,
  std::vector<MeshObject>& mesh_grid,
  const Vec3<float>& cs,
  const Vec3<int32_t>& gs,
  const unsigned int f1,
  const unsigned int f2,
  const unsigned int f3
) {
  const Vec3 v1(vertices[3*f1+0], vertices[3*f1+1], vertices[3*f1+2]);
  const Vec3 v2(vertices[3*f2+0], vertices[3*f2+1], vertices[3*f2+2]);
  const Vec3 v3(vertices[3*f3+0], vertices[3*f3+1], vertices[3*f3+2]);

  auto z1 = zones[f1];
  auto z2 = zones[f2];
  auto z3 = zones[f3];

  Vec3<int32_t> g1 = zone2grid(z1, gs);
  Vec3<int32_t> g2 = zone2grid(z2, gs);
  Vec3<int32_t> g3 = zone2grid(z3, gs);

  uint32_t gxs = std::min(std::min(g1.x, g2.x), g3.x);
  uint32_t gxe = std::max(std::max(g1.x, g2.x), g3.x);

  uint32_t gys = std::min(std::min(g1.y, g2.y), g3.y);
  uint32_t gye = std::max(std::max(g1.y, g2.y), g3.y);

  uint32_t gzs = std::min(std::min(g1.z, g2.z), g3.z);
  uint32_t gze = std::max(std::max(g1.z, g2.z), g3.z);

  std::vector<Triangle> cur_tris;
  std::vector<Triangle> next_tris;
  
  cur_tris.emplace_back(v1, v2, v3);

  for (uint32_t x = gxs; x <= gxe; x++) {
    float xplane = minpt.x + x * cs.x;
    for (const auto& tri : cur_tris) {
      auto new_tris = divide_triangle(0, xplane, tri);
      next_tris.insert(next_tris.end(), new_tris.begin(), new_tris.end());
    }
    std::swap(cur_tris, next_tris);
    next_tris.clear();
  }

  for (uint32_t y = gys; y <= gye; y++) {
    float yplane = minpt.y + y * cs.y;
    for (const auto& tri : cur_tris) {
      auto new_tris = divide_triangle(1, yplane, tri);
      next_tris.insert(next_tris.end(), new_tris.begin(), new_tris.end());
    }
    std::swap(cur_tris, next_tris);
    next_tris.clear();
  }

  for (uint32_t z = gzs; z <= gze; z++) {
    float zplane = minpt.z + z * cs.z;
    for (const auto& tri : cur_tris) {
      auto new_tris = divide_triangle(2, zplane, tri);
      next_tris.insert(next_tris.end(), new_tris.begin(), new_tris.end());
    }
    std::swap(cur_tris, next_tris);
    next_tris.clear();
  }

  const float icx = 1 / cs.x;
  const float icy = 1 / cs.y;
  const float icz = 1 / cs.z;

  auto zonefn = [&](const Vec3<float>& pt) {
    unsigned int ix = static_cast<unsigned int>((pt.x - minpt.x) * icx);
    unsigned int iy = static_cast<unsigned int>((pt.y - minpt.y) * icy);
    unsigned int iz = static_cast<unsigned int>((pt.z - minpt.z) * icz);

    ix = std::min(std::max(ix, static_cast<unsigned int>(0)), static_cast<unsigned int>(gs.x - 1));
    iy = std::min(std::max(iy, static_cast<unsigned int>(0)), static_cast<unsigned int>(gs.y - 1));
    iz = std::min(std::max(iz, static_cast<unsigned int>(0)), static_cast<unsigned int>(gs.z - 1));

    return ix + gs.x * (iy + gs.y * iz);
  };

  for (const auto& tri : cur_tris) {
    // v1 guaranteed to not be a border point (unless the triangle is degenerate)
    unsigned int z1 = zonefn(tri.v1);
    unsigned int z2 = zonefn(tri.v2);
    unsigned int z3 = zonefn(tri.v3);

    unsigned int zone = std::min(std::min(z1,z2), z3);

    mesh_grid[zone].add_triangle(tri);
  }
}

// cx = chunk size x, etc
std::vector<MeshObject> chunk_mesh_accelerated(
  const float* vertices, 
  const uint64_t num_vertices,
  const unsigned int* faces,
  const uint64_t num_faces,
  const float cx, const float cy, const float cz,
  const std::optional<float> origin_x = std::nullopt, 
  const std::optional<float> origin_y = std::nullopt, 
  const std::optional<float> origin_z = std::nullopt
) {

  if (cx <= 0 || cy <= 0 || cz <= 0) {
    throw std::runtime_error("Chunk size must have a positive non-zero volume.");
  }

  const Vec3 cs(cx,cy,cz);

  float min_x = INFINITY;
  float min_y = INFINITY;
  float min_z = INFINITY;
  float max_x = -INFINITY;
  float max_y = -INFINITY;
  float max_z = -INFINITY;

  for (uint64_t i = 0; i < num_vertices * 3; i += 3) {
    min_x = std::min(min_x, vertices[i]);
    max_x = std::max(max_x, vertices[i]);

    min_y = std::min(min_y, vertices[i+1]);
    max_y = std::max(max_y, vertices[i+1]);

    min_z = std::min(min_z, vertices[i+2]);
    max_z = std::max(max_z, vertices[i+2]);
  }

  if (origin_x.has_value()) {
    min_x = *origin_x;
  }
  if (origin_y.has_value()) {
    min_y = *origin_y;
  }
  if (origin_z.has_value()) {
    min_z = *origin_z;
  }

  const Vec3 minpt(min_x, min_y, min_z);

  const int32_t gx = std::max(static_cast<int32_t>(std::ceil((max_x - min_x) / cx)), static_cast<int32_t>(1));
  const int32_t gy = std::max(static_cast<int32_t>(std::ceil((max_y - min_y) / cy)), static_cast<int32_t>(1));
  const int32_t gz = std::max(static_cast<int32_t>(std::ceil((max_z - min_z) / cz)), static_cast<int32_t>(1));

  const Vec3<int32_t> gs(gx,gy,gz);

  std::vector<int32_t> zones(num_vertices);

  const float icx = 1 / cx;
  const float icy = 1 / cy;
  const float icz = 1 / cz;

  for (uint64_t i = 0, j = 0; j < num_vertices; i += 3, j++) {
    int ix = static_cast<int>((vertices[i] - min_x) * icx) ;
    int iy = static_cast<int>((vertices[i+1] - min_y) * icy);
    int iz = static_cast<int>((vertices[i+2] - min_z) * icz);

    ix = std::min(std::max(ix, static_cast<int>(0)), static_cast<int>(gx - 1));
    iy = std::min(std::max(iy, static_cast<int>(0)), static_cast<int>(gy - 1));
    iz = std::min(std::max(iz, static_cast<int>(0)), static_cast<int>(gz - 1));

    zones[j] = ix + gx * (iy + gy * iz);
  }

  std::vector<MeshObject> mesh_grid(gx * gy * gz);
  
  for (uint64_t i = 0; i < num_faces * 3; i += 3) {
    auto f1 = faces[i+0];
    auto f2 = faces[i+1];
    auto f3 = faces[i+2];

    resect_triangle_iterative(
      vertices, minpt, 
      zones,
      mesh_grid, cs, gs,
      f1, f2, f3
    );
  }

  return mesh_grid;
}

std::vector<float> compute_vertex_normals_from_faces(
  const float* vertices,
  const uint64_t Nv,
  const uint32_t* faces,
  const uint64_t Nf
) {

  std::vector<Vec3<float>> normals(Nv);

  for (uint64_t i = 0; i < Nf; i += 3) {
    const uint32_t f0 = faces[i];
    const uint32_t f1 = faces[i+1];
    const uint32_t f2 = faces[i+2];

    const Vec3<float> v0(vertices[3*f0], vertices[3*f0 + 1], vertices[3*f0 + 2]);
    const Vec3<float> v1(vertices[3*f1], vertices[3*f1 + 1], vertices[3*f1 + 2]);
    const Vec3<float> v2(vertices[3*f2], vertices[3*f2 + 1], vertices[3*f2 + 2]);

    const Vec3<float> center = (v0 + v1 + v2) / 3;
    const Vec3<float> nhat = (v1 - v0).cross(v2 - v0).hat();

    normals[f0] += nhat * (v0 - center).len();
    normals[f1] += nhat * (v1 - center).len();
    normals[f2] += nhat * (v2 - center).len();
  }

  std::vector<float> normals_linear(Nv * 3);
  for (uint64_t i = 0; i < Nv; i++) {
    // Note: No need to average because the hat
    // operator takes care of that. The proof is
    // left as an exercise to the reader.
    normals[i] = normals[i].hat();

    normals_linear[i*3+0] = normals[i].x;
    normals_linear[i*3+1] = normals[i].y;
    normals_linear[i*3+2] = normals[i].z;
  }

  return normals_linear;
}

};

#endif
