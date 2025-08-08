#ifndef __ZMESH_UTILITY_HPP__
#define __ZMESH_UTILITY_HPP__

#include <algorithm>
#include <vector>
#include <limits>
#include <cmath>
#include <cstdint>

namespace zmesh::utility {

template <typename T = float>
class Vec3 {
public:
  T x, y, z;
  Vec3() : x(0), y(0), z(0) {}
  Vec3(T x, T y, T z) : x(x), y(y), z(z) {}

  Vec3 operator+(const Vec3& other) const {
    return Vec3(x + other.x, y + other.y, z + other.z);
  }
  void operator+=(const Vec3& other) {
    x += other.x;
    y += other.y;
    z += other.z;
  }
  Vec3 operator+(const T other) const {
    return Vec3(x + other, y + other, z + other);
  }
  void operator+=(const T other) {
    x += other;
    y += other;
    z += other;
  }
  Vec3 operator-() const {
    return Vec3(-x,-y,-z);
  }
  Vec3 operator-(const Vec3& other) const {
    return Vec3(x - other.x, y - other.y, z - other.z);
  }
  Vec3 operator-(const T scalar) const {
    return Vec3(x - scalar, y - scalar, z - scalar);
  }
  Vec3 operator*(const T scalar) const {
    return Vec3(x * scalar, y * scalar, z * scalar);
  }
  void operator*=(const T scalar) {
    x *= scalar;
    y *= scalar;
    z *= scalar;
  }
  Vec3 operator*(const Vec3& other) const {
    return Vec3(x * other.x, y * other.y, z * other.z);
  }
  void operator*=(const Vec3& other) {
    x *= other.x;
    y *= other.y;
    z *= other.z;
  }
  Vec3 operator/(const Vec3& other) const {
    return Vec3(x/other.x, y/other.y, z/other.z);
  }
  Vec3 operator/(const T divisor) const {
    return Vec3(x/divisor, y/divisor, z/divisor);
  }
  void operator/=(const T divisor) {
    x /= divisor;
    y /= divisor;
    z /= divisor;
  }
  bool operator==(const Vec3& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
  T& operator[](const int idx) {
    if (idx == 0) {
      return x;
    }
    else if (idx == 1) {
      return y;
    }
    else if (idx == 2) {
      return z;
    }
    else {
      throw new std::runtime_error("Index out of bounds.");
    }
  }
  T get(const int idx) const {
    if (idx == 0) {
      return x;
    }
    else if (idx == 1) {
      return y;
    }
    else if (idx == 2) {
      return z;
    }
    else {
      throw new std::runtime_error("Index out of bounds.");
    }
  }
  T dot(const Vec3& o) const {
    return x * o.x + y * o.y + z * o.z;
  }
  Vec3 abs() const {
    return Vec3(std::abs(x), std::abs(y), std::abs(z));
  }
  int argmax() const {
    if (x >= y) {
      return (x >= z) ? 0 : 2;
    }
    return (y >= z) ? 1 : 2;
  }
  T max() const {
    return std::max(x,std::max(y,z));
  }
  T min() const {
    return std::min(x,std::min(y,z));
  }
  float norm() const {
    return sqrt(x*x + y*y + z*z);
  }
  float norm2() const {
    return x*x + y*y + z*z;
  }
  bool close(const Vec3& o) const {
    return (*this - o).norm2() < 1e-4;
  }
  Vec3 cross(const Vec3& o) const {
    return Vec3(
      y * o.z - z * o.y, 
      z * o.x - x * o.z,
      x * o.y - y * o.x
    );
  }
  bool is_null() const {
    return x == 0 && y == 0 && z == 0;
  }
  int num_zero_dims() const {
    return (x == 0) + (y == 0) + (z == 0);
  }
  int num_non_zero_dims() const {
    return (x != 0) + (y != 0) + (z != 0);
  }
  bool is_axis_aligned() const {
    return ((x != 0) + (y != 0) + (z != 0)) == 1;
  }
  void print(const std::string &name) const {
    if constexpr (std::is_same<T, float>::value) {
      printf("%s %.3f, %.3f, %.3f\n",name.c_str(), x, y, z);  
    }
    else {
      printf("%s %d, %d, %d\n",name.c_str(), x, y, z);
    }
  }
};

struct Triangle {
    Vec3<float> v1, v2, v3;

    Triangle(const Vec3<float>& v1, const Vec3<float>& v2, const Vec3<float>& v3) : v1(v1), v2(v2), v3(v3) {}

    void print() const {
      printf("tri\n v1 %.1f %.1f %.1f\n v2 %.1f %.1f %.1f\n v3 %.1f %.1f %.1f\n",
        v1.x, v1.y, v1.z,
        v2.x, v2.y, v2.z,
        v3.x, v3.y, v3.z
      );
    }
};

struct MeshObject {
  std::vector<float> points;
  std::vector<float> normals;
  std::vector<unsigned int> faces;

  void add_point(const Vec3<float>& pt) {
    points.push_back(pt.x);
    points.push_back(pt.y);
    points.push_back(pt.z);
  }

  void add_triangle(
    const unsigned int f1, 
    const unsigned int f2, 
    const unsigned int f3
  ) {
    faces.push_back(f1);
    faces.push_back(f2);
    faces.push_back(f3);
  }

  void add_triangle(const Vec3<unsigned int>& face) {
    faces.push_back(face.x);
    faces.push_back(face.y);
    faces.push_back(face.z);
  }

  void add_triangle(const Triangle& tri) {
    unsigned int i = last_face();

    points.push_back(tri.v1.x);
    points.push_back(tri.v1.y);
    points.push_back(tri.v1.z);

    points.push_back(tri.v2.x);
    points.push_back(tri.v2.y);
    points.push_back(tri.v2.z);

    points.push_back(tri.v3.x);
    points.push_back(tri.v3.y);
    points.push_back(tri.v3.z);

    faces.push_back(i + 1);
    faces.push_back(i + 2);
    faces.push_back(i + 3);
  }

  unsigned int last_face() const {
    return (points.size() > 0) 
      ? ((points.size() - 1) / 3)
      : -1;
  }
};

Vec3<int32_t> zone2grid(int32_t zone, const Vec3<int32_t>& gs) {
    int32_t z = zone / gs.x / gs.y;
    int32_t y = (zone - gs.x * gs.y * z) / gs.x;
    int32_t x = zone - gs.x * (y + gs.y * z);
    return Vec3<int32_t>(x,y,z);
}

Vec3<float> intersect(int axis, float plane_offset, const Vec3<float> &p, const Vec3<float> &q) {
  float t = (plane_offset - p.get(axis)) / (q.get(axis) - p.get(axis));
  return p + (q - p) * t;
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
    const Vec3<float> verts[3] = { v1, v2, v3 };

    constexpr float epsilon = 1e-5;
    int aboveCount = 0, belowCount = 0, equalCount = 0;

    uint8_t i = 0;
    for (const auto& vertex : verts) {
      const float dist = vertex.get(axis) - plane_value;

      if (std::abs(dist) < epsilon) {
        equal[equalCount++] = i;
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

      // const int next = (below[0] == 2) 
      //   ? 0 
      //   : below[0] + 1;
      // const bool ccw = (above[0] == next);

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

  Vec3<float> planes;

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
    for (auto tri : cur_tris) {
      auto new_tris = divide_triangle(0, xplane, tri);
      next_tris.insert(next_tris.end(), new_tris.begin(), new_tris.end());
    }
    std::swap(cur_tris, next_tris);
    next_tris.clear();
  }

  for (uint32_t y = gys; y <= gye; y++) {
    float yplane = minpt.y + y * cs.y;
    for (auto tri : cur_tris) {
      auto new_tris = divide_triangle(1, yplane, tri);
      next_tris.insert(next_tris.end(), new_tris.begin(), new_tris.end());
    }
    std::swap(cur_tris, next_tris);
    next_tris.clear();
  }

  for (uint32_t z = gzs; z <= gze; z++) {
    float zplane = minpt.z + z * cs.z;
    for (auto tri : cur_tris) {
      auto new_tris = divide_triangle(2, zplane, tri);
      next_tris.insert(next_tris.end(), new_tris.begin(), new_tris.end());
    }
    std::swap(cur_tris, next_tris);
    next_tris.clear();
  }

  const float icx = 1 / cs.x;
  const float icy = 1 / cs.y;
  const float icz = 1 / cs.z;

  for (const auto& tri : cur_tris) {
    const Vec3<float>& pt = tri.v1; // v1 guaranteed to not be a border point (unless the triangle is degenerate)

    int ix = static_cast<int>((pt.x - minpt.x) * icx);
    int iy = static_cast<int>((pt.y - minpt.y) * icy);
    int iz = static_cast<int>((pt.z - minpt.z) * icz);

    ix = std::min(std::max(ix, static_cast<int>(0)), static_cast<int>(gs.x - 1));
    iy = std::min(std::max(iy, static_cast<int>(0)), static_cast<int>(gs.y - 1));
    iz = std::min(std::max(iz, static_cast<int>(0)), static_cast<int>(gs.z - 1));

    unsigned int zone = ix + gs.x * (iy + gs.y * iz);

    mesh_grid[zone].add_triangle(tri);
  }
}

void fix_all_different(
  const float* vertices,
  const Vec3<float> minpt,
  const std::vector<uint32_t>& face_remap,
  const std::vector<int32_t>& zones,
  std::vector<MeshObject>& mesh_grid,
  const Vec3<float>& cs,
  const Vec3<int32_t>& gs,
  unsigned int f1, // f1 and f2 should match on zone
  unsigned int f2,
  unsigned int f3 // outlier
) {
  Vec3 v1(vertices[3*f1+0], vertices[3*f1+1], vertices[3*f1+2]);
  Vec3 v2(vertices[3*f2+0], vertices[3*f2+1], vertices[3*f2+2]);
  Vec3 v3(vertices[3*f3+0], vertices[3*f3+1], vertices[3*f3+2]);

  auto z1 = zones[f1];
  auto z2 = zones[f2];
  auto z3 = zones[f3];

  Vec3<int32_t> g1 = zone2grid(z1, gs);
  Vec3<int32_t> g2 = zone2grid(z2, gs);
  Vec3<int32_t> g3 = zone2grid(z3, gs);

  Vec3<int32_t> delta12 = (g2 - g1).abs();
  Vec3<int32_t> delta13 = (g3 - g1).abs();
  Vec3<int32_t> delta23 = (g2 - g3).abs();

  if (delta12.max() > 1 || delta13.max() > 1 || delta23.max() > 1) {
    resect_triangle_iterative(
      vertices, minpt, 
      zones,
      mesh_grid, cs, gs,
      f1, f2, f3
    );
    return;
  }

  Vec3<int32_t> is3d = delta12.abs() + delta23.abs() + delta13.abs();

  if (is3d.num_non_zero_dims() == 3) {
    resect_triangle_iterative(
      vertices, minpt, 
      zones,
      mesh_grid, cs, gs,
      f1, f2, f3
    );
    return;
  }

  int ordering = 1;

  // rearrange vertices so that vertex 1 is in the single 
  // zone difference position for both 2 and 3
  if (delta12.num_non_zero_dims() == 1 && delta13.num_non_zero_dims() != 1) {
    std::swap(v1, v2);
    std::swap(f1, f2);
    std::swap(g1, g2);
    std::swap(z1, z2);
    ordering = 2;
  }
  else if (delta12.num_non_zero_dims() != 1 && delta13.num_non_zero_dims() == 1) {
    std::swap(v1, v3);
    std::swap(f1, f3);
    std::swap(g1, g3);
    std::swap(z1, z3);
    ordering = 3;
  }

  delta12 = (g2 - g1);

  int xaxis = 0;
  if (delta12.y != 0) {
    xaxis = 1;
  }
  else if (delta12.z != 0) {
    xaxis = 2;
  }

  delta13 = (g3 - g1);

  int yaxis = 0;
  if (delta13.y != 0 && xaxis != 1) {
    yaxis = 1;
  }
  else if (delta13.z != 0) {
    yaxis = 2;
  }

  if (xaxis == yaxis) {
    throw std::runtime_error("zmesh::utility::chunk_mesh_accelerated: xaxis should not equal yaxis.");
  }

  float plane_offset_x = minpt.get(xaxis) + std::max(std::max(g1[xaxis], g2[xaxis]), g3[xaxis]) * cs.get(xaxis);
  float plane_offset_y = minpt.get(yaxis) + std::max(std::max(g1[yaxis], g2[yaxis]), g3[yaxis]) * cs.get(yaxis);

  const Vec3 i23_0 = intersect(xaxis, plane_offset_x, v2, v3);
  const Vec3 i23_1 = intersect(yaxis, plane_offset_y, v2, v3);

  Vec3 corner = i23_0;
  corner[xaxis] = plane_offset_x;
  corner[yaxis] = plane_offset_y;

  MeshObject& m1 = mesh_grid[z1];
  MeshObject& m2 = mesh_grid[z2];
  MeshObject& m3 = mesh_grid[z3];
    
  auto g4 = g1;
  g4[xaxis] += delta12[xaxis];
  g4[yaxis] += delta13[yaxis];

  auto z4 = g4.x + gs.x * (g4.y + gs.y * g4.z);

  MeshObject& m4 = mesh_grid[z4];

  const Vec3 i13 = intersect(yaxis, plane_offset_y, v1, v3);
  const Vec3 i12 = intersect(xaxis, plane_offset_x, v1, v2);

  unsigned int m1i12 = 0;
  unsigned int m1i13 = 0;
  unsigned int m1i23_0= 0;
  unsigned int m1i23_1 = 0;
  unsigned int m1corner = 0;

  unsigned int m2i12 = 0;
  unsigned int m2i23_0 = 0;
  unsigned int m2i23_1 = 0;
  unsigned int m2corner = 0;

  unsigned int m3i13 = 0;
  unsigned int m3i23_0 = 0;
  unsigned int m3i23_1 = 0;
  unsigned int m3corner = 0;

  unsigned int m4corner = 0;
  unsigned int m4i23_0 = 0;
  unsigned int m4i23_1 = 0;

  if (i23_0.close(i23_1)) {
    m1.add_point(corner);
    m1corner = m1.last_face();

    m1.add_point(i13);
    m1i13 = m1.last_face();

    m1.add_point(i12);
    m1i12 = m1.last_face();

    m2.add_point(i12);
    m2i12 = m2.last_face();

    m2.add_point(corner);
    m2corner = m2.last_face();

    m3.add_point(corner);
    m3corner = m3.last_face();

    m3.add_point(i13);
    m3i13 = m3.last_face();

    if (ordering == 1) {
      m1.add_triangle(m1i12, m1corner, face_remap[f1]);
      m1.add_triangle(m1corner, m1i13, face_remap[f1]);
      m2.add_triangle(face_remap[f2], m2corner, m2i12);
      m3.add_triangle(face_remap[f3], m3i13, m3corner);      
    }
    else {
      m1.add_triangle(m1i12, face_remap[f1], m1corner);
      m1.add_triangle(m1corner, face_remap[f1], m1i13);
      m2.add_triangle(face_remap[f2], m2i12, m2corner);
      m3.add_triangle(face_remap[f3], m3corner, m3i13);
    }
  }
  else if (
    (v3.get(yaxis) > plane_offset_y && i23_0.get(yaxis) <= plane_offset_y)
    || (v3.get(yaxis) < plane_offset_y && i23_0.get(yaxis) >= plane_offset_y)
  ) {
    // 5 triangle situation
    m3.add_point(i13);
    m3i13 = m3.last_face();
    m3.add_point(i23_1);
    m3i23_1 = m3.last_face();

    m1.add_point(i13);
    m1i13 = m1.last_face();
    m1.add_point(i23_1);
    m1i23_1 = m1.last_face();
    m1.add_point(i23_0);
    m1i23_0 = m1.last_face();
    m1.add_point(i12);
    m1i12 = m1.last_face();

    m2.add_point(i23_0);
    m2i23_0 = m2.last_face();
    m2.add_point(i12);
    m2i12 = m2.last_face();

    if (ordering == 1) {
      m3.add_triangle(face_remap[f3], m3i13  , m3i23_1);
      m1.add_triangle(face_remap[f1], m1i23_1, m1i13  );
      m1.add_triangle(face_remap[f1], m1i23_0, m1i23_1);
      m1.add_triangle(face_remap[f1], m1i12,   m1i23_0);
      m2.add_triangle(face_remap[f2], m2i23_0, m2i12  );
    }
    else {
      m3.add_triangle(face_remap[f3], m3i23_1, m3i13  );
      m1.add_triangle(face_remap[f1], m1i13  , m1i23_1);
      m1.add_triangle(face_remap[f1], m1i23_1, m1i23_0);
      m1.add_triangle(face_remap[f1], m1i23_0, m1i12  );
      m2.add_triangle(face_remap[f2], m2i12  , m2i23_0);
    }
  }
  else {
    m3.add_point(corner);
    m3corner = m3.last_face();
    m3.add_point(i13);
    m3i13 = m3.last_face();
    m3.add_point(i23_0);
    m3i23_0 = m3.last_face();

    m1.add_point(i13);
    m1i13 = m1.last_face();
    m1.add_point(corner);
    m1corner = m1.last_face();
    m1.add_point(i12);
    m1i12 = m1.last_face();

    m2.add_point(i23_1);
    m2i23_1 = m2.last_face();
    m2.add_point(corner);
    m2corner = m2.last_face();
    m2.add_point(i12);
    m2i12 = m2.last_face();

    m4.add_point(corner);
    m4corner = m4.last_face();

    m4.add_point(i23_0);
    m4i23_0 = m4.last_face();

    m4.add_point(i23_1);
    m4i23_1 = m4.last_face();

    if (ordering == 1) {
      m1.add_triangle(face_remap[f1], m1corner,       m1i13);
      m1.add_triangle(face_remap[f1], m1i12,          m1corner);
      m2.add_triangle(face_remap[f2], m2corner,       m2i12);
      m2.add_triangle(face_remap[f2], m2i23_1,        m2corner);
      m3.add_triangle(face_remap[f3], m3corner,       m3i23_0);
      m3.add_triangle(m3corner,       face_remap[f3], m3i13);
      m4.add_triangle(m4corner,       m4i23_1,        m4i23_0);
    }
    else {
      m1.add_triangle(face_remap[f1], m1i13,          m1corner);
      m1.add_triangle(face_remap[f1], m1corner,       m1i12);
      m2.add_triangle(face_remap[f2], m2i12,          m2corner);
      m2.add_triangle(face_remap[f2], m2corner,       m2i23_1);
      m3.add_triangle(face_remap[f3], m3i23_0,        m3corner);
      m3.add_triangle(m3corner,       m3i13,          face_remap[f3]);
      m4.add_triangle(m4corner,       m4i23_0,        m4i23_1);
    }
  }
}

void fix_single_outlier_18_connected(
  const float* vertices,
  const Vec3<float> minpt,
  const std::vector<uint32_t>& face_remap,
  const std::vector<int32_t>& zones,
  std::vector<MeshObject>& mesh_grid,
  const Vec3<float>& cs,
  const Vec3<int32_t>& gs,
  const unsigned int f1, // f1 and f2 should match on zone
  const unsigned int f2,
  const unsigned int f3 // outlier
) {

  const Vec3 v1(vertices[3*f1+0], vertices[3*f1+1], vertices[3*f1+2]);
  const Vec3 v2(vertices[3*f2+0], vertices[3*f2+1], vertices[3*f2+2]);
  const Vec3 v3(vertices[3*f3+0], vertices[3*f3+1], vertices[3*f3+2]);

  auto z1 = zones[f1];
  auto z3 = zones[f3];

  Vec3<int32_t> g1 = zone2grid(z1, gs);
  Vec3<int32_t> g3 = zone2grid(z3, gs);

  Vec3<int32_t> delta = g3 - g1;

  if (delta.abs().max() > 1) {
    resect_triangle_iterative(
      vertices, minpt, 
      zones,
      mesh_grid, cs, gs,
      f1, f2, f3
    );
    return;
  }

  int xaxis = 0;
  int yaxis = 1;

  if (delta.x) {
    if (delta.z) {
      yaxis = 2;
    }
  }
  else if (delta.y) {
    xaxis = 1;
    yaxis = 2;
  }
  else {
    throw std::runtime_error("zmesh::utility::chunk_mesh_accelerated::fix_single_outlier_18_connected: Should never happen.");
  }

  float plane_offset_x = minpt.get(xaxis) + std::max(g1[xaxis], g3[xaxis]) * cs.get(xaxis);
  float plane_offset_y = minpt.get(yaxis) + std::max(g1[yaxis], g3[yaxis]) * cs.get(yaxis);

  const Vec3 i13x = intersect(xaxis, plane_offset_x, v1, v3);
  const Vec3 i23x = intersect(xaxis, plane_offset_x, v2, v3);

  const Vec3 i13y = intersect(yaxis, plane_offset_y, v1, v3);
  const Vec3 i23y = intersect(yaxis, plane_offset_y, v2, v3);

  Vec3 corner = i13x;
  corner[xaxis] = plane_offset_x;
  corner[yaxis] = plane_offset_y;

  // three cases, if both points are to one side of the corner (left and right)
  // or if they straddle the corner

  // m1,m3 meshes corresponding to original vertices
  // m4,m5 meshes corresponding to adjacent zones

  MeshObject& m1 = mesh_grid[z1];
  MeshObject& m3 = mesh_grid[z3];

  Vec3 g4 = g1;
  g4[xaxis] += delta[xaxis];

  Vec3 g5 = g1;
  g5[yaxis] += delta[yaxis];

  auto z4 = g4.x + gs.x * (g4.y + gs.y * g4.z);
  auto z5 = g5.x + gs.x * (g5.y + gs.y * g5.z);

  MeshObject& m4 = mesh_grid[z4];
  MeshObject& m5 = mesh_grid[z5];

  unsigned int m1i13x = 0;
  unsigned int m1i23x = 0;
  unsigned int m1i13y = 0;
  unsigned int m1i23y = 0;
  unsigned int m1corner = 0;

  unsigned int m3i13x = 0;
  unsigned int m3i23x = 0;
  unsigned int m3i13y = 0;
  unsigned int m3i23y = 0;
  unsigned int m3corner = 0;

  unsigned int m4i13x = 0;
  unsigned int m4i23x = 0;
  unsigned int m4i13y = 0;
  unsigned int m4i23y = 0;
  unsigned int m4corner = 0;

  unsigned int m5i13x = 0;
  unsigned int m5i23x = 0;
  unsigned int m5i13y = 0;
  unsigned int m5i23y = 0;
  unsigned int m5corner = 0;
  
  if (i13x.get(xaxis) < corner.get(xaxis) && i23x.get(xaxis) < corner.get(xaxis)) {
    m1.add_point(i13x);
    m1i13x = m1.last_face();
    m1.add_point(i23x);
    m1i23x = m1.last_face();

    m4.add_point(i13x);
    m4i13x = m4.last_face();

    m4.add_point(i23x);
    m4i23x = m4.last_face();

    m4.add_point(i13y);
    m4i13y = m4.last_face();

    m4.add_point(i23y);
    m4i23y = m4.last_face();

    m3.add_point(i13y);
    m3i13y = m3.last_face();

    m3.add_point(i23y);
    m3i23y = m3.last_face();

    m1.add_triangle(face_remap[f1], m1i23x, m1i13x);
    m1.add_triangle(face_remap[f1], face_remap[f2], m1i23x);
    m4.add_triangle(m4i13x, m4i23y, m4i13y);
    m4.add_triangle(m4i13x, m4i23x, m4i23y);
    m3.add_triangle(face_remap[f3], m3i13y, m3i23y);
  }
  else if (i13x.get(xaxis) > corner.get(xaxis) && i23x.get(xaxis) > corner.get(xaxis)) {
    m1.add_point(i13y);
    m1i13y = m1.last_face();
    m1.add_point(i23y);
    m1i23y = m1.last_face();

    m5.add_point(i13x);
    m5i13x = m5.last_face();

    m5.add_point(i23x);
    m5i23x = m5.last_face();

    m5.add_point(i13y);
    m5i13y = m5.last_face();

    m5.add_point(i23y);
    m5i23y = m5.last_face();

    m3.add_point(i13x);
    m3i13x = m3.last_face();

    m3.add_point(i23x);
    m3i23x = m3.last_face();

    m1.add_triangle(face_remap[f1], m1i23y, m1i13y);
    m1.add_triangle(face_remap[f1], face_remap[f2], m1i23y);

    m3.add_triangle(face_remap[f3], m3i13x, m3i23x);

    m5.add_triangle(m5i23y, m5i13y, m5i13x);
    m5.add_triangle(m5i23y, m5i23x, m5i13x);
  }
  else {
    m1.add_point(i13x);
    m1i13x = m1.last_face();

    m1.add_point(corner);
    m1corner = m1.last_face();

    m1.add_point(i23y);
    m1i23y = m1.last_face();

    m4.add_point(i13x);
    m4i13x = m4.last_face();

    m4.add_point(i13y);
    m4i13y = m4.last_face();

    m4.add_point(corner);
    m4corner = m4.last_face();

    m5.add_point(i23y);
    m5i23y = m5.last_face();

    m5.add_point(corner);
    m5corner = m5.last_face();

    m5.add_point(i23x);
    m5i23x = m5.last_face();

    m3.add_point(i13y);
    m3i13y = m3.last_face();

    m3.add_point(corner);
    m3corner = m3.last_face();

    m3.add_point(i23x);
    m3i23x = m3.last_face();

    m1.add_triangle(face_remap[f1], m1i13x, face_remap[f2]);
    m1.add_triangle(face_remap[f2], m1i13x, m1i23y);
    m1.add_triangle(m1i23y, m1i13x, m1corner);

    m3.add_triangle(m3i13y, face_remap[f3], m3corner);
    m3.add_triangle(m3corner, face_remap[f3], m3i23x);

    m4.add_triangle(m4i13x, m4i13y, m4corner);

    m5.add_triangle(m5i23y, m5corner, m5i23x);
  }
}

void fix_single_outlier_6_connected(
  const float* vertices,
  const Vec3<float> minpt,
  const std::vector<uint32_t>& face_remap,
  const std::vector<int32_t>& zones,
  std::vector<MeshObject>& mesh_grid,
  const Vec3<float>& cs,
  const Vec3<int32_t>& gs,
  const unsigned int f1, // f1 and f2 should match on zone
  const unsigned int f2,
  const unsigned int f3, // outlier
  const int ordering
) {
  const Vec3 v1(vertices[3*f1+0], vertices[3*f1+1], vertices[3*f1+2]);
  const Vec3 v2(vertices[3*f2+0], vertices[3*f2+1], vertices[3*f2+2]);
  const Vec3 v3(vertices[3*f3+0], vertices[3*f3+1], vertices[3*f3+2]);

  auto z1 = zones[f1];
  auto z3 = zones[f3];

  Vec3<int32_t> g1 = zone2grid(z1, gs);
  Vec3<int32_t> g3 = zone2grid(z3, gs);

  Vec3<int32_t> delta = g3 - g1;

  if (delta.abs().max() > 1) {
    resect_triangle_iterative(
      vertices, minpt, 
      zones,
      mesh_grid, cs, gs,
      f1, f2, f3
    );
    return;
  } 

  int axis = 0;
  if (delta.y != 0) {
    axis = 1;
  }
  else if (delta.z != 0) {
    axis = 2;
  }

  const float plane_offset = minpt.get(axis) + std::max(g1[axis], g3[axis]) * cs.get(axis);

  const Vec3 i13 = intersect(axis, plane_offset, v1, v3);
  const Vec3 i23 = intersect(axis, plane_offset, v2, v3);

  MeshObject& m1 = mesh_grid[z1];
  MeshObject& m3 = mesh_grid[z3];

  m1.add_point(i13);
  const unsigned int m1f13 = m1.last_face();

  m1.add_point(i23);
  const unsigned int m1f23 = m1f13 + 1;

  m3.add_point(i13);
  const unsigned int m3f13 = m3.last_face();

  m3.add_point(i23);
  const unsigned int m3f23 = m3f13 + 1;

  if (ordering & 1) { // 1 or 3
    m1.add_triangle(face_remap[f1], m1f23, m1f13);
    m1.add_triangle(face_remap[f1], face_remap[f2], m1f23);
    m3.add_triangle(face_remap[f3], m3f13, m3f23);
  }
  else {
    m1.add_triangle(face_remap[f1], m1f13, m1f23);
    m1.add_triangle(face_remap[f1], m1f23, face_remap[f2]);
    m3.add_triangle(face_remap[f3], m3f23, m3f13);
  }
}

void fix_single_outlier(
  const float* vertices,
  const Vec3<float> minpt,
  const std::vector<uint32_t>& face_remap,
  const std::vector<int32_t>& zones,
  std::vector<MeshObject>& mesh_grid,
  const Vec3<float>& cs,
  const Vec3<int32_t>& gs,
  const unsigned int f1, // f1 and f2 should match on zone
  const unsigned int f2,
  const unsigned int f3, // outlier
  const int ordering
) {
  auto z1 = zones[f1];
  auto z3 = zones[f3];

  Vec3<int32_t> g1 = zone2grid(z1, gs);
  Vec3<int32_t> g3 = zone2grid(z3, gs);

  Vec3<int32_t> delta = g3 - g1;

  if (delta.num_non_zero_dims() == 1) {
    fix_single_outlier_6_connected(
      vertices, minpt, 
      face_remap, zones, 
      mesh_grid, cs, gs,
      f1, f2, f3, ordering
    );
  }
  else if (delta.num_non_zero_dims() == 2) {
    fix_single_outlier_18_connected(
      vertices, minpt, 
      face_remap, zones, 
      mesh_grid, cs, gs,
      f1, f2, f3
    );
  }
  else if (delta.num_non_zero_dims() == 3) {
    // 26 connected
    resect_triangle_iterative(
      vertices, minpt, 
      zones,
      mesh_grid, cs, gs,
      f1, f2, f3
    );
  }
  else {
    throw std::runtime_error("zmesh::utility::chunk_mesh_accelerated: Non-zero delta was not 1,2, or 3.");
  }
}

// cx = chunk size x, etc
std::vector<MeshObject> chunk_mesh_accelerated_simplified(
  const float* vertices, 
  const uint64_t num_vertices,
  const unsigned int* faces,
  const uint64_t num_faces,
  const float cx, const float cy, const float cz
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


// cx = chunk size x, etc
std::vector<MeshObject> chunk_mesh_accelerated(
  const float* vertices, 
  const uint64_t num_vertices,
  const unsigned int* faces,
  const uint64_t num_faces,
  const float cx, const float cy, const float cz
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
  
  std::vector<uint32_t> zonect(gx * gy * gz);
  std::vector<uint32_t> face_remap(num_vertices);

  for (uint32_t i = 0; i < num_vertices; i++) {
    uint32_t zone = zones[i];
    MeshObject& obj = mesh_grid[zone];
    face_remap[i] = zonect[zone];
    zonect[zone]++;

    obj.points.push_back(vertices[(i * 3) + 0]);
    obj.points.push_back(vertices[(i * 3) + 1]);
    obj.points.push_back(vertices[(i * 3) + 2]);
  }

  for (uint64_t i = 0; i < num_faces * 3; i += 3) {
    auto f1 = faces[i+0];
    auto f2 = faces[i+1];
    auto f3 = faces[i+2];

    if (zones[f1] == zones[f2] && zones[f1] == zones[f3]) {
      printf("HERE\n");
      auto zone = zones[f1];
      MeshObject& obj = mesh_grid[zone];
      obj.faces.push_back(face_remap[f1]);
      obj.faces.push_back(face_remap[f2]);
      obj.faces.push_back(face_remap[f3]);
      continue;
    }
    
    if (zones[f1] == zones[f2]) {
      printf("HERE 1\n");
      fix_single_outlier(
        vertices, minpt, 
        face_remap, zones, 
        mesh_grid, cs, gs,
        f1, f2, f3, 1
      );
    }
    else if (zones[f1] == zones[f3]) {
      printf("HERE 2\n");
      fix_single_outlier(
        vertices, minpt, 
        face_remap, zones, 
        mesh_grid, cs, gs,
        f1, f3, f2, 2
      );
    }
    else if (zones[f2] == zones[f3]) {
      printf("HERE 3\n");
      fix_single_outlier(
        vertices, minpt, 
        face_remap, zones, 
        mesh_grid, cs, gs,
        f2, f3, f1, 3
      );
    }
    else {
      printf("HERE 4\n");
      fix_all_different(
        vertices, minpt, 
        face_remap, zones, 
        mesh_grid, cs, gs,
        f1, f2, f3
      );
    }
  }

  return mesh_grid;
}

};

#endif
