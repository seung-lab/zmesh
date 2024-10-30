#ifndef __ZMESH_UTILITY_HPP__
#define __ZMESH_UTILITY_HPP__

#include <algorithm>
#include <vector>
#include <limits>
#include <cmath>
#include <cstdint>

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

  unsigned int last_face() const {
    return (points.size() - 1) / 3;
  }
};

Vec3<int32_t> zone2grid(int32_t zone, const Vec3<int32_t>& gs) {
    int32_t z = zone / gs.x / gs.y;
    int32_t y = (zone - gs.x * gs.y * z) / gs.x;
    int32_t x = zone - gs.x * (y + gs.y * z);
    return Vec3<int32_t>(x,y,z);
}

Vec3<float> intersect(int axis, float plane_offset, const Vec3<float> &p, const Vec3<float> &q) {
  float t = (plane_offset - p.get(axis)) / (p.get(axis) - q.get(axis));
  return p + (p - q) * t;
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
    throw std::runtime_error("This code only handles differences of a single 26-connected grid space.");
  }

  Vec3<int32_t> is3d = delta12.abs() + delta23.abs() + delta13.abs();

  if (is3d.num_non_zero_dims() == 3) {
    return;
  }

  // rearrange vertices so that vertex 1 is in the single 
  // zone difference position for both 2 and 3
  if (delta12.num_non_zero_dims() == 1 && delta13.num_non_zero_dims() != 1) {
    std::swap(v1, v2);
    std::swap(f1, f2);
    std::swap(g1, g2);
    std::swap(z1, z2);
  }
  else if (delta12.num_non_zero_dims() != 1 && delta13.num_non_zero_dims() == 1) {
    std::swap(v1, v3);
    std::swap(f1, f3);
    std::swap(g1, g3);
    std::swap(z1, z3);
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
    throw std::runtime_error("xaxis should not equal yaxis.");
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

    m1.add_triangle(m1i12, face_remap[f1], m1corner);
    m1.add_triangle(m1corner, face_remap[f1], m1i13);

    m2.add_point(i12);
    m2i12 = m2.last_face();

    m2.add_point(corner);
    m2corner = m2.last_face();

    m2.add_triangle(face_remap[f2], m2i12, m2corner);

    m3.add_point(corner);
    m3corner = m3.last_face();

    m3.add_point(i13);
    m3i13 = m3.last_face();

    m3.add_triangle(face_remap[f3], m3corner, m3i13);
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
    m3.add_triangle(face_remap[f3], m3i13, m3i23_1);

    m1.add_point(i13);
    m1i13 = m1.last_face();
    m1.add_point(i23_1);
    m1i23_1 = m1.last_face();
    m1.add_point(i23_0);
    m1i23_0 = m1.last_face();
    m1.add_point(i12);
    m1i12 = m1.last_face();

    m1.add_triangle(face_remap[f1], m1i13, m1i23_1);
    m1.add_triangle(face_remap[f1], m1i23_1, m1i23_0);
    m1.add_triangle(face_remap[f1], m1i23_0, m1i12);

    m2.add_point(i23_0);
    m2i23_0 = m2.last_face();
    m2.add_point(i12);
    m2i12 = m2.last_face();

    m2.add_triangle(face_remap[f2], m2i23_0, m2i12);
  }
  else {
    m3.add_point(corner);
    m3corner = m3.last_face();
    m3.add_point(i13);
    m3i13 = m3.last_face();
    m3.add_point(i23_0);
    m3i23_0 = m3.last_face();

    m3.add_triangle(face_remap[f3], m3corner, m3i23_0);
    m3.add_triangle(m3corner, face_remap[f3], m3i13);

    m1.add_point(i13);
    m1i13 = m1.last_face();
    m1.add_point(corner);
    m1corner = m1.last_face();
    m1.add_point(i12);
    m1i12 = m1.last_face();

    m1.add_triangle(face_remap[f1], m1i13, m1corner);
    m1.add_triangle(face_remap[f1], m1corner, m1i12);

    m2.add_point(i23_1);
    m2i23_1 = m2.last_face();
    m2.add_point(corner);
    m2corner = m2.last_face();
    m2.add_point(i12);
    m2i12 = m2.last_face();

    m2.add_triangle(face_remap[f2], m2corner, m2i23_1);
    m2.add_triangle(face_remap[f2], m2corner, m2i12);

    m4.add_point(corner);
    m4corner = m4.last_face();

    m4.add_point(i23_0);
    m4i23_0 = m4.last_face();

    m4.add_point(i23_1);
    m4i23_1 = m4.last_face();

    m4.add_triangle(m4corner, m4i23_0, m4i23_1);
  }
}

void fix_single_outlier_26_connected(
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

  printf("single outlier 26 connected not implemented!\n");
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
    throw std::runtime_error("This code only handles differences of a single grid space.");
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
    throw std::runtime_error("18: Should never happen.");
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

    m1.add_triangle(face_remap[f1], m1i13x, m1i23x);
    m1.add_triangle(face_remap[f1], face_remap[f2], m1i23x);

    m4.add_point(i13x);
    m4i13x = m4.last_face();

    m4.add_point(i23x);
    m4i23x = m4.last_face();

    m4.add_point(i13y);
    m4i13y = m4.last_face();

    m4.add_point(i23y);
    m4i23y = m4.last_face();

    m4.add_triangle(m4i13x, m4i13y, m4i23y);
    m4.add_triangle(m4i13x, m4i23x, m4i23y);

    m3.add_point(i13y);
    m3i13y = m3.last_face();

    m3.add_point(i23y);
    m3i23y = m3.last_face();

    m3.add_triangle(face_remap[f3], m3i13y, m3i23y);
  }
  else if (i13x.get(xaxis) > corner.get(xaxis) && i23x.get(xaxis) > corner.get(xaxis)) {
    m1.add_point(i13y);
    m1i13y = m1.last_face();
    m1.add_point(i23y);
    m1i23y = m1.last_face();

    m1.add_triangle(face_remap[f1], m1i13y, m1i23y);
    m1.add_triangle(face_remap[f1], face_remap[f2], m1i23y);

    m5.add_point(i13x);
    m5i13x = m5.last_face();

    m5.add_point(i23x);
    m5i23x = m5.last_face();

    m5.add_point(i13y);
    m5i13y = m5.last_face();

    m5.add_point(i23y);
    m5i23y = m5.last_face();

    m5.add_triangle(m5i13y, m5i13x, m5i23y);
    m5.add_triangle(m5i23y, m5i13x, m5i23x);

    m3.add_point(i13x);
    m3i13x = m3.last_face();

    m3.add_point(i23x);
    m3i23x = m3.last_face();

    m3.add_triangle(face_remap[f3], m3i13x, m3i23x);
  }
  else {
    m1.add_point(i13x);
    m1i13x = m1.last_face();

    m1.add_point(corner);
    m1corner = m1.last_face();

    m1.add_point(i23y);
    m1i23y = m1.last_face();

    m1.add_triangle(face_remap[f1], m1i13x, face_remap[f2]);
    m1.add_triangle(face_remap[f2], m1i13x, m1i23y);
    m1.add_triangle(m1i23y, m1i13x, m1corner);

    m4.add_point(i13x);
    m4i13x = m4.last_face();

    m4.add_point(i13y);
    m4i13y = m4.last_face();

    m4.add_point(corner);
    m4corner = m4.last_face();

    m4.add_triangle(m4i13x, m4i13y, m4corner);

    m5.add_point(i23y);
    m5i23y = m5.last_face();

    m5.add_point(corner);
    m5corner = m5.last_face();

    m5.add_point(i23x);
    m5i23x = m5.last_face();

    m5.add_triangle(m5i23y, m5corner, m5i23x);

    m3.add_point(i13x);
    m3i13x = m3.last_face();

    m3.add_point(corner);
    m3corner = m3.last_face();

    m3.add_point(i23x);
    m3i23x = m3.last_face();

    m3.add_triangle(m3i13x, face_remap[f3], m3corner);
    m3.add_triangle(m3corner, face_remap[f3], m3i23y);
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
    throw std::runtime_error("This code only handles differences of a single grid space.");
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

  const unsigned int m1f23 = m1.last_face();

  m1.add_triangle(face_remap[f1], m1f13, m1f23);
  m1.add_triangle(face_remap[f1], face_remap[f2], m1f23);

  m3.add_point(i13);
  
  const unsigned int m3f13 = m3.last_face();

  m3.add_point(i23);

  const unsigned int m3f23 = m3.last_face();

  m3.add_triangle(face_remap[f3], m3f13, m3f23);
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
  const unsigned int f3 // outlier
) {

  auto z1 = zones[f1];
  auto z2 = zones[f2];
  auto z3 = zones[f3];

  Vec3<int32_t> g1 = zone2grid(z1, gs);
  Vec3<int32_t> g3 = zone2grid(z3, gs);

  Vec3<int32_t> delta = g3 - g1;

  if (delta.num_non_zero_dims() == 1) {
    fix_single_outlier_6_connected(
      vertices, minpt, 
      face_remap, zones, 
      mesh_grid, cs, gs,
      f1, f2, f3
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
    fix_single_outlier_26_connected(
      vertices, minpt, 
      face_remap, zones, 
      mesh_grid, cs, gs,
      f1, f2, f3
    );
  }
  else {
    throw std::runtime_error("Non-zero delta was not 1,2, or 3.");
  }
}

std::vector<MeshObject> chunk_mesh_accelerated(
  const float* vertices, 
  const uint64_t num_vertices,
  const unsigned int* faces,
  const uint64_t num_faces,
  const float cx, const float cy, const float cz
) {

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

    if (!(zones[f1] == zones[f2] && zones[f1] == zones[f3])) {
      if (zones[f1] == zones[f2]) {
        fix_single_outlier(
          vertices, minpt, 
          face_remap, zones, 
          mesh_grid, cs, gs,
          f1, f2, f3
        );
      }
      else if (zones[f1] == zones[f3]) {
        fix_single_outlier(
          vertices, minpt, 
          face_remap, zones, 
          mesh_grid, cs, gs,
          f1, f3, f2
        );
      }
      else if (zones[f2] == zones[f3]) {
        fix_single_outlier(
          vertices, minpt, 
          face_remap, zones, 
          mesh_grid, cs, gs,
          f2, f3, f1
        );
      }
      else {
        fix_all_different(
          vertices, minpt, 
          face_remap, zones, 
          mesh_grid, cs, gs,
          f1, f2, f3
        );
      }

      continue;
    }

    auto zone = zones[f1];
    MeshObject& obj = mesh_grid[zone];
    obj.faces.push_back(face_remap[f1]);
    obj.faces.push_back(face_remap[f2]);
    obj.faces.push_back(face_remap[f3]);
  }

  return mesh_grid;
}


#endif
