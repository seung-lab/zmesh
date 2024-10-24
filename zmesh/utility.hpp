#ifndef __ZMESH_UTILITY_HPP__
#define __ZMESH_UTILITY_HPP__

#include <vector>
#include <limits>
#include <cmath>
#include <cstdint>

struct MeshObject {
  std::vector<float> points;
  std::vector<float> normals;
  std::vector<unsigned int> faces;
};

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
  T operator[](const int idx) const {
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
  bool is_axis_aligned() const {
    return ((x != 0) + (y != 0) + (z != 0)) == 1;
  }
  void print(const std::string &name) const {
    if constexpr (std::is_same<T, float>::value) {
      printf("%s %.7f, %.3f, %.3f\n",name.c_str(), x, y, z);  
    }
    else {
      printf("%s %d, %d, %d\n",name.c_str(), x, y, z);
    }
  }
};

void fix_single_outlier(
  const float* vertices,
  const Vec3<float> minpt,
  const std::vector<uint32_t>& face_remap,
  const std::vector<uint32_t>& zones,
  std::vector<MeshObject>& mesh_grid,
  const Vec3<float>& cs,
  const Vec3<uint32_t>& gs,
  const unsigned int f1, // f1 and f2 should match on zone
  const unsigned int f2,
  const unsigned int f3 // outlier
) {

  Vec3 v1(vertices[3*f1+0], vertices[3*f1+1], vertices[3*f1+2]);
  Vec3 v2(vertices[3*f2+0], vertices[3*f2+1], vertices[3*f2+2]);
  Vec3 v3(vertices[3*f3+0], vertices[3*f3+1], vertices[3*f3+2]);

  auto z1 = zones[f1];
  auto z3 = zones[f3];

  auto zone2grid = [&](uint32_t zone) {
    uint32_t z = zone / gs.x / gs.y;
    uint32_t y = (zone - gs.x * gs.y * z) / gs.x;
    uint32_t x = zone - gs.x * (y + gs.y * z);
    return Vec3<uint32_t>(x,y,z);
  }; 

  Vec3<uint32_t> g1 = zone2grid(z1);
  Vec3<uint32_t> g3 = zone2grid(z3);

  auto delta = g3 - g1;

  int axis = 0;
  if (delta.y != 0) {
    axis = 1;
  }
  else if (delta.z != 0) {
    axis = 2;
  }

  const float plane_offset = minpt[axis] + std::max(g1[axis], g3[axis]) * cs[axis];

  auto intersect_x_fn = [&](const Vec3<float> &p, const Vec3<float> &q) {
    float t = (plane_offset - p[axis]) / (p[axis] - q[axis]);
    return p + (p - q) * t;
  };

  const Vec3 i13 = intersect_x_fn(v1, v3);
  const Vec3 i23 = intersect_x_fn(v2, v3);

  MeshObject& m1 = mesh_grid[z1];
  MeshObject& m3 = mesh_grid[z3];

  m1.points.push_back(i13.x);
  m1.points.push_back(i13.y);
  m1.points.push_back(i13.z);

  const unsigned int m1f13 = (m1.points.size() - 1) / 3;

  m1.points.push_back(i23.x);
  m1.points.push_back(i23.y);
  m1.points.push_back(i23.z);

  const unsigned int m1f23 = (m1.points.size() - 1) / 3;

  m1.faces.push_back(face_remap[f1]);
  m1.faces.push_back(m1f13);
  m1.faces.push_back(m1f23);

  m1.faces.push_back(face_remap[f1]);
  m1.faces.push_back(face_remap[f2]);
  m1.faces.push_back(m1f23);

  m3.points.push_back(i13.x);
  m3.points.push_back(i13.y);
  m3.points.push_back(i13.z);

  const unsigned int m3f13 = (m3.points.size() - 1) / 3;

  m3.points.push_back(i23.x);
  m3.points.push_back(i23.y);
  m3.points.push_back(i23.z);

  const unsigned int m3f23 = (m3.points.size() - 1) / 3;

  m3.faces.push_back(face_remap[f3]);
  m3.faces.push_back(m3f13);
  m3.faces.push_back(m3f23);
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

  const uint32_t gx = std::max(static_cast<uint32_t>(((max_x - min_x) / cx) + 0.5), static_cast<uint32_t>(1));
  const uint32_t gy = std::max(static_cast<uint32_t>(((max_y - min_y) / cy) + 0.5), static_cast<uint32_t>(1));
  const uint32_t gz = std::max(static_cast<uint32_t>(((max_z - min_z) / cz) + 0.5), static_cast<uint32_t>(1));

  const Vec3<uint32_t> gs(gx,gy,gz);

  std::vector<uint32_t> zones(num_vertices);

  const float epsilon = 1e-4;

  const float icx = 1 / cx;
  const float icy = 1 / cy;
  const float icz = 1 / cz;

  for (uint64_t i = 0, j = 0; j < num_vertices; i += 3, j++) {
    int ix = static_cast<int>((vertices[i] - min_x - epsilon) * icx) ;
    int iy = static_cast<int>((vertices[i+1] - min_y - epsilon) * icy);
    int iz = static_cast<int>((vertices[i+2] - min_z - epsilon) * icz);

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
      // else {
      //   // do nothing
      // }

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
