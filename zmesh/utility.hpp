#ifndef __ZMESH_UTILITY_HPP__
#define __ZMESH_UTILITY_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

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
  float len() const {
    return sqrt(x*x + y*y + z*z);
  }
  float len2() const {
    return x*x + y*y + z*z;
  }
  Vec3 hat() const {
    const float l = len();
    Vec3 ret(x,y,z);
    if (l == 1) {
      return ret;
    }
    ret.x /= l;
    ret.y /= l;
    ret.z /= l;
    return ret;
  }
  bool close(const Vec3& o) const {
    return (*this - o).len2() < 1e-4;
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

};

#endif
