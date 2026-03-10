#ifndef __ZMESH_UTILITY_HPP__
#define __ZMESH_UTILITY_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <vector>

#include "builtins.hpp"

namespace zmesh::utility {

template <typename T = float>
class Vec3 {
public:
  T x, y, z;
  Vec3() : x(0), y(0), z(0) {}
  Vec3(T x, T y, T z) : x(x), y(y), z(z) {}

  Vec3& operator--() {
    x--;
    y--;
    z--;
    return *this;
  }
  Vec3 operator--(int) {
    Vec3 old = *this;
    x--; 
    y--; 
    z--;
    return old;
  }
  Vec3& operator++() {
    x++;
    y++;
    z++;
    return *this;
  }
  Vec3 operator++(int) {
    Vec3 old = *this;
    x++; 
    y++; 
    z++;
    return old;
  }
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
  void operator+=(const T scalar) {
    x += scalar;
    y += scalar;
    z += scalar;
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
  void operator-=(const T scalar) {
    x -= scalar;
    y -= scalar;
    z -= scalar;
  }
  void operator-=(const Vec3& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
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
    switch (idx) {
      case 0:
        return x;
      case 1:
        return y;
      case 2:
        return z;
      default:
        throw std::runtime_error("Index out of bounds.");
    }
  }
  T get(const int idx) const {
    switch (idx) {
      case 0:
        return x;
      case 1:
        return y;
      case 2:
        return z;
      default:
        throw std::runtime_error("Index out of bounds.");
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

  template <typename T>
  void add_triangle(const Vec3<T>& face) {
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

  // Adapted from fqmr.hpp
  // https://github.com/Kramer84/pyfqmr-Fast-Quadric-Mesh-Reduction
  // MIT Licensed
  void load_obj(const std::string& filename) {
    points.clear();
    faces.clear();
    normals.clear();

    if (filename.size() == 0) {
      return;
    }
    
    // Using C I/O for performance reasons
    FILE* fn = fopen(filename.c_str(), "r");
    if (fn == NULL) {
      printf("File %s not found!\n", filename.c_str());
      return;
    }

    fseek(fn, 0, SEEK_END);
    long file_size = ftell(fn);
    rewind(fn);
    
    points.reserve(file_size / 10);
    faces.reserve(file_size / 10);
    
    char* line = NULL;
    size_t n = 0;
    int64_t len = 0;

    // Normals for OBJ are face-vertex, not vertex normals
    // so let's just skip them since we use vertex normals
    Vec3<float> v;
    Vec3<int64_t> f;
    int discard;

    auto check_f = [&](const Vec3<int64_t>& f) {
      // In OBJ, negative indices can be relative,
      // but they are not supported by this parser.
      if (f.x <= 0 || f.y <= 0 || f.z <= 0) {
        free(line);
        fclose(fn);
        throw std::runtime_error("load_obj: unsupported relative or zero face index.");
      }
    };

    while ((len = getline(&line, &n, fn)) != -1) {
      char *p, *end;
      
      if (len == 0 || line[0] == '\n') {
        continue;
      }
      // We don't work with materials or UVs for connectomics
      // so just skip parsing.
      else if (line[0] == 'v' && line[1] == ' ') {
        p = line + 2;
        v.x = strtof(p, &end); if (end == p) continue; p = end;
        v.y = strtof(p, &end); if (end == p) continue; p = end;
        v.z = strtof(p, &end); if (end == p) continue;
        add_point(v);
      }
      else if (line[0] == 'v' && (line[1] == 't' || line[1] == 'n') && line[2] == ' ') {
        continue;
      }
      else if (line[0] == 'f') {
        p = line + 2;
        char* slash = strchr(p, '/');

        // f -= 1 because obj faces are indexed from 1
        if (slash == NULL) {
          f.x = strtol(p, &end, 10); if (end == p) continue; p = end;
          f.y = strtol(p, &end, 10); if (end == p) continue; p = end;
          f.z = strtol(p, &end, 10); if (end == p) continue;
          check_f(f);
          --f;
          add_triangle(f);
        }
        else if (sscanf(line,"f %lld// %lld// %lld//", &f.x, &f.y, &f.z) == 3) {
          check_f(f);
          --f;
          add_triangle(f);
        }
        else if (
          sscanf(line,"f %lld//%d %lld//%d %lld//%d",
            &f.x, &discard,
            &f.y, &discard,
            &f.z, &discard) == 6
        ) {
          check_f(f);
          --f;
          add_triangle(f);
        }
        else if (sscanf(line,"f %lld/%d/%d %lld/%d/%d %lld/%d/%d",
          &f.x, &discard, &discard,
          &f.y, &discard, &discard,
          &f.z, &discard, &discard) == 9) {
          
          check_f(f);
          --f;
          add_triangle(f);
        }
        else {
          free(line);
          fclose(fn);
          throw std::runtime_error("load_obj: unrecognized face.");
        }
      }
      else if (strncmp(line, "mtllib", 6) == 0) {
        continue;
      }
      else if (strncmp(line, "usemtl", 6) == 0) {
        continue;
      }
    }

    free(line);
    fclose(fn);
  }

  void save_obj(const std::string& filename) {
    FILE *file = fopen(filename.c_str(), "w");

    if (!file) {
      printf("write_obj: can't write data file \"%s\".\n", filename.c_str());
      return;
    }

    for (size_t i = 0; i < points.size() / 3; i += 3) {
      fprintf(file, "v %.5f %.5f %.5f\n", points[i], points[i+1], points[i+2]);
    }

    for (size_t i = 0; i < faces.size() / 3; i += 3) {
      fprintf(file, "f %u %u %u\n", faces[i]+1, faces[i+1]+1, faces[i+2]+1);
    }

    fclose(file);
  }
};

};

#endif
