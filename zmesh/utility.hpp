#ifndef __ZMESH_UTILITY_HPP__
#define __ZMESH_UTILITY_HPP__

#include <vector>
#include <limits>
#include <cmath>
#include <cstdint>

#include <zi/utility/robin_hood.hpp>

struct MeshObject {
  std::vector<float> points;
  std::vector<float> normals;
  std::vector<unsigned int> faces;
};

std::vector<MeshObject> chunk_mesh_accelerated(
  float* vertices, 
  const uint64_t num_vertices,
  unsigned int* faces,
  const uint64_t num_faces,
  const float cx, const float cy, const float cz
) {

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

  const uint32_t gx = std::max(static_cast<uint32_t>(((max_x - min_x) / cx) + 0.5), static_cast<uint32_t>(1));
  const uint32_t gy = std::max(static_cast<uint32_t>(((max_y - min_y) / cy) + 0.5), static_cast<uint32_t>(1));
  const uint32_t gz = std::max(static_cast<uint32_t>(((max_z - min_z) / cz) + 0.5), static_cast<uint32_t>(1));

  std::vector<uint32_t> zones(num_vertices);

  for (uint64_t i = 0, j = 0; j < num_vertices; i += 3, j++) {
    uint32_t ix = static_cast<uint32_t>(static_cast<int64_t>(vertices[i] - min_x + 0.5) / cx);
    uint32_t iy = static_cast<uint32_t>(static_cast<int64_t>(vertices[i+1] - min_y + 0.5) / cy);
    uint32_t iz = static_cast<uint32_t>(static_cast<int64_t>(vertices[i+2] - min_z + 0.5) / cz);
    zones[j] = ix + gx * (iy + gy * iz);
  }

  std::vector<MeshObject> mesh_grid(gx * gy * gz);
  std::vector<robin_hood::unordered_flat_map<uint32_t, uint32_t>> 
    face_remap(gx * gy * gz);

  for (uint32_t i = 0; i < num_vertices; i++) {
    uint32_t zone = zones[i];
    MeshObject& obj = mesh_grid[zone];
    obj.points.push_back(vertices[i * 3 + 0]);
    obj.points.push_back(vertices[i * 3 + 1]);
    obj.points.push_back(vertices[i * 3 + 2]);
    face_remap[zone][i] = obj.points.size() / 3;
  }

  for (uint64_t i = 0; i < num_faces * 3; i += 3) {
    auto f1 = faces[i+0];
    auto f2 = faces[i+1];
    auto f3 = faces[i+2];

    if (!(zones[f1] == zones[f2] && zones[f1] == zones[f3])) {
      continue;
    }

    auto zone = zones[f1];
    MeshObject& obj = mesh_grid[zone];
    auto& remap = face_remap[zone];
    obj.faces.push_back(remap[f1]);
    obj.faces.push_back(remap[f2]);
    obj.faces.push_back(remap[f3]);
  }

  return mesh_grid;
}


#endif
