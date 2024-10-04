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

std::vector<MeshObject> chunk_mesh(
  std::vector<float> vertices,
  std::vector<unsigned int> faces,
  const uint32_t cx, const uint32_t cy, const uint32_t cz,
) {

  float min_x = INFINITY;
  float min_y = INFINITY;
  float min_z = INFINITY;
  float max_x = -INFINITY;
  float max_y = -INFINITY;
  float max_z = -INFINITY;

  for (uint64_t i = 0; i < vertices.size(); i += 3) {
    min_x = std::min(min_x, vertices[i]);
    max_x = std::max(max_x, vertices[i]);

    min_y = std::min(min_y, vertices[i+1]);
    max_y = std::max(max_y, vertices[i+1]);

    min_z = std::min(min_z, vertices[i+2]);
    max_z = std::max(max_z, vertices[i+2]);
  }

  uint32_t gx = static_cast<uint32_t>(((max_x - min_x) / static_cast<float>(cx)) + 0.5);
  uint32_t gy = static_cast<uint32_t>(((max_y - min_y) / static_cast<float>(cy)) + 0.5);
  uint32_t gz = static_cast<uint32_t>(((max_z - min_z) / static_cast<float>(cz)) + 0.5);

  gx = std::max(gx, 1);
  gy = std::max(gy, 1);
  gz = std::max(gz, 1);

  std::vector<uint32_t> zones(vertices.size() / 3);

  for (uint64_t i = 0, j = 0; i < vertices.size(); i += 3, j++) {
    zones[j] = (
      static_cast<int64_t>(vertices[i] - min_x + 0.5)
      + gx * static_cast<int64_t>(vertices[i+1] - min_y + 0.5)
      + gx * gy * (static_cast<int64_t>(vertices[i+2] - min_z + 0.5))
    );
  }

  std::vector<MeshObject> mesh_grid(gx * gy * gz);
  std::vector<robin_hood::unordered_flat_map<uint32_t, uint32_t>> 
    face_remap(gx * gy * gz);

  for (uint32_t i = 0; i < zones.size(); i++) {
    uint32_t zone = zones[i];
    MeshObject& obj = mesh_grid[zone];
    obj.points.push_back(vertices[i * 3 + 0]);
    obj.points.push_back(vertices[i * 3 + 1]);
    obj.points.push_back(vertices[i * 3 + 2]);
    face_remap[zone][i] = obj.points.size() / 3;
  }

  for (uint64_t i = 0; i < faces.size(); i += 3) {
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
