#ifndef CMESHER_H
#define CMESHER_H

#include <vector>
#include <zi/mesh/marching_cubes.hpp>
#include <zi/mesh/int_mesh.hpp>
#include <zi/mesh/quadratic_simplifier.hpp>
#include <zi/vl/vec.hpp>

struct MeshObject {
  std::vector<float> points;
  std::vector<float> normals;
  std::vector<unsigned int> faces;
};

template <typename PositionType, typename LabelType, typename SimplifierType>
class CMesher {
 private:
  zi::mesh::marching_cubes<PositionType, LabelType> marchingcubes_;
  zi::mesh::simplifier<SimplifierType> simplifier_;
  std::vector<float> voxelresolution_;

 public:
  CMesher(const std::vector<float> &voxelresolution) {
    voxelresolution_ = voxelresolution;
  }
  ~CMesher() {};

  void mesh(
      const LabelType* data, 
      const size_t sx, const size_t sy, const size_t sz,
      const bool c_order = true
    ) {
    // Run global marching cubes, a mesh is generated for each segment ID group
    marchingcubes_.marche(data, sx, sy, sz, c_order);
  }

  std::vector<LabelType> ids() {
    std::vector<LabelType> keys;
    for (auto it = marchingcubes_.meshes().begin();
         it != marchingcubes_.meshes().end(); ++it) {
      keys.push_back(it->first);
    }

    return keys;
  }

  PositionType pack_coords(PositionType x, PositionType y, PositionType z) {
    return marchingcubes_.pack_coords(x,y,z);
  }

  MeshObject get_mesh(
    LabelType segid, 
    bool generate_normals,
    int simplification_factor,
    float max_simplification_error,
    float min_simplification_error,
    bool transpose = true
  ) {

    // MC produces no triangles if either
    // none or all voxels were labeled.
    MeshObject empty_obj;
    if (marchingcubes_.count(segid) == 0) { 
      return empty_obj;
    }

    std::vector< zi::vl::vec< PositionType, 3> > triangles = marchingcubes_.get_triangles(segid);

    return simplify(
      triangles,
      generate_normals,
      simplification_factor,
      max_simplification_error,
      min_simplification_error,
      transpose
    );
  }

  MeshObject simplify(      
      const std::vector< zi::vl::vec< PositionType, 3> >& triangles,
      bool generate_normals,
      int simplification_factor,
      float max_simplification_error,
      float min_simplification_error,

      // This is to support the old broken version
      // w/ backwards compatibility
      bool transpose = true 
    ) {

    MeshObject obj;

    zi::mesh::int_mesh<PositionType, LabelType> im;
    im.add(triangles);

    float wx = voxelresolution_[0];
    float wy = voxelresolution_[1]; 
    float wz = voxelresolution_[2];
    if (transpose) {
      wx = voxelresolution_[2];
      wy = voxelresolution_[1];
      wz = voxelresolution_[0];
    }

    im.template fill_simplifier<SimplifierType>(
      simplifier_, 
      0, 0, 0, 
      wx, wy, wz
    );

    if (simplification_factor > 0) {
      simplifier_.prepare(generate_normals);
      // This is the most cpu intensive line
      simplifier_.optimize(
          simplifier_.face_count() / simplification_factor,
          max_simplification_error,
          min_simplification_error
      );
    }

    std::vector<zi::vl::vec<SimplifierType, 3> > points;
    std::vector<zi::vl::vec<SimplifierType, 3> > normals;
    std::vector<zi::vl::vec<unsigned, 3> > faces;

    simplifier_.get_faces(points, normals, faces);
    obj.points.reserve(3 * points.size());
    obj.faces.reserve(3 * faces.size());

    if (generate_normals) {
      obj.normals.reserve(3 * points.size());
    }

    if (transpose) {
      for (auto v = points.begin(); v != points.end(); ++v) {
        obj.points.push_back((*v)[2]);
        obj.points.push_back((*v)[1]);
        obj.points.push_back((*v)[0]);
      }

      if (generate_normals) {
        for (auto vn = normals.begin(); vn != normals.end(); ++vn) {
          obj.normals.push_back((*vn)[2]);
          obj.normals.push_back((*vn)[1]);
          obj.normals.push_back((*vn)[0]);
        }
      }

      for (auto f = faces.begin(); f != faces.end(); ++f) {
        obj.faces.push_back((*f)[0]);
        obj.faces.push_back((*f)[2]);
        obj.faces.push_back((*f)[1]);
      }
    }
    else {
      for (auto v = points.begin(); v != points.end(); ++v) {
        obj.points.push_back((*v)[0]);
        obj.points.push_back((*v)[1]);
        obj.points.push_back((*v)[2]);
      }

      if (generate_normals) {
        for (auto vn = normals.begin(); vn != normals.end(); ++vn) {
          obj.normals.push_back((*vn)[0]);
          obj.normals.push_back((*vn)[1]);
          obj.normals.push_back((*vn)[2]);
        }
      }

      for (auto f = faces.begin(); f != faces.end(); ++f) {
        obj.faces.push_back((*f)[1]);
        obj.faces.push_back((*f)[2]);
        obj.faces.push_back((*f)[0]);
      }
    }

    return obj;
  }

  MeshObject simplify_points(
    const uint64_t* points,
    const size_t Nv,
    bool generate_normals,
    int simplification_factor,
    float max_simplification_error,
    float min_simplification_error
  ) {

    std::vector< zi::vl::vec< PositionType, 3> > triangles;
    triangles.reserve(Nv);

    for (size_t i = 0; i < Nv; i++) {
      triangles.push_back(
        zi::vl::vec<PositionType, 3>(
          points[3 * i + 0], 
          points[3 * i + 1], 
          points[3 * i + 2]
        )
      );
    }

    return simplify(
      triangles, 
      generate_normals, 
      simplification_factor, 
      max_simplification_error,
      min_simplification_error,
      false
    );
  }

  void clear() {
    marchingcubes_.clear();
    simplifier_.clear();
  }

  bool erase(LabelType segid) {
    return marchingcubes_.erase(segid);
  }
};


#endif
