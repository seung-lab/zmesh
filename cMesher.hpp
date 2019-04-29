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
  std::vector<uint32_t> voxelresolution_;

 public:
  CMesher(const std::vector<uint32_t> &voxelresolution) {
    voxelresolution_ = voxelresolution;
  }
  ~CMesher() {};

  void mesh(
      const std::vector<LabelType> &data, 
      unsigned int sx, unsigned int sy, unsigned int sz
    ) {
    // Create Marching Cubes class for type T volume

    const LabelType *a = &data[0];
    // Run global marching cubes, a mesh is generated for each segment ID group
    marchingcubes_.marche(a, sx, sy, sz);
  }

  std::vector<LabelType> ids() {
    std::vector<LabelType> keys;
    for (auto it = marchingcubes_.meshes().begin();
         it != marchingcubes_.meshes().end(); ++it) {
      keys.push_back(it->first);
    }

    return keys;
  }

  MeshObject get_mesh(
      LabelType segid, bool generate_normals,
      int simplification_factor,
      int max_simplification_error
    ) {

    MeshObject obj;

    if (marchingcubes_.count(segid) == 0) {  // MC produces no triangles if either
                                          // none or all voxels were labeled!
      return obj;
    }

    zi::mesh::int_mesh<PositionType, LabelType> im;
    im.add(marchingcubes_.get_triangles(segid));
    im.template fill_simplifier<SimplifierType>(
      simplifier_, 
      0, 0, 0, 
      voxelresolution_[2], voxelresolution_[1], voxelresolution_[0]
    );
    simplifier_.prepare(generate_normals);

    if (simplification_factor > 0) {
      // This is the most cpu intensive line
      simplifier_.optimize(
          simplifier_.face_count() / simplification_factor,
          max_simplification_error
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

    return obj;
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
