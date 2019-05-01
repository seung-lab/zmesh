# distutils: language = c++
# distutils: sources = cMesher.cpp

from libc.stdint cimport uint64_t, uint32_t, uint16_t, uint8_t
from libcpp.vector cimport vector
from libcpp cimport bool

import numpy as np
from zmesh.mesh import Mesh

cdef extern from "cMesher.hpp":
  cdef struct MeshObject:
    vector[float] points
    vector[float] normals
    vector[unsigned int] faces

  cdef cppclass CMesher[P,L,S]:
    CMesher(vector[uint32_t] voxel_res) except +
    void mesh(vector[L], unsigned int, unsigned int, unsigned int)
    vector[L] ids()
    MeshObject get_mesh(L segid, bool normals, int simplification_factor, int max_simplification_error)
    bool erase(L segid)
    void clear()

class Mesher:
  def __init__(self, voxel_res):
    voxel_res = np.array(voxel_res, dtype=np.uint32)
    self._mesher = Mesher6464(voxel_res)
    self.voxel_res = voxel_res

  def mesh(self, data):
    del self._mesher

    shape = np.array(data.shape)
    nbytes = np.dtype(data.dtype).itemsize

    # z is allocated position 10 bits, y 11, and x 11 bits.
    # For some reason, x has an overflow error if you 
    # try to use the last bit, so we play it safe and 
    # restrict it to 10 bits.
    if shape[0] > 511 or shape[1] > 1023 or shape[2] > 511:
      MesherClass = Mesher6432 if nbytes <= 4 else Mesher6464
    else:
      MesherClass = Mesher3232 if nbytes <= 4 else Mesher3264

    self._mesher = MesherClass(self.voxel_res)

    return self._mesher.mesh(data)

  def ids(self):
    return self._mesher.ids()

  def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=40):
    """
    get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=40)

    Returns: MeshObject
    """
    mesh = self._mesher.get_mesh(
      mesh_id, normals, simplification_factor, max_simplification_error
    )

    points = np.array(mesh['points'], dtype=np.float32)
    points /= 2.0
    Nv = points.size // 3
    Nf = len(mesh['faces']) // 3

    points = points.reshape(Nv, 3)
    faces = np.array(mesh['faces'], dtype=np.uint32).reshape(Nf, 3)

    normals = None
    if mesh['normals']:
      normals = np.array(mesh['normals'], dtype=np.float32).reshape(Nv, 3)

    return Mesh(points, faces, normals)
  
  def clear(self):
    self._mesher.clear()
   
  def erase(self, segid):
    return self._mesher.erase(segid)


# NOTE: THE USER INTERFACE CLASS IS "Mesher" ABOVE

# The following wrapper classes are all similar
# and could be generated from a template. 

# The pattern is: Mesher[PositionType bits][LabelType bits]
# 64 bit position type can represent very large arrays
# 32 bit can represent 511 x 1023 x 511 arrays
#
# Because the 64 bit version can handle very large arrays, it also
# gets the double type simplifier since maybe precision will matter 
# more.

cdef class Mesher6464:
  cdef CMesher[uint64_t, uint64_t, double] *ptr      # hold a C++ instance which we're wrapping

  def __cinit__(self, voxel_res):
    self.ptr = new CMesher[uint64_t, uint64_t, double](voxel_res.astype(np.uint32))

  def __dealloc__(self):
    del self.ptr

  def mesh(self, data):
    self.ptr.mesh(
      data.astype(np.uint64).flatten(), 
      data.shape[0], data.shape[1], data.shape[2]
    )

  def ids(self):
    return self.ptr.ids()
  
  def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=40):
    return self.ptr.get_mesh(mesh_id, normals, simplification_factor, max_simplification_error)
  
  def clear(self):
    self.ptr.clear()

  def erase(self, mesh_id):
    return self.ptr.erase(mesh_id)

cdef class Mesher6432:
  cdef CMesher[uint64_t, uint32_t, double] *ptr      # hold a C++ instance which we're wrapping

  def __cinit__(self, voxel_res):
    self.ptr = new CMesher[uint64_t, uint32_t, double](voxel_res.astype(np.uint32))

  def __dealloc__(self):
    del self.ptr

  def mesh(self, data):
    self.ptr.mesh(
      data.astype(np.uint64).flatten(), 
      data.shape[0], data.shape[1], data.shape[2]
    )

  def ids(self):
    return self.ptr.ids()
  
  def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=40):
    return self.ptr.get_mesh(mesh_id, normals, simplification_factor, max_simplification_error)
  
  def clear(self):
    self.ptr.clear()

  def erase(self, mesh_id):
    return self.ptr.erase(mesh_id)

cdef class Mesher3264:
  cdef CMesher[uint32_t, uint64_t, float] *ptr      # hold a C++ instance which we're wrapping

  def __cinit__(self, voxel_res):
    self.ptr = new CMesher[uint32_t, uint64_t, float](voxel_res.astype(np.uint32))

  def __dealloc__(self):
    del self.ptr

  def mesh(self, data):
    self.ptr.mesh(
      data.astype(np.uint64).flatten(), 
      data.shape[0], data.shape[1], data.shape[2]
    )

  def ids(self):
    return self.ptr.ids()
  
  def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=40):
    return self.ptr.get_mesh(mesh_id, normals, simplification_factor, max_simplification_error)
  
  def clear(self):
    self.ptr.clear()

  def erase(self, mesh_id):
    return self.ptr.erase(mesh_id)

cdef class Mesher3232:
  cdef CMesher[uint32_t, uint32_t, float] *ptr      # hold a C++ instance which we're wrapping

  def __cinit__(self, voxel_res):
    self.ptr = new CMesher[uint32_t, uint32_t, float](voxel_res.astype(np.uint32))

  def __dealloc__(self):
    del self.ptr

  def mesh(self, data):
    self.ptr.mesh(
      data.astype(np.uint32).flatten(), 
      data.shape[0], data.shape[1], data.shape[2]
    )

  def ids(self):
    return self.ptr.ids()
  
  def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=40):
    return self.ptr.get_mesh(mesh_id, normals, simplification_factor, max_simplification_error)
  
  def clear(self):
    self.ptr.clear()

  def erase(self, mesh_id):
    return self.ptr.erase(mesh_id)
