# distutils: language = c++
# distutils: sources = cMesher.cpp

from libc.stdint cimport uint64_t, uint32_t, uint16_t, uint8_t
from libcpp.vector cimport vector
from libcpp cimport bool

import numpy as np
import struct

__version__ = '0.1.0'

class Mesh:
  """
  Represents the vertices, faces, and normals of a mesh
  as numpy arrays.

  class Mesh:
    ndarray[float32, ndim=2] self.vertices: [ [x,y,z], ... ]
    ndarray[uint32,  ndim=2] self.faces:    [ [v1,v2,v3], ... ]
    ndarray[float32, ndim=2] self.normals:  [ [nx,ny,nz], ... ]

  """
  def __init__(self, vertices, faces, normals):
    self.vertices = vertices
    self.faces = faces
    self.normals = normals

  def __eq__(self, other):
    """Tests strict equality between two meshes."""

    if self.normals is None and other.normals is not None:
      return False
    elif self.normals is not None and other.normals is None:
      return False

    equality = np.all(self.vertices == other.vertices) \
      and np.all(self.faces == other.faces)

    if self.normals is None:
      return equality

    return (equality and np.all(self.normals == other.normals))

  def clone(self):
    if self.normals is None:
      return Mesh(np.copy(self.vertices), np.copy(self.faces), None)
    else:
      return Mesh(np.copy(self.vertices), np.copy(self.faces), np.copy(self.normals))

  @classmethod
  def from_precomputed(self, bytes binary):
    """
    Mesh from_precomputed(self, bytes binary)

    Decode a Precomputed format mesh from a byte string.
    
    Format:
      uint32        Nv * float32 * 3   uint32 * 3 until end
      Nv            (x,y,z)            (v1,v2,v2)
      N Vertices    Vertices           Faces
    """
    num_vertices = struct.unpack("=I", binary[0:4])[0]
    try:
      # count=-1 means all data in buffer
      vertices = np.frombuffer(binary, dtype=np.float32, count=3*num_vertices, offset=4)
      faces = np.frombuffer(binary, dtype=np.uint32, count=-1, offset=(4 + 12 * num_vertices)) 
    except ValueError:
      raise ValueError("""
        The input buffer is too small for the Precomputed format.
        Minimum Bytes: {} 
        Actual Bytes: {}
      """.format(4 + 4 * num_vertices, len(binary)))

    vertices = vertices.reshape(num_vertices, 3)
    faces = faces.reshape(faces.size // 3, 3)

    return Mesh(vertices, faces, normals=None)

  def to_precomputed(self):
    """
    bytes to_precomputed(self)

    Convert mesh into binary format compatible with Neuroglancer.
    Does not preserve normals.
    """
    vertex_index_format = [
      np.uint32(self.vertices.shape[0]), # Number of vertices (3 coordinates)
      self.vertices,
      self.faces
    ]
    return b''.join([ array.tobytes('C') for array in vertex_index_format ])

  def __len__(self):
    return self.vertices.shape[0]

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

  def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8):
    """
    get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8)

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
  
  def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8):
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
  
  def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8):
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
  
  def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8):
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
  
  def get_mesh(self, mesh_id, normals=False, simplification_factor=0, max_simplification_error=8):
    return self.ptr.get_mesh(mesh_id, normals, simplification_factor, max_simplification_error)
  
  def clear(self):
    self.ptr.clear()

  def erase(self, mesh_id):
    return self.ptr.erase(mesh_id)
