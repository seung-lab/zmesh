import numpy as np
import struct

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

  def __len__(self):
    return self.vertices.shape[0]

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

  def __repr__(self):
    return "Mesh(vertices<{}>, faces<{}>, normals<{}>)".format(
      self.vertices.shape[0], self.faces.shape[0], 
      (None if self.normals is None else self.normals.shape[0])
    )

  def clone(self):
    if self.normals is None:
      return Mesh(np.copy(self.vertices), np.copy(self.faces), None)
    else:
      return Mesh(np.copy(self.vertices), np.copy(self.faces), np.copy(self.normals))

  @classmethod
  def from_precomputed(self, binary):
    """
    Mesh from_precomputed(self, binary)

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

  def to_obj(self):
    """Return a string representing a .obj file."""
    objdata = []
    objdata += [ 'v {:.1f} {:.1f} {:.1f}'.format(*vertex) for vertex in self.vertices ]
    objdata += [ 'f {} {} {}'.format(*face) for face in (self.faces+1) ] # obj is 1 indexed
    objdata = '\n'.join(objdata) + '\n'
    return objdata.encode('utf8')

  def to_ply(self):
    """
    Return a bytearray in .ply format, 
    a more compact format than .obj that's still widely readable.
    """
    vertexct = self.vertices.shape[0]
    trianglect = self.faces.shape[0]

    # Header
    plydata = bytearray("""ply
format binary_little_endian 1.0
element vertex {}
property float x
property float y
property float z
element face {}
property list int int vertex_indices
end_header
""".format(vertexct, trianglect).encode('utf8'))

    # Vertex data (x y z): "fff" 
    plydata.extend(self.vertices.tobytes('C'))

    # Faces (3 f1 f2 f3): "3iii" 
    plydata.extend(
      np.insert(self.faces, 0, 3, axis=1).tobytes('C')
    )

    return plydata
