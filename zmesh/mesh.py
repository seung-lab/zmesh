from typing import Optional, Iterator, Union

import numpy as np
import numpy.typing as npt
import re
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
  def __init__(
    self, 
    vertices:Optional[npt.NDArray[np.float32]] = None, 
    faces:Optional[npt.NDArray[np.uint32]] = None, 
    normals:Optional[npt.NDArray[np.float32]] = None, 
    id:int = None,
  ):
    if vertices is None:
      self.vertices = np.zeros([0,3], dtype=np.float32)
    else:
      self.vertices = np.asarray(vertices, dtype=np.float32)
    
    face_dtype = np.uint32
    if self.vertices.shape[0] > np.iinfo(np.uint32).max:
      face_dtype = np.uint64

    if faces is None:
      self.faces = np.zeros([0,3], dtype=face_dtype)
    else:
      self.faces = np.asarray(faces, dtype=face_dtype)

    if normals is None:
      self.normals = normals
    else:
      self.normals = np.asarray(normals, dtype=np.float32)

    self.id = id

  @property
  def segid(self):
    return self.id

  @segid.setter
  def segid(self, val:int):
    self.id = val

  def __len__(self) -> int:
    return self.vertices.shape[0]

  def __eq__(self, other) -> bool:
    """Tests strict equality between two meshes."""

    no_self_normals = self.normals is None or self.normals.size == 0
    no_other_normals = other.normals is None or other.normals.size == 0

    if no_self_normals != no_other_normals:
      return False
       
    equality = np.all(self.vertices == other.vertices) \
      and np.all(self.faces == other.faces)

    if no_self_normals:
      return equality

    return (equality and np.all(self.normals == other.normals))

  def __repr__(self) -> str:
    return "Mesh(vertices<{}>, faces<{}>, normals<{}>)".format(
      self.vertices.shape[0], self.faces.shape[0], 
      (None if self.normals is None else self.normals.shape[0])
    )

  def empty(self) -> bool:
    return self.faces.size == 0 or self.vertices.size == 0

  @property
  def nbytes(self) -> int:
    nbytes = self.vertices.nbytes if self.vertices is not None else 0
    nbytes += self.faces.nbytes if self.faces is not None else 0
    nbytes += self.normals.nbytes if self.normals is not None else 0
    return nbytes

  def clone(self):
    if self.normals is None:
      return Mesh(np.copy(self.vertices), np.copy(self.faces), None, id=self.id)
    else:
      return Mesh(np.copy(self.vertices), np.copy(self.faces), np.copy(self.normals), id=self.id)

  def triangles(self) -> npt.NDArray[np.float32]:
    """Returns vertex triples representing triangluar faces."""
    return self.vertices[self.faces]

  @classmethod
  def concatenate(cls, *meshes, id:Optional[int] = None) -> "Mesh":
    vertex_ct = np.zeros(len(meshes) + 1, np.uint64)
    vertex_ct[1:] = np.cumsum([ len(mesh) for mesh in meshes ])

    face_dtype = np.uint64
    if vertex_ct[-1] < np.iinfo(np.uint32).max:
      vertex_ct = vertex_ct.astype(np.uint32)
      face_dtype = np.uint32

    vertices = np.concatenate([ mesh.vertices for mesh in meshes ])
    
    faces = np.concatenate([ 
      mesh.faces.astype(face_dtype, copy=False) + vertex_ct[i]
      for i, mesh in enumerate(meshes) 
    ])

    # normals = np.concatenate([ mesh.normals for mesh in meshes ])

    return Mesh(vertices, faces, None, id=id)

  def merge_close_vertices(self, radius:float = 1e-5) -> "Mesh":
    """Merge vertices that are closer to each other than radius."""
    from scipy.spatial import cKDTree as KDTree

    if radius is None:
      radius = np.inf

    if radius is not None and radius <= 0:
      raise ValueError("radius must be greater than zero: " + str(radius))

    mesh = self.consolidate()

    tree = KDTree(mesh.vertices)
    pairs = tree.query_pairs(
      r=radius,
      p=2, # euclidean distance
      eps=0, # approximate search
      output_type='ndarray',
    )

    remap = np.arange(len(mesh.vertices), dtype=np.uint32)
    remap[pairs[:,1]] = pairs[:,0]

    mesh.faces = remap[mesh.faces]
    return mesh.consolidate()

  def remove_unreferenced_vertices(self) -> "Mesh":
    if self.empty():
      return Mesh([], [], normals=None)

    visited_faces = np.zeros([len(self.vertices)], dtype=bool)
    visited_faces[self.faces] = True
    unreferenced_f = np.where(visited_faces == False)[0]

    verts = np.delete(self.vertices, unreferenced_f, axis=0)

    mapping = np.cumsum(visited_faces)
    mapping -= 1

    faces = self.faces.copy()
    faces = mapping[faces]

    return Mesh(verts, faces, None)

  def remove_degenerate_faces(self) -> "Mesh":
    """Remove faces that reference the same vertex two or three times."""
    if self.empty():
      return Mesh([], [], normals=None)

    # find degenerate faces
    f = self.faces
    index = np.where((f[:,0] == f[:,1]) | (f[:,1] == f[:,2]) | (f[:,0] == f[:,2]))
    f = np.delete(f, index, axis=0)

    # find duplicate faces e.g. f1,f2,f3  ; f1,f3,f2 which won't be found without sorting
    f_sorted = np.sort(f, axis=1)
    _, unique_indices = np.unique(f_sorted, axis=0, return_index=True)
    f = f[unique_indices]

    return Mesh(self.vertices, f, self.normals, id=self.id)

  def consolidate(self) -> "Mesh":
    """Remove duplicate vertices and faces and degenerate faces. Returns a new mesh object."""
    if self.empty():
      return Mesh([], [], normals=None)

    vertices = self.vertices
    faces = self.faces
    normals = self.normals

    eff_verts, uniq_idx, idx_representative = np.unique(
      vertices, axis=0, return_index=True, return_inverse=True
    )

    eff_faces = idx_representative[faces]
    eff_faces = np.unique(eff_faces, axis=0)

    eff_normals = None
    if normals is not None and normals.size > 0:
      eff_normals = normals[uniq_idx]

    mesh = Mesh(eff_verts, eff_faces, eff_normals, id=self.id)
    mesh = mesh.remove_degenerate_faces()
    return mesh.remove_unreferenced_vertices()

  @classmethod
  def from_precomputed(self, binary:bytes) -> "Mesh":
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

  def to_precomputed(self) -> bytes:
    """
    bytes to_precomputed(self)

    Convert mesh into binary format compatible with Neuroglancer.
    Does not preserve normals.
    """
    vertex_index_format = [
      np.uint32(self.vertices.shape[0]), # Number of vertices (3 coordinates)
      self.vertices,
      self.faces.astype(np.uint32, copy=False)
    ]
    return b''.join([ array.tobytes('C') for array in vertex_index_format ])

  @classmethod
  def from_obj(self, text:Union[str,bytes]):
    """Given a string representing a Wavefront OBJ file, decode to a zmesh.Mesh."""

    vertices = []
    faces = []
    normals = []

    if isinstance(text, bytes):
      text = text.decode('utf8')

    for line in text.split('\n'):
      line = line.strip()
      if len(line) == 0:
        continue
      elif line[0] == '#':
        continue
      elif line[0] == 'f':
        if line.find('/') != -1:
          # e.g. f 6092/2095/6079 6087/2092/6075 6088/2097/6081
          # i.e. f vertex_1/texture_1/normal_1 etc
          (v1, vt1, vn1, v2, vt2, vn2, v3, vt3, vn3) = re.match(r'f\s+(\d+)/(\d*)?/(\d+)?\s+(\d+)/(\d*)?/(\d+)?\s+(\d+)/(\d*)?/(\d+)?', line).groups()
        else:
          (v1, v2, v3) = re.match(r'f\s+(\d+)\s+(\d+)\s+(\d+)', line).groups()
        faces.append( (int(v1), int(v2), int(v3)) )
      elif line[0] == 'v':
        if line[1] == 't': # vertex textures not supported
          # e.g. vt 0.351192 0.337058
          continue 
        elif line[1] == 'n': # vertex normals
          # e.g. vn 0.992266 -0.033290 -0.119585
          (n1, n2, n3) = re.match(r'vn\s+([-\d\.e]+)\s+([-\d\.e]+)\s+([-\d\.e]+)', line).groups()
          normals.append( (float(n1), float(n2), float(n3)) )
        else:
          # e.g. v -0.317868 -0.000526 -0.251834
          (v1, v2, v3) = re.match(r'v\s+([-\d\.e]+)\s+([-\d\.e]+)\s+([-\d\.e]+)', line).groups()
          vertices.append( (float(v1), float(v2), float(v3)) )

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.uint32)
    normals = np.array(normals, dtype=np.float32)

    return Mesh(vertices, faces - 1, normals)

  def to_obj(self) -> bytes:
    """Return a string representing a .obj file."""
    objdata = []
    objdata += [ 'v {:.5f} {:.5f} {:.5f}'.format(*vertex) for vertex in self.vertices ]
    objdata += [ 'f {} {} {}'.format(*face) for face in (self.faces+1) ] # obj is 1 indexed
    objdata = '\n'.join(objdata) + '\n'
    return objdata.encode('utf8')

  @classmethod
  def from_ply(self, plydata:bytes) -> "Mesh":
    """
    Read from a binary representation formated as ply
    Note that this code is limited to parse the ply file saved by to_ply function
    It do not support ply file written by other softwares.
    """
    header, _, data = plydata.partition(b'end_header\n')
    lines = header.splitlines()
    assert lines[0] == b'ply'
    assert lines[1] == b'format binary_little_endian 1.0'
    vertexct = int(lines[2].decode('utf-8').split()[-1])
    trianglect = int(lines[6].decode('utf-8').split()[-1])

    # interpreate the data
    vertices = np.frombuffer(data, dtype=np.float32, count=3*vertexct, offset=0).reshape(vertexct, 3)
    faces = np.frombuffer(data, dtype=np.uint32, offset=vertexct*3*4).reshape(trianglect, 4)[:, 1:]
    return Mesh(vertices, faces, normals=None) 

  def to_ply(self) -> bytes:
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

  def viewer(self):
    # thanks to ChatGPT for making it easy to figure out
    # how to display VTK meshes.
    import vtk
    polydata = _create_vtk_mesh(self.vertices, self.faces)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    render_window.SetSize(1024, 1024)
      
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.3)  # Background color

    render_window.Render()
    render_window_interactor.Start()

  def trimesh(
    self,
    process:bool = False,
    validate:bool = False,
  ) -> "trimesh.Trimesh":
    """Convert zmesh.Mesh to a trimesh mesh."""
    import trimesh
    return trimesh.Trimesh(
      vertices=self.vertices,
      faces=self.faces,
      vertex_normals=(self.normals if hasattr(self, "normals") else None),
      process=process,
      validate=validate,
    )

  @classmethod
  def from_trimesh(kls, tmesh:"trimesh.Trimesh") -> "Mesh":
    return kls(vertices=tmesh.vertices, faces=tmesh.faces, normals=tmesh.vertex_normals)

  def save(self, filename:str):
    """
    Open supported file formats. 
    By default assumes the file is a Wavefront OBJ 
    unless the file extension says otherwise.

    Supported: obj, ply
    """
    with open(filename, "wb") as f:
      if filename.endswith(".ply"):
        f.write(self.to_ply())
      else:
        f.write(self.to_obj())

  def load(self, filename:str) -> "Mesh":
    """
    Save supported file formats. 
    By default assumes the file is a Wavefront OBJ 
    unless the file extension says otherwise.

    Supported: obj, ply
    """
    with open(filename, "rb") as f:
      if filename.endswith(".ply"):
        return Mesh.from_ply(f.read())
      else:
        return Mesh.from_obj(f.read())

def _create_vtk_mesh(vertices, faces):
  import vtk
  from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

  vtk_points = vtk.vtkPoints()
  vtk_points.SetData(numpy_to_vtk(vertices))
  
  polydata = vtk.vtkPolyData()
  polydata.SetPoints(vtk_points)
  
  vtk_faces = vtk.vtkCellArray()
  vtk_faces.SetCells(
    faces.shape[0], 
    numpy_to_vtkIdTypeArray(np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten())
  )

  polydata.SetPolys(vtk_faces)
  
  return polydata


