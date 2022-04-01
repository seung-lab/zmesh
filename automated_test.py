import pytest
import numpy as np
import gzip

import zmesh

DTYPE = [ np.uint8, np.uint16, np.uint32, np.uint64 ]

@pytest.fixture
def connectomics_labels():
  with gzip.open('./connectomics.npy.gz', 'rb') as f:
    return np.load(f)

@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("close", [ False, True ])
def test_executes(dtype, close):
  labels = np.zeros( (11,17,19), dtype=dtype)
  labels[1:-1, 1:-1, 1:-1] = 1

  mesher = zmesh.Mesher( (4,4,40) )
  mesher.mesh(labels, close=close)

  mesh = mesher.get_mesh(1, normals=False)
  assert len(mesh.vertices) > 0
  assert len(mesh.faces) > 0
  assert mesh.normals is None

  mesh = mesher.get_mesh(1, normals=True)
  assert len(mesh.vertices) > 0
  assert len(mesh.faces) > 0
  assert len(mesh.normals) > 0

@pytest.mark.parametrize("dtype", DTYPE)
def test_simplify(dtype):
  labels = np.zeros( (11,17,19), dtype=dtype)
  labels[1:-1, 1:-1, 1:-1] = 1

  mesher = zmesh.Mesher( (4,4,40) )
  mesher.mesh(labels)

  mesh = mesher.get_mesh(1, normals=False)
  Nv = len(mesh.vertices)
  Nf = len(mesh.faces)
  
  mesh = mesher.simplify(mesh, reduction_factor=10, max_error=40)
  assert len(mesh) > 0
  assert len(mesh) < Nv 
  assert mesh.faces.shape[0] < Nf
  assert mesh.normals is None or mesh.normals.size == 0

  mesh = mesher.simplify(mesh, reduction_factor=10, max_error=40, compute_normals=True)
  assert len(mesh) > 0
  assert len(mesh) < Nv 
  assert mesh.faces.shape[0] < Nf
  assert mesh.normals.size > 0

@pytest.mark.parametrize("dtype", DTYPE)
def test_precomputed(dtype):
  labels = np.zeros( (11,17,19), dtype=dtype)
  labels[1:-1, 1:-1, 1:-1] = 1

  mesher = zmesh.Mesher( (4,4,40) )
  mesher.mesh(labels)
  mesh = mesher.get_mesh(1, normals=False)

  precomputed_mesh = mesh.to_precomputed()
  reconstituted = zmesh.Mesh.from_precomputed(precomputed_mesh)
  assert reconstituted == mesh

  mesh = mesher.get_mesh(1, normals=True)

  precomputed_mesh = mesh.to_precomputed()
  reconstituted = zmesh.Mesh.from_precomputed(precomputed_mesh)
  assert reconstituted != mesh # Precomputed doesn't preserve normals

def test_obj_import():
  labels = np.zeros( (11,17,19), dtype=np.uint32)
  labels[1:-1, 1:-1, 1:-1] = 1

  mesher = zmesh.Mesher( (4,4,40) )
  mesher.mesh(labels)
  mesh = mesher.get_mesh(1, normals=False)

  obj_str = mesh.to_obj()
  mesh2 = zmesh.Mesh.from_obj(obj_str)
  
  assert mesh == mesh2

def test_ply_import():
  labels = np.zeros( (11,17,19), dtype=np.uint32)
  labels[1:-1, 1:-1, 1:-1] = 1

  mesher = zmesh.Mesher( (4,4,40) )
  mesher.mesh(labels)
  mesh = mesher.get_mesh(1, normals=False)

  plydata = mesh.to_ply()
  mesh2 = zmesh.Mesh.from_ply(plydata)
  
  assert mesh == mesh2

# def test_meshes_remain_the_same(connectomics_labels):
#   mesher = zmesh.Mesher( (32,32,40) )
#   mesher.mesh(connectomics_labels)

#   for lbl in mesher.ids():
#     with open(f"./connectomics_npy_meshes/simplified/{lbl}.ply", "rb") as f:
#       old_mesh = zmesh.Mesh.from_ply(f.read())
#     new_mesh = mesher.get_mesh(lbl, normals=False, simplification_factor=100, max_simplification_error=40)
#     assert old_mesh == new_mesh
#     print(lbl, "ok")

