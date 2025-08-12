import pytest
import numpy as np
import gzip
import sys

import zmesh

DTYPE = [ np.uint8, np.uint16, np.uint32, np.uint64 ]

@pytest.fixture
def connectomics_labels():
  with gzip.open('./connectomics.npy.gz', 'rb') as f:
    return np.load(f)

@pytest.fixture
def fanc_label():
  with gzip.open('./fanc_bug.npy.gz', 'rb') as f:
    return np.load(f)[...,0]

@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("close", [ False, True ])
@pytest.mark.parametrize("order", [ 'C', 'F' ])
def test_executes_legacy(dtype, close, order):
  labels = np.zeros( (11,17,19), dtype=dtype, order=order)
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
@pytest.mark.parametrize("close", [ False, True ])
@pytest.mark.parametrize("order", [ 'C', 'F' ])
def test_executes(dtype, close, order):
  labels = np.zeros( (11,17,19), dtype=dtype, order=order)
  labels[1:-1, 1:-1, 1:-1] = 1

  mesher = zmesh.Mesher( (4,4,40) )
  mesher.mesh(labels, close=close)

  mesh = mesher.get(1, normals=False)
  assert len(mesh.vertices) > 0
  assert len(mesh.faces) > 0
  assert mesh.normals is None

  mesh = mesher.get(1, normals=True)
  assert len(mesh.vertices) > 0
  assert len(mesh.faces) > 0
  assert len(mesh.normals) > 0

@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("order", [ 'C', 'F' ])
def test_simplify(dtype, order):
  labels = np.zeros( (11,17,19), dtype=dtype, order=order)
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

  mesher.voxel_res = (1,1,1)
  mesh = mesher.get_mesh(1, normals=False)

  # ensure negative vertices work
  mesh.vertices -= 10000
  mesh = mesher.simplify(mesh, reduction_factor=2, max_error=40, compute_normals=True)
  assert len(mesh) > 0
  assert len(mesh) <= Nv 
  assert mesh.faces.shape[0] <= Nf
  assert mesh.normals.size > 0

  # check that upper limit of precision errors
  try:
    mesh.vertices[0] = 0
    mesh.vertices[1:] = 2**20
    mesh = mesher.simplify(mesh, reduction_factor=2, max_error=40, compute_normals=True)
    assert False
  except ValueError:
    pass

@pytest.mark.parametrize("dtype", DTYPE)
def test_precomputed_legacy(dtype):
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

@pytest.mark.parametrize("dtype", DTYPE)
def test_precomputed(dtype):
  labels = np.zeros( (11,17,19), dtype=dtype)
  labels[1:-1, 1:-1, 1:-1] = 1

  mesher = zmesh.Mesher( (4,4,40) )
  mesher.mesh(labels)
  mesh = mesher.get(1, normals=False)

  precomputed_mesh = mesh.to_precomputed()
  reconstituted = zmesh.Mesh.from_precomputed(precomputed_mesh)
  assert reconstituted == mesh

  mesh = mesher.get(1, normals=True)

  precomputed_mesh = mesh.to_precomputed()
  reconstituted = zmesh.Mesh.from_precomputed(precomputed_mesh)
  assert reconstituted != mesh # Precomputed doesn't preserve normals

def test_obj_import():
  labels = np.zeros( (11,17,19), dtype=np.uint32)
  labels[1:-1, 1:-1, 1:-1] = 1

  mesher = zmesh.Mesher( (4,4,40) )
  mesher.mesh(labels)
  mesh = mesher.get(1, normals=False)

  obj_str = mesh.to_obj()
  mesh2 = zmesh.Mesh.from_obj(obj_str)
  
  assert mesh == mesh2

def test_ply_import():
  labels = np.zeros( (11,17,19), dtype=np.uint32)
  labels[1:-1, 1:-1, 1:-1] = 1

  mesher = zmesh.Mesher( (4,4,40) )
  mesher.mesh(labels)
  mesh = mesher.get(1, normals=False)

  plydata = mesh.to_ply()
  mesh2 = zmesh.Mesh.from_ply(plydata)
  
  assert mesh == mesh2

def test_C_F_meshes_same_legacy(connectomics_labels):
  connectomics_labels = connectomics_labels[102:,31:,17:]

  fdata = np.asfortranarray(connectomics_labels)
  cdata = np.ascontiguousarray(connectomics_labels)

  f_mesher = zmesh.Mesher((1,1,1))
  f_mesher.mesh(fdata)

  c_mesher = zmesh.Mesher((1,1,1))
  c_mesher.mesh(cdata)

  cids = c_mesher.ids()
  cids.sort()
  fids = f_mesher.ids()
  fids.sort()
  assert cids == fids

  for label in c_mesher.ids()[:300]:
    c_mesh = c_mesher.get_mesh(label, normals=False, simplification_factor=0)
    f_mesh = f_mesher.get_mesh(label, normals=False, simplification_factor=0)
    assert np.isclose(c_mesh.vertices.mean(), f_mesh.vertices.mean())

@pytest.mark.parametrize("transpose", [True,False])
def test_fanc_bug(fanc_label, transpose):
  if transpose:
    fanc_label = fanc_label.T
  fdata = np.asfortranarray(fanc_label)
  cdata = np.ascontiguousarray(fanc_label)

  f_mesher = zmesh.Mesher((1,1,1))
  f_mesher.mesh(fdata)

  c_mesher = zmesh.Mesher((1,1,1))
  c_mesher.mesh(cdata)

  assert c_mesher.ids() == f_mesher.ids()

  for label in c_mesher.ids():
    c_mesh = c_mesher.get(label, normals=False, reduction_factor=0)
    f_mesh = f_mesher.get(label, normals=False, reduction_factor=0)
    assert np.isclose(c_mesh.vertices.mean(), f_mesh.vertices.mean())

@pytest.mark.parametrize("order", [ 'C', 'F' ])
def test_unsimplified_meshes_remain_the_same(connectomics_labels, order):
  if order == "C":
    connectomics_labels = np.ascontiguousarray(connectomics_labels)
  else:
    connectomics_labels = np.asfortranarray(connectomics_labels)

  mesher = zmesh.Mesher( (32,32,40) )
  mesher.mesh(connectomics_labels)

  for lbl in mesher.ids()[:300]:
    with gzip.open(f"./connectomics_npy_meshes/unsimplified/{lbl}.ply.gz", "rb") as f:
      old_mesh = zmesh.Mesh.from_ply(f.read())
    new_mesh = mesher.get_mesh(lbl, normals=False, simplification_factor=0, max_simplification_error=40)
    assert np.all(np.sort(old_mesh.vertices[old_mesh.faces], axis=0) == np.sort(new_mesh.vertices[new_mesh.faces], axis=0))
    print(lbl, "ok")


  mesher = zmesh.Mesher( (1,1,1) )
  mesher.mesh(connectomics_labels)

  mesher2 = zmesh.Mesher( (1,1,1) )
  mesher2.mesh(connectomics_labels)

  for lbl in mesher.ids()[:50]:
    old_mesh = mesher.get_mesh(lbl, normals=False, simplification_factor=0, max_simplification_error=40)
    new_mesh = mesher2.get(lbl, normals=False, reduction_factor=0, max_error=40)

    old_pts = old_mesh.vertices[old_mesh.faces]
    old_pts = old_pts.reshape(old_pts.shape[0] * old_pts.shape[1], 3)
    transposed_old = np.copy(old_pts)
    transposed_old[:,0] = old_pts[:,2]
    transposed_old[:,2] = old_pts[:,0]
    del old_pts
    transposed_old.sort(axis=0)

    new_pts = new_mesh.vertices[new_mesh.faces]
    new_pts = new_pts.reshape(new_pts.shape[0] * new_pts.shape[1], 3)
    new_pts.sort(axis=0)

    assert np.all(transposed_old == new_pts)
    print(lbl, "ok")  

# F order meshes are processed in a different order and so
# the simplifier produces a different mesh. Will have to add F order examples
# in order to test.
@pytest.mark.parametrize("order", [ 'C' ])
@pytest.mark.skipif(sys.platform != 'darwin', reason="Different implementations of unordered_map on different platforms have different iteration behavior. Only MacOS will match.")
def test_simplified_meshes_remain_the_same(connectomics_labels, order):
  if order == "C":
    connectomics_labels = np.ascontiguousarray(connectomics_labels)
  else:
    connectomics_labels = np.asfortranarray(connectomics_labels)

  mesher = zmesh.Mesher( (32,32,40) )
  mesher.mesh(connectomics_labels)

  for lbl in mesher.ids()[:300]:
    with gzip.open(f"./connectomics_npy_meshes/simplified/{lbl}.ply.gz", "rb") as f:
      old_mesh = zmesh.Mesh.from_ply(f.read())
    new_mesh = mesher.get_mesh(lbl, normals=False, simplification_factor=100, max_simplification_error=40)

    f1 = np.sort(np.sort(old_mesh.faces, axis=0), axis=1)
    f2 = np.sort(np.sort(new_mesh.faces, axis=0), axis=1)
    v1 = np.sort(old_mesh.vertices[f1], axis=0)
    v2 = np.sort(new_mesh.vertices[f2], axis=0)
    assert np.all(np.isclose(v1, v2))
    print(lbl, "ok")


@pytest.mark.parametrize("reduction_factor", [2,5,10])
def test_min_error_skip(reduction_factor):
  for order in ['C','F']:
    labels = np.zeros((11, 17, 19), dtype=np.uint8, order=order)
    labels[1:-1, 1:-1, 1:-1] = 1

    mesher = zmesh.Mesher((4, 4, 40))
    mesher.mesh(labels)

    original_mesh = mesher.get_mesh(1, normals=False)
    mesh = mesher.simplify(
        original_mesh, 
        reduction_factor=reduction_factor, 
        max_error=40, 
        min_error=0,
        compute_normals=True
    )
    factor = len(original_mesh.faces) / len(mesh.faces)
    assert abs(factor - reduction_factor) < 1


def test_chunk_shape():
  labels = np.zeros( (10, 10, 10), dtype=np.uint32)
  labels[1:-1, 1:-1, 1:-1] = 1

  mesher = zmesh.Mesher( (1, 1, 1) )
  mesher.mesh(labels)
  mesh = mesher.get_mesh(1, normals=False, simplification_factor=0, max_simplification_error=100)

  meshes = zmesh.chunk_mesh(
    mesh,
    [5.,5.,5.],
  )

  assert len(meshes) == 8
  assert not any([
    m.empty() for m in meshes.values()
  ])

def test_delete_unreference_vertices():
  vertices = [
    [0,0,0],
    [0,1,0],
    [1,0,0],
    [5,5,5],
    [7,7,7],
    [8,7,7],
    [7,8,8],
  ]
  faces = [[0,1,2],[4,5,6]]

  mesh = zmesh.Mesh(vertices=vertices, faces=faces, normals=None)
  res = mesh.remove_unreferenced_vertices()

  ans_verts = vertices[:3] + vertices[4:]
  ans_faces = [[0,1,2],[3,4,5]]

  assert res.vertices.shape == (6,3)
  assert res.faces.shape == (2,3)
  assert np.all(res.vertices == np.array(ans_verts))
  assert np.all(res.faces == np.array(ans_faces))

def test_chunk_mesh_triangle():
  vertices = [
    [0,0,0],
    [0,1,0],
    [1,0,0],
  ]
  faces = [[0,1,2]]

  mesh = zmesh.Mesh(vertices=vertices, faces=faces, normals=None)

  meshes = zmesh.chunk_mesh(mesh, [.5,.5,.5])

  meshes = [ m for m in meshes.values() if not m.empty() ]

  assert len(meshes) == 3

  m = zmesh.Mesh.concatenate(*meshes).consolidate()

  assert m.vertices.shape[0] == 6

  assert [0,0,0] in m.vertices
  assert [1,0,0] in m.vertices
  assert [0,1,0] in m.vertices
  assert [0,0.5,0] in m.vertices
  assert [0.5,0,0] in m.vertices
  assert [0.5,0.5,0] in m.vertices

  










