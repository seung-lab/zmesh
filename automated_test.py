import numpy as np

import zmesh

def test_executes():
  for dtype in (np.uint32, np.uint64):
    labels = np.zeros( (11,17,19), dtype=dtype)
    labels[1:-1, 1:-1, 1:-1] = 1

    mesher = zmesh.Mesher( (4,4,40) )
    mesher.mesh(labels)

    mesh = mesher.get_mesh(1, normals=False)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert mesh.normals is None

    mesh = mesher.get_mesh(1, normals=True)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert len(mesh.normals) > 0

def test_simplify():
  for dtype in (np.uint32, np.uint64):
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

def test_precomputed():
  for dtype in (np.uint32, np.uint64):
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

